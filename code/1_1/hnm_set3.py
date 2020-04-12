import argparse
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
from apex import amp
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import random
import re
import json
from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel, BertConfig, get_linear_schedule_with_warmup


class TFQADataset(Dataset):
    def __init__(self, id_list):
        self.id_list=id_list 
  
    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        return self.id_list[index]


class Collator(object):
    def __init__(self, data_dict, tokenizer, max_seq_len=384, max_question_len=64):
        self.data_dict = data_dict
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_question_len = max_question_len

    def _get_input_ids(self, doc_id, candidate_index):
        data = self.data_dict[doc_id]
        question_tokens = self.tokenizer.tokenize(data['question_text'])[:self.max_question_len]
        doc_words = data['document_text'].split()

        max_answer_tokens = self.max_seq_len-len(question_tokens)-3 # [CLS],[SEP],[SEP]
        candidate = data['long_answer_candidates'][candidate_index]
        candidate_start = candidate['start_token']
        candidate_end = candidate['end_token']
        candidate_words = doc_words[candidate_start:candidate_end]         

        words_to_tokens_index = []
        candidate_tokens = []
        for i, word in enumerate(candidate_words):
            words_to_tokens_index.append(len(candidate_tokens))
            if re.match(r'<.+>', word):  # remove paragraph tag
                continue
            tokens = self.tokenizer.tokenize(word)
            if len(candidate_tokens)+len(tokens) > max_answer_tokens:
                break
            for token in tokens:
                candidate_tokens.append(token)

        input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + candidate_tokens + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        return input_ids, len(input_ids)
        
    def __call__(self, batch_ids):
        batch_size = len(batch_ids)

        batch_input_ids_temp = []
        batch_seq_len = []

        for i, (doc_id, candidate_index) in enumerate(batch_ids):
            input_ids, seq_len = self._get_input_ids(doc_id, candidate_index)
            batch_input_ids_temp.append(input_ids)
            batch_seq_len.append(seq_len)

        batch_max_seq_len = max(batch_seq_len)
        batch_input_ids = np.zeros((batch_size, batch_max_seq_len), dtype=np.int64)
        batch_token_type_ids = np.ones((batch_size, batch_max_seq_len), dtype=np.int64)

        for i in range(batch_size):
            input_ids = batch_input_ids_temp[i]
            batch_input_ids[i, :len(input_ids)] = input_ids
            batch_token_type_ids[i, :len(input_ids)] = [0 if k<=input_ids.index(102) else 1 for k in range(len(input_ids))]

        batch_attention_mask = batch_input_ids > 0

        return torch.from_numpy(batch_input_ids), torch.from_numpy(batch_attention_mask), torch.from_numpy(batch_token_type_ids)


class BertForQuestionAnswering(BertPreTrainedModel):
    """BERT model for QA and classification tasks.
    
    Parameters
    ----------
    config : transformers.BertConfig. Configuration class for BERT.
        
    Returns
    -------
    start_logits : torch.Tensor with shape (batch_size, sequence_size).
        Starting scores of each tokens.
    end_logits : torch.Tensor with shape (batch_size, sequence_size).
        Ending scores of each tokens.
    classifier_logits : torch.Tensor with shape (batch_size, num_classes).
        Classification scores of each labels.
    """

    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)  # start/end
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # predict start & end position
        qa_logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
    
        # classification
        pooled_output = self.dropout(pooled_output)
        classifier_logits = self.classifier(pooled_output)

        return start_logits, end_logits, classifier_logits


# prepare input
json_dir = '../../input/simplified-nq-train.jsonl'

id_candidate_list = []
id_candidate_len_list = [] 
id_list = []
data_dict = {}
max_data_low = 150000
max_data_high = 225000
with open(json_dir) as f:
    for n, line in tqdm(enumerate(f)):
        if n >= max_data_low and n < max_data_high:
            data = json.loads(line)
            data_id = data['example_id']
            id_list.append(data_id)

            is_pos = False
            annotations = data['annotations'][0]
            if annotations['yes_no_answer'] == 'YES':
                is_pos = True
            elif annotations['yes_no_answer'] == 'NO':
                is_pos = True
            elif annotations['short_answers']:
                is_pos = True
            elif annotations['long_answer']['candidate_index'] != -1:
                is_pos = True

            # initialize data_dict
            data_dict[data_id] = {
                                  'document_text': data['document_text'],
                                  'question_text': data['question_text'], 
                                  'long_answer_candidates': data['long_answer_candidates'],                
                                 }
        
            question_len = len(data['question_text'].split())
            #
            for i in range(len(data['long_answer_candidates'])):
                if is_pos: 
                    if i != data['annotations'][0]['long_answer']['candidate_index']:
                        id_candidate_list.append((data_id, i))
                        id_candidate_len_list.append(question_len+data['long_answer_candidates'][i]['end_token']-data['long_answer_candidates'][i]['start_token'])
                else:
                    id_candidate_list.append((data_id, i))
                    id_candidate_len_list.append(question_len+data['long_answer_candidates'][i]['end_token']-data['long_answer_candidates'][i]['start_token'])


print(len(id_list), len(id_candidate_list))

id_candidate_len_list = np.array(id_candidate_len_list)
sorted_index = np.argsort(id_candidate_len_list)
id_candidate_list_sorted = []
for i in range(len(id_candidate_list)):
    id_candidate_list_sorted.append(id_candidate_list[sorted_index[i]])


# hyperparameters
max_seq_len = 384
max_question_len = 64
learning_rate = 0.00005
batch_size = 960
num_epoch = 1


# build model
model_path = '../../huggingface_pretrained/bert-base-uncased/'
config = BertConfig.from_pretrained(model_path)
config.num_labels = 5
tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
model = BertForQuestionAnswering.from_pretrained('weights/epoch1/', config=config)


model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)


# testing

# iterator for testing
test_datagen = TFQADataset(id_list=id_candidate_list_sorted)
test_collate = Collator(data_dict=data_dict, 
                        tokenizer=tokenizer, 
                        max_seq_len=max_seq_len, 
                        max_question_len=max_question_len)
test_generator = DataLoader(dataset=test_datagen,
                            collate_fn=test_collate,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=16,
                            pin_memory=True)


model.eval()
test_prob = np.zeros((len(id_candidate_list_sorted),),dtype=np.float32) # the negative class
for j,(batch_input_ids, batch_attention_mask, batch_token_type_ids) in tqdm(enumerate(test_generator)):
    with torch.no_grad():
        start = j*batch_size
        end = start+batch_size
        if j == len(test_generator)-1:
            end = len(test_generator.dataset)
        batch_input_ids = batch_input_ids.cuda()
        batch_attention_mask = batch_attention_mask.cuda()
        batch_token_type_ids = batch_token_type_ids.cuda()
        logits1, logits2, logits3 = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
        test_prob[start:end] += F.softmax(logits3,dim=1).cpu().data.numpy()[:,0]
test_prob = 1.0 - test_prob # only store the positive


# initialize
distribution_dict = {}
for doc_id in id_list:
    distribution_dict[doc_id] = {'candidate_index_list': [], 'prob_list': []}

# from cadidates to document
for i, (doc_id, candidate_index) in tqdm(enumerate(id_candidate_list_sorted)):
    distribution_dict[doc_id]['candidate_index_list'].append(candidate_index)
    distribution_dict[doc_id]['prob_list'].append(test_prob[i])

import pickle
with open('distribution_dict3.pickle', 'wb') as f:
    pickle.dump(distribution_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


