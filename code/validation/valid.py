import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
import re
import json
from transformers import BertTokenizer, BertModel, BertPreTrainedModel, BertConfig, AlbertTokenizer, AlbertModel, AlbertPreTrainedModel, AlbertConfig
import time
from apex import amp


def reduce1(n_candidate=10, th_candidate=0.2):

    class TFQADataset(Dataset):
        def __init__(self, id_list):
            self.id_list=id_list 
        def __len__(self):
            return len(self.id_list)
        def __getitem__(self, index):
            return self.id_list[index]

    class Collator(object):
        def __init__(self, data_dict, new_token_dict, tokenizer, max_seq_len=384, max_question_len=64):
            self.data_dict = data_dict
            self.new_token_dict = new_token_dict
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
            # Loop through to add html tokens as new tokens here
            for i, word in enumerate(candidate_words):
                if re.match(r'<.+>', word):
                    if word in self.new_token_dict: 
                        candidate_words[i] = self.new_token_dict[word]
                    else:
                        candidate_words[i] = '<'     

            words_to_tokens_index = []
            candidate_tokens = []
            for i, word in enumerate(candidate_words):
                words_to_tokens_index.append(len(candidate_tokens))
                tokens = self.tokenizer.tokenize(word)
                if len(candidate_tokens)+len(tokens) > max_answer_tokens: # token length cannot be longer than the global max length (360)
                    break
                for token in tokens:
                    candidate_tokens.append(token)

            input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + candidate_tokens + ['[SEP]']
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

            return input_ids, len(input_ids)
        
        def __call__(self, batch_ids):
            batch_size = len(batch_ids)

            # pre-compute all the input within the batch without padding to determine the actual batch max sequence length
            batch_input_ids_temp = []
            batch_seq_len = []

            for i, (doc_id, candidate_index) in enumerate(batch_ids):
                input_ids, seq_len = self._get_input_ids(doc_id, candidate_index)
                batch_input_ids_temp.append(input_ids)
                batch_seq_len.append(seq_len)

            batch_max_seq_len = max(batch_seq_len) # set max sequence length to be the maximun length in a batch, to save computation 
            batch_input_ids = np.zeros((batch_size, batch_max_seq_len), dtype=np.int64)
            batch_token_type_ids = np.ones((batch_size, batch_max_seq_len), dtype=np.int64)

            for i in range(batch_size):
                input_ids = batch_input_ids_temp[i]
                batch_input_ids[i, :len(input_ids)] = input_ids
                batch_token_type_ids[i, :len(input_ids)] = [0 if k<=input_ids.index(102) else 1 for k in range(len(input_ids))]

            batch_attention_mask = batch_input_ids > 0

            return torch.from_numpy(batch_input_ids),torch.from_numpy(batch_attention_mask),torch.from_numpy(batch_token_type_ids)

    # https://www.kaggle.com/sakami/tfqa-pytorch-baseline
    class BertForQuestionAnswering(BertPreTrainedModel):

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
    json_dir = '../../input/natural_questions/simplified-nq-valid.jsonl'

    # The id_candidate_list keeps all the combination of document ids and their candidates. So we essentially run predictions on all the candidates.
    id_candidate_list = []
    # Store the lengths of each candidate for rearranging based on candidate length, this can help improve inference speed significantly.
    id_candidate_len_list = [] 
    # Keep a dictionary for length checking.
    id_candidate_len_dict = {}
    # list of document ids.
    id_list = []
    # Keeps the texts and candidates.
    data_dict = {}
    # for debugging only
    max_data = 9999999999
    with open(json_dir) as f:
        for n, line in tqdm(enumerate(f)):
            if n > max_data:
                break
            data = json.loads(line)
            data_id = data['example_id']
            id_list.append(data_id)

            # initialize data_dict
            data_dict[data_id] = {
                                  'document_text': data['document_text'],
                                  'question_text': data['question_text'], 
                                  'long_answer_candidates': data['long_answer_candidates'],                
                                 }
        
            question_len = len(data['question_text'].split())

            # We use the wite space tokenzied version to estimate candidate length here.
            for i in range(len(data['long_answer_candidates'])):
                id_candidate_list.append((data_id, i))
                candidate_len = question_len+data['long_answer_candidates'][i]['end_token']-data['long_answer_candidates'][i]['start_token']
                id_candidate_len_list.append(candidate_len)
                id_candidate_len_dict[(data_id, i)] = candidate_len

    print(len(id_candidate_list))

    # Sort based on the length of each candidate.
    id_candidate_len_list = np.array(id_candidate_len_list)
    sorted_index = np.argsort(id_candidate_len_list)
    id_candidate_list_sorted = []
    for i in range(len(id_candidate_list)):
        id_candidate_list_sorted.append(id_candidate_list[sorted_index[i]])


    # hyperparameters
    max_seq_len = 360
    max_question_len = 64
    batch_size = 768


    # build model
    model_path = '../bert-base-uncased_1/model/'
    config = BertConfig.from_pretrained(model_path)
    config.num_labels = 5
    config.vocab_size = 30531
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
    model = BertForQuestionAnswering.from_pretrained('../bert-base-uncased_1/weights/epoch2/', config=config)

    # add new tokens
    new_token_dict = {
                      '<P>':'qw1',
                      '<Table>':'qw2',
                      '<Tr>':'qw3',
                      '<Ul>':'qw4',
                      '<Ol>':'qw5',
                      '<Fl>':'qw6',
                      '<Li>':'qw7',
                      '<Dd>':'qw8',
                      '<Dt>':'qw9',
                     }
    new_token_list = [
                      'qw1',
                      'qw2',
                      'qw3',
                      'qw4',
                      'qw5',
                      'qw6',
                      'qw7',
                      'qw8',
                      'qw9',
                     ]

    num_added_toks = tokenizer.add_tokens(new_token_list)
    print('We have added', num_added_toks, 'tokens')
    model.resize_token_embeddings(len(tokenizer))


    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)


    # testing

    # iterator for testing
    test_datagen = TFQADataset(id_list=id_candidate_list_sorted)
    test_collate = Collator(data_dict=data_dict, 
                            new_token_dict=new_token_dict,
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
    test_prob3 = np.zeros((len(id_candidate_list_sorted),5),dtype=np.float32) # class
    for j,(batch_input_ids, batch_attention_mask, batch_token_type_ids) in tqdm(enumerate(test_generator)):
        with torch.no_grad():
            start = j*batch_size
            end = start+batch_size
            if j == len(test_generator)-1:
                end = len(test_generator.dataset)
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            batch_token_type_ids = batch_token_type_ids.cuda()
            # We don't need the span output here.
            _, _, logits3 = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
            test_prob3[start:end] += F.softmax(logits3,dim=1).cpu().data.numpy()


    # initialize a temp dictionary
    temp_dict = {}
    for doc_id in id_list:
        temp_dict[doc_id] = np.zeros((len(data_dict[doc_id]['long_answer_candidates']),),dtype=np.float32)

    # input long answer probs into the temp dictionary
    for i, (doc_id, candidate_index) in tqdm(enumerate(id_candidate_list_sorted)):
        temp_dict[doc_id][candidate_index] = 1.0 - test_prob3[i,0] # 1-no_answer_score

    # get list of survived id-candidates
    id_candidate_list1 = []
    id_candidate_len_list1 = []
    for doc_id in tqdm(id_list):
        long_prob_array = temp_dict[doc_id].copy()
        sorted_index = np.argsort(long_prob_array)[::-1]
        count = 0
        for n in range(len(sorted_index)):
            if count>=n_candidate:
                break
            else:
                if temp_dict[doc_id][sorted_index[n]]>th_candidate:
                    id_candidate_list1.append((doc_id, sorted_index[n]))
                    id_candidate_len_list1.append(id_candidate_len_dict[(doc_id, sorted_index[n])])
                    count += 1

    # sort and return
    sorted_index = np.argsort(id_candidate_len_list1)
    id_candidate_list_sorted1 = []
    for i in range(len(id_candidate_list1)):
        id_candidate_list_sorted1.append(id_candidate_list1[sorted_index[i]])

    print(len(id_candidate_list_sorted1))

    return data_dict, id_list, id_candidate_len_dict, id_candidate_list_sorted1


def reduce2(data_dict, id_list, id_candidate_len_dict, id_candidate_list_sorted, n_candidate=10, th_candidate=0.2):

    class TFQADataset(Dataset):
        def __init__(self, id_list):
            self.id_list=id_list 
        def __len__(self):
            return len(self.id_list)
        def __getitem__(self, index):
            return self.id_list[index]

    class Collator(object):
        def __init__(self, data_dict, new_token_dict, tokenizer, max_seq_len=384, max_question_len=64):
            self.data_dict = data_dict
            self.new_token_dict = new_token_dict
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
            for i, word in enumerate(candidate_words):
                if re.match(r'<.+>', word):
                    if word in self.new_token_dict: 
                        candidate_words[i] = self.new_token_dict[word]
                    else:
                        candidate_words[i] = '<'     

            words_to_tokens_index = []
            candidate_tokens = []
            for i, word in enumerate(candidate_words):
                words_to_tokens_index.append(len(candidate_tokens))
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

            return torch.from_numpy(batch_input_ids),torch.from_numpy(batch_attention_mask),torch.from_numpy(batch_token_type_ids)


    class BertForQuestionAnswering(BertPreTrainedModel):

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


    # hyperparameters
    max_seq_len = 360
    max_question_len = 64
    batch_size = 384


    # build model
    model_path = '../bert-large-uncased_4/model/'
    config = BertConfig.from_pretrained(model_path)
    config.num_labels = 5
    config.vocab_size = 30531
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
    model = BertForQuestionAnswering.from_pretrained('../bert-large-uncased_4/weights/epoch3/', config=config)

    # add new tokens
    new_token_dict = {
                      '<P>':'qw1',
                      '<Table>':'qw2',
                      '<Tr>':'qw3',
                      '<Ul>':'qw4',
                      '<Ol>':'qw5',
                      '<Fl>':'qw6',
                      '<Li>':'qw7',
                      '<Dd>':'qw8',
                      '<Dt>':'qw9',
                     }
    new_token_list = [
                      'qw1',
                      'qw2',
                      'qw3',
                      'qw4',
                      'qw5',
                      'qw6',
                      'qw7',
                      'qw8',
                      'qw9',
                     ]

    num_added_toks = tokenizer.add_tokens(new_token_list)
    print('We have added', num_added_toks, 'tokens')
    model.resize_token_embeddings(len(tokenizer))


    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)


    # testing

    # iterator for testing
    test_datagen = TFQADataset(id_list=id_candidate_list_sorted)
    test_collate = Collator(data_dict=data_dict, 
                            new_token_dict=new_token_dict,
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
    test_prob3 = np.zeros((len(id_candidate_list_sorted),5),dtype=np.float32) # class
    for j,(batch_input_ids, batch_attention_mask, batch_token_type_ids) in tqdm(enumerate(test_generator)):
        with torch.no_grad():
            start = j*batch_size
            end = start+batch_size
            if j == len(test_generator)-1:
                end = len(test_generator.dataset)
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            batch_token_type_ids = batch_token_type_ids.cuda()
            _, _, logits3 = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
            test_prob3[start:end] += F.softmax(logits3,dim=1).cpu().data.numpy()


    # initialize a temp dictionary
    temp_dict = {}
    for doc_id in id_list:
        temp_dict[doc_id] = np.zeros((len(data_dict[doc_id]['long_answer_candidates']),),dtype=np.float32)

    # input long answer probs into the temp dictionary
    for i, (doc_id, candidate_index) in tqdm(enumerate(id_candidate_list_sorted)):
        temp_dict[doc_id][candidate_index] = 1.0 - test_prob3[i,0] # 1-no_answer_score

    # get list of survived id-candidates
    id_candidate_list1 = []
    id_candidate_len_list1 = []
    for doc_id in tqdm(id_list):
        long_prob_array = temp_dict[doc_id].copy()
        sorted_index = np.argsort(long_prob_array)[::-1]
        count = 0
        for n in range(len(sorted_index)):
            if count>=n_candidate:
                break
            else:
                if temp_dict[doc_id][sorted_index[n]]>th_candidate:
                    id_candidate_list1.append((doc_id, sorted_index[n]))
                    id_candidate_len_list1.append(id_candidate_len_dict[(doc_id, sorted_index[n])])
                    count += 1

    # sort and return
    sorted_index = np.argsort(id_candidate_len_list1)
    id_candidate_list_sorted1 = []
    for i in range(len(id_candidate_list1)):
        id_candidate_list_sorted1.append(id_candidate_list1[sorted_index[i]])

    print(len(id_candidate_list_sorted1))

    return id_candidate_list_sorted1


def bert_large_predict(data_dict, id_list, id_candidate_len_dict, id_candidate_list_sorted, model_dir, word_len):

    class TFQADataset(Dataset):
        def __init__(self, id_list):
            self.id_list=id_list 
        def __len__(self):
            return len(self.id_list)
        def __getitem__(self, index):
            return self.id_list[index]

    class Collator(object):
        def __init__(self, data_dict, new_token_dict, tokenizer, max_seq_len=384, max_question_len=64):
            self.data_dict = data_dict
            self.new_token_dict = new_token_dict
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
            for i, word in enumerate(candidate_words):
                if re.match(r'<.+>', word):
                    if word in self.new_token_dict: 
                        candidate_words[i] = self.new_token_dict[word]
                    else:
                        candidate_words[i] = '<'     

            words_to_tokens_index = []
            candidate_tokens = []
            for i, word in enumerate(candidate_words):
                words_to_tokens_index.append(len(candidate_tokens))
                tokens = self.tokenizer.tokenize(word)
                if len(candidate_tokens)+len(tokens) > max_answer_tokens:
                    break
                for token in tokens:
                    candidate_tokens.append(token)

            input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + candidate_tokens + ['[SEP]']
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

            return input_ids, words_to_tokens_index, len(input_ids), len(question_tokens)+2
        
        def __call__(self, batch_ids):
            batch_size = len(batch_ids)

            batch_input_ids_temp = []
            batch_seq_len = []

            batch_offset = []
            batch_words_to_tokens_index = []

            for i, (doc_id, candidate_index) in enumerate(batch_ids):
                input_ids, words_to_tokens_index, seq_len, offset = self._get_input_ids(doc_id, candidate_index)
                batch_input_ids_temp.append(input_ids)
                batch_seq_len.append(seq_len)
                batch_offset.append(offset)
                batch_words_to_tokens_index.append(words_to_tokens_index)

            batch_max_seq_len = max(batch_seq_len)
            batch_input_ids = np.zeros((batch_size, batch_max_seq_len), dtype=np.int64)
            batch_token_type_ids = np.ones((batch_size, batch_max_seq_len), dtype=np.int64)

            for i in range(batch_size):
                input_ids = batch_input_ids_temp[i]
                batch_input_ids[i, :len(input_ids)] = input_ids
                batch_token_type_ids[i, :len(input_ids)] = [0 if k<=input_ids.index(102) else 1 for k in range(len(input_ids))]

            batch_attention_mask = batch_input_ids > 0

            return torch.from_numpy(batch_input_ids), torch.from_numpy(batch_attention_mask), torch.from_numpy(batch_token_type_ids), batch_words_to_tokens_index, batch_offset, batch_max_seq_len


    class BertForQuestionAnswering(BertPreTrainedModel):

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


    # hyperparameters
    max_seq_len = 360
    max_question_len = 64
    batch_size = 384


    # build model
    model_path = '../bert-large-uncased_4/model/'
    config = BertConfig.from_pretrained(model_path)
    config.num_labels = 5
    config.vocab_size = 30531
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
    model = BertForQuestionAnswering.from_pretrained(model_dir, config=config)

    # add new tokens
    new_token_dict = {
                      '<P>':'qw1',
                      '<Table>':'qw2',
                      '<Tr>':'qw3',
                      '<Ul>':'qw4',
                      '<Ol>':'qw5',
                      '<Fl>':'qw6',
                      '<Li>':'qw7',
                      '<Dd>':'qw8',
                      '<Dt>':'qw9',
                     }
    new_token_list = [
                      'qw1',
                      'qw2',
                      'qw3',
                      'qw4',
                      'qw5',
                      'qw6',
                      'qw7',
                      'qw8',
                      'qw9',
                     ]

    num_added_toks = tokenizer.add_tokens(new_token_list)
    print('We have added', num_added_toks, 'tokens')
    model.resize_token_embeddings(len(tokenizer))


    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)


    # testing

    # iterator for testing
    test_datagen = TFQADataset(id_list=id_candidate_list_sorted)
    test_collate = Collator(data_dict=data_dict, 
                            new_token_dict=new_token_dict,
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
    list_offset = []
    list_words_to_tokens_index = []
    test_prob1 = np.zeros((len(id_candidate_list_sorted),max_seq_len),dtype=np.float32) # start
    test_prob2 = np.zeros((len(id_candidate_list_sorted),max_seq_len),dtype=np.float32) # end
    test_prob3 = np.zeros((len(id_candidate_list_sorted),5),dtype=np.float32) # class
    for j,(batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_words_to_tokens_index, batch_offset, batch_max_seq_len) in tqdm(enumerate(test_generator)):
        with torch.no_grad():
            start = j*batch_size
            end = start+batch_size
            if j == len(test_generator)-1:
                end = len(test_generator.dataset)
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            batch_token_type_ids = batch_token_type_ids.cuda()
            logits1, logits2, logits3 = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
            test_prob1[start:end, :batch_max_seq_len] += F.softmax(logits1,dim=1).cpu().data.numpy()
            test_prob2[start:end, :batch_max_seq_len] += F.softmax(logits2,dim=1).cpu().data.numpy()
            test_prob3[start:end] += F.softmax(logits3,dim=1).cpu().data.numpy()
            list_words_to_tokens_index += batch_words_to_tokens_index
            list_offset += batch_offset

    # From token-level to word-level span predictions. Use the first token of each word for word-level representation.
    test_word_prob1 = np.zeros((len(id_candidate_list_sorted),word_len),dtype=np.float32) # start
    test_word_prob2 = np.zeros((len(id_candidate_list_sorted),word_len),dtype=np.float32) # end
    for i in range(len(id_candidate_list_sorted)):
        for j in range(len(list_words_to_tokens_index[i])):
            test_word_prob1[i,j] = test_prob1[i, list_words_to_tokens_index[i][j]+list_offset[i]]
            test_word_prob2[i,j] = test_prob2[i, list_words_to_tokens_index[i][j]+list_offset[i]]

    return test_word_prob1, test_word_prob2, test_prob3


def albert_predict(data_dict, id_list, id_candidate_len_dict, id_candidate_list_sorted, model_dir, word_len):

    class TFQADataset(Dataset):
        def __init__(self, id_list):
            self.id_list=id_list 
        def __len__(self):
            return len(self.id_list)
        def __getitem__(self, index):
            return self.id_list[index]

    class Collator(object):
        def __init__(self, data_dict, new_token_dict, tokenizer, max_seq_len=384, max_question_len=64):
            self.data_dict = data_dict
            self.new_token_dict = new_token_dict
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
            for i, word in enumerate(candidate_words):
                if re.match(r'<.+>', word):
                    if word in self.new_token_dict: 
                        candidate_words[i] = self.new_token_dict[word]
                    else:
                        candidate_words[i] = 'qw99'    

            words_to_tokens_index = []
            tokens_to_words_index = []
            candidate_tokens = []
            for i, word in enumerate(candidate_words):
                words_to_tokens_index.append(len(candidate_tokens))
                tokens = self.tokenizer.tokenize(word)
                if len(candidate_tokens)+len(tokens) > max_answer_tokens:
                    break
                for token in tokens:
                    tokens_to_words_index.append(i)
                    candidate_tokens.append(token)

            input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + candidate_tokens + ['[SEP]']
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

            return input_ids, words_to_tokens_index, len(input_ids), len(question_tokens)+2
        
        def __call__(self, batch_ids):
            batch_size = len(batch_ids)

            batch_input_ids_temp = []
            batch_seq_len = []

            batch_offset = []
            batch_words_to_tokens_index = []

            for i, (doc_id, candidate_index) in enumerate(batch_ids):
                input_ids, words_to_tokens_index, seq_len, offset = self._get_input_ids(doc_id, candidate_index)
                batch_input_ids_temp.append(input_ids)
                batch_seq_len.append(seq_len)
                batch_offset.append(offset)
                batch_words_to_tokens_index.append(words_to_tokens_index)

            batch_max_seq_len = max(batch_seq_len)
            batch_input_ids = np.zeros((batch_size, batch_max_seq_len), dtype=np.int64)
            batch_token_type_ids = np.ones((batch_size, batch_max_seq_len), dtype=np.int64)

            for i in range(batch_size):
                input_ids = batch_input_ids_temp[i]
                batch_input_ids[i, :len(input_ids)] = input_ids
                batch_token_type_ids[i, :len(input_ids)] = [0 if k<=input_ids.index(3) else 1 for k in range(len(input_ids))]

            batch_attention_mask = batch_input_ids > 0

            return torch.from_numpy(batch_input_ids), torch.from_numpy(batch_attention_mask), torch.from_numpy(batch_token_type_ids), batch_words_to_tokens_index, batch_offset, batch_max_seq_len


    class AlbertForQuestionAnswering(AlbertPreTrainedModel):

        def __init__(self, config):
            super(AlbertForQuestionAnswering, self).__init__(config)
            self.albert = AlbertModel(config)
            self.qa_outputs = nn.Linear(config.hidden_size, 2)  # start/end
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            self.init_weights()

        def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
            outputs = self.albert(input_ids,
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


    # hyperparameters
    max_seq_len = 360
    max_question_len = 64
    batch_size = 128


    # build model
    model_path = '../albert-xxlarge-v2_2/model/'
    config = AlbertConfig.from_pretrained(model_path)
    config.num_labels = 5
    config.vocab_size = 30010
    tokenizer = AlbertTokenizer.from_pretrained(model_path, do_lower_case=True)
    model = AlbertForQuestionAnswering.from_pretrained(model_dir, config=config)

    # add new tokens
    new_token_dict = {
                      '<P>':'qw1',
                      '<Table>':'qw2',
                      '<Tr>':'qw3',
                      '<Ul>':'qw4',
                      '<Ol>':'qw5',
                      '<Fl>':'qw6',
                      '<Li>':'qw7',
                      '<Dd>':'qw8',
                      '<Dt>':'qw9',
                     }
    new_token_list = [
                      'qw1',
                      'qw2',
                      'qw3',
                      'qw4',
                      'qw5',
                      'qw6',
                      'qw7',
                      'qw8',
                      'qw9',
                      'qw99',
                     ]

    num_added_toks = tokenizer.add_tokens(new_token_list)
    print('We have added', num_added_toks, 'tokens')
    model.resize_token_embeddings(len(tokenizer))


    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)


    # testing

    # iterator for testing
    test_datagen = TFQADataset(id_list=id_candidate_list_sorted)
    test_collate = Collator(data_dict=data_dict, 
                            new_token_dict=new_token_dict,
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
    list_offset = []
    list_words_to_tokens_index = []
    test_prob1 = np.zeros((len(id_candidate_list_sorted),max_seq_len),dtype=np.float32) # start
    test_prob2 = np.zeros((len(id_candidate_list_sorted),max_seq_len),dtype=np.float32) # end
    test_prob3 = np.zeros((len(id_candidate_list_sorted),5),dtype=np.float32) # class
    for j,(batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_words_to_tokens_index, batch_offset, batch_max_seq_len) in tqdm(enumerate(test_generator)):
        with torch.no_grad():
            start = j*batch_size
            end = start+batch_size
            if j == len(test_generator)-1:
                end = len(test_generator.dataset)
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            batch_token_type_ids = batch_token_type_ids.cuda()
            logits1, logits2, logits3 = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
            test_prob1[start:end, :batch_max_seq_len] += F.softmax(logits1,dim=1).cpu().data.numpy()
            test_prob2[start:end, :batch_max_seq_len] += F.softmax(logits2,dim=1).cpu().data.numpy()
            test_prob3[start:end] += F.softmax(logits3,dim=1).cpu().data.numpy()
            list_words_to_tokens_index += batch_words_to_tokens_index
            list_offset += batch_offset

    test_word_prob1 = np.zeros((len(id_candidate_list_sorted),word_len),dtype=np.float32) # start
    test_word_prob2 = np.zeros((len(id_candidate_list_sorted),word_len),dtype=np.float32) # end
    for i in range(len(id_candidate_list_sorted)):
        for j in range(len(list_words_to_tokens_index[i])):
            test_word_prob1[i,j] = test_prob1[i, list_words_to_tokens_index[i][j]+list_offset[i]]
            test_word_prob2[i,j] = test_prob2[i, list_words_to_tokens_index[i][j]+list_offset[i]]


    return test_word_prob1, test_word_prob2, test_prob3


# This function performs a full prediction on the validation set using a fast model (bert-base) to reduce the candidates for larger model predictions.
# Propose only the top-k (top10 in this case) most probable candidates from each document, each candidate must have long answer probability larger than a threshold (0.2), the rest candidates are set to negative.
# id_candidate_list_sorted stores (document id, candidate number) as keys, each of each contains its long answer probability (score).
start_time = time.time()
data_dict, id_list, id_candidate_len_dict, id_candidate_list_sorted = reduce1(n_candidate=10, th_candidate=0.2)
print("--- %s seconds ---" % (time.time() - start_time))

# Futher reduce the number of candidates from top10 to top4.
start_time = time.time()
id_candidate_list_sorted = reduce2(data_dict, id_list, id_candidate_len_dict, id_candidate_list_sorted, n_candidate=4, th_candidate=0.35)
print("--- %s seconds ---" % (time.time() - start_time))

# Acutual predictions start here. 
start_time = time.time()
# We keep word-level posterior start and end vectors. Since the token-level length is set to 360, this number should be enough for word-level.
word_len = 360
# Initialize start and end ""word-level"" prob vectors for easier probability averaging between different models equiped with different tokenizers.
start_prob = np.zeros((len(id_candidate_list_sorted),word_len),dtype=np.float32)
end_prob = np.zeros((len(id_candidate_list_sorted),word_len),dtype=np.float32)
start_label = np.zeros((len(id_candidate_list_sorted),),dtype=int)
end_label = np.zeros((len(id_candidate_list_sorted),),dtype=int)
# class_prob stores the 5-class classifier prob outputs.
# no answer(0), long but not short answer(1), short answer with span(2), NO(3), YES(4)
class_prob = np.zeros((len(id_candidate_list_sorted),5),dtype=np.float32)

# Perform prediction using two albert-xxl and two bert-large models. Weighted average of both long and short predictions for ensembling.
model_dir = '../albert-xxlarge-v2_2/weights/epoch2/'
test_prob1, test_prob2, test_prob3 = albert_predict(data_dict, id_list, id_candidate_len_dict, id_candidate_list_sorted, model_dir, word_len)
start_prob += 0.3*test_prob1
end_prob += 0.3*test_prob2
class_prob += 0.3*test_prob3
model_dir = '../albert-xxlarge-v2_3/weights/epoch2/'
test_prob1, test_prob2, test_prob3 = albert_predict(data_dict, id_list, id_candidate_len_dict, id_candidate_list_sorted, model_dir, word_len)
start_prob += 0.3*test_prob1
end_prob += 0.3*test_prob2
class_prob += 0.3*test_prob3
model_dir = '../bert-large-uncased_4/weights/epoch3/'
test_prob1, test_prob2, test_prob3 = bert_large_predict(data_dict, id_list, id_candidate_len_dict, id_candidate_list_sorted, model_dir, word_len)
start_prob += 0.2*test_prob1
end_prob += 0.2*test_prob2
class_prob += 0.2*test_prob3
model_dir = '../bert-large-uncased_5/weights/epoch3/'
test_prob1, test_prob2, test_prob3 = bert_large_predict(data_dict, id_list, id_candidate_len_dict, id_candidate_list_sorted, model_dir, word_len)
start_prob += 0.2*test_prob1
end_prob += 0.2*test_prob2
class_prob += 0.2*test_prob3

# The start and end words have the largest probabilities.
start_label = np.argmax(start_prob, axis=1)
end_label = np.argmax(end_prob, axis=1)

# initialize a temporary dictionary to store prediction values.
temp_dict = {}
for doc_id in id_list:
    temp_dict[doc_id] = {
                         'long_answer': {'start_token': -1, 'end_token': -1},
                         'long_answer_score': -1.0,
                         'short_answers': [{'start_token': -1, 'end_token': -1}],
                         'short_answers_score': -1.0,
                         'yes_no_answer': 'NONE'
                        }

# from cadidates to document
for i, (doc_id, candidate_index) in tqdm(enumerate(id_candidate_list_sorted)):
    # process long answer
    long_answer_score = 1.0 - class_prob[i,0] # 1 - no_answer_score
    if long_answer_score > temp_dict[doc_id]['long_answer_score']:
        temp_dict[doc_id]['long_answer_score'] = long_answer_score
        temp_dict[doc_id]['long_answer']['start_token'] = data_dict[doc_id]['long_answer_candidates'][candidate_index]['start_token']
        temp_dict[doc_id]['long_answer']['end_token'] = data_dict[doc_id]['long_answer_candidates'][candidate_index]['end_token']
        # process short answer
        short_answer_score = 1.0 - class_prob[i,0] - class_prob[i,1] # 1 - no_answer_score - long_but_not_short_answer_score
        temp_dict[doc_id]['short_answers_score'] = short_answer_score

        temp_dict[doc_id]['short_answers'][0]['start_token'] = -1
        temp_dict[doc_id]['short_answers'][0]['end_token'] = -1
        temp_dict[doc_id]['yes_no_answer'] = 'NONE'
        if max([class_prob[i,3], class_prob[i,4]]) > class_prob[i,2]:
            if class_prob[i,3] > class_prob[i,4]:
                temp_dict[doc_id]['yes_no_answer'] = 'NO'
            else:
                temp_dict[doc_id]['yes_no_answer'] = 'YES'
        else:
            short_start_word = int(start_label[i]) + data_dict[doc_id]['long_answer_candidates'][candidate_index]['start_token']
            short_end_word = int(end_label[i]) + data_dict[doc_id]['long_answer_candidates'][candidate_index]['start_token']
            if short_end_word > short_start_word:
                temp_dict[doc_id]['short_answers'][0]['start_token'] = short_start_word
                temp_dict[doc_id]['short_answers'][0]['end_token'] = short_end_word

# Copy the temporary dictionary into the final dictionary that meets the required format for validation.
final_dict = {}
final_dict['predictions'] = []
for doc_id in id_list:
    prediction_dict = {
                       'example_id': doc_id,
                       'long_answer': {'start_byte': -1, 'end_byte': -1, 'start_token': temp_dict[doc_id]['long_answer']['start_token'], 'end_token': temp_dict[doc_id]['long_answer']['end_token']},
                       'long_answer_score': temp_dict[doc_id]['long_answer_score'],
                       'short_answers': [{'start_byte': -1, 'end_byte': -1, 'start_token': temp_dict[doc_id]['short_answers'][0]['start_token'], 'end_token': temp_dict[doc_id]['short_answers'][0]['end_token']}],
                       'short_answers_score': temp_dict[doc_id]['short_answers_score'],
                       'yes_no_answer': temp_dict[doc_id]['yes_no_answer']
                      }
    final_dict['predictions'].append(prediction_dict)

# dump to json
with open('predictions.json', 'w') as fp:
    json.dump(final_dict, fp)
print("--- %s seconds ---" % (time.time() - start_time))




