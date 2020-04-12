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


def get_class_accuracy(logits, labels):
    predictions = np.argmax(F.softmax(logits,dim=1).cpu().data.numpy(), axis=1)
    return np.float32(np.sum(predictions=labels)) / len(labels), len(labels)

def get_position_accuracy(logits, labels):
    predictions = np.argmax(F.softmax(logits,dim=1).cpu().data.numpy(), axis=1)
    total_num = 0
    sum_correct = 0
    for i in range(len(labels)):
        if labels[i] >= 0:
            total_num += 1
            if predictions[i] == labels[i]:
                sum_correct += 1
    if total_num == 0:
        total_num = 1e-7
    return np.float32(sum_correct) / total_num, total_num


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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

    def _get_positive_input_ids(self, data, question_tokens):
        max_answer_tokens = self.max_seq_len-len(question_tokens)-3 # [CLS],[SEP],[SEP]
        candidate_start = data['positive_start']
        candidate_end = data['positive_end']
        candidate_words = data['positive_text']        

        words_to_tokens_index = []
        candidate_tokens = []
        for i, word in enumerate(candidate_words):
            words_to_tokens_index.append(len(candidate_tokens))
            if re.match(r'<.+>', word):  # remove paragraph tag
                continue
            tokens = self.tokenizer.tokenize(word)
            if len(candidate_tokens)+len(tokens) > max_answer_tokens:
                break
            candidate_tokens += tokens
        
        start_position = -1
        end_position = -1
        if data['annotations'][0]['short_answers']:
            start_position1 = data['annotations'][0]['short_answers'][0]['start_token']
            end_position1 = data['annotations'][0]['short_answers'][0]['end_token']
            if (start_position1 >= candidate_start and end_position1 <= candidate_end) and ((end_position1-candidate_start) < len(words_to_tokens_index)):
                start_position = words_to_tokens_index[start_position1-candidate_start]+len(question_tokens)+2
                end_position = words_to_tokens_index[end_position1-candidate_start]+len(question_tokens)+2
        return candidate_tokens, start_position, end_position

    def _get_negative_input_ids(self, data, question_tokens):
        max_answer_tokens = self.max_seq_len-len(question_tokens)-3 # [CLS],[SEP],[SEP]
        candidate_start = data['negative_start']
        candidate_end = data['negative_end']
        candidate_words = data['negative_text']        

        words_to_tokens_index = []
        candidate_tokens = []
        for i, word in enumerate(candidate_words):
            words_to_tokens_index.append(len(candidate_tokens))
            if re.match(r'<.+>', word):  # remove paragraph tag
                continue
            tokens = self.tokenizer.tokenize(word)
            if len(candidate_tokens)+len(tokens) > max_answer_tokens:
                break
            candidate_tokens += tokens
        
        start_position = -1
        end_position = -1
        return candidate_tokens, start_position, end_position
        
    def __call__(self, batch_ids):
        batch_size = 2*len(batch_ids)

        batch_input_ids = np.zeros((batch_size, self.max_seq_len), dtype=np.int64)
        batch_token_type_ids = np.ones((batch_size, self.max_seq_len), dtype=np.int64)

        batch_y_start = np.zeros((batch_size,), dtype=np.int64)
        batch_y_end = np.zeros((batch_size,), dtype=np.int64)
        batch_y = np.zeros((batch_size,), dtype=np.int64)

        for i, doc_id in enumerate(batch_ids):
            data = self.data_dict[doc_id]

            # get label
            annotations = data['annotations'][0]
            if annotations['yes_no_answer'] == 'YES':
                batch_y[i*2] = 4
            elif annotations['yes_no_answer'] == 'NO':
                batch_y[i*2] = 3
            elif annotations['short_answers']:
                batch_y[i*2] = 2
            elif annotations['long_answer']['candidate_index'] != -1:
                batch_y[i*2] = 1
            batch_y[i*2+1] = 0

            # get positive and negative samples
            question_tokens = self.tokenizer.tokenize(data['question_text'])[:self.max_question_len]
            # positive
            answer_tokens, start_position, end_position = self._get_positive_input_ids(data, question_tokens)
            input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + answer_tokens + ['[SEP]']
            #if annotations['short_answers']:
            #    print(data['question_text'],"[AAA]",input_tokens[start_position:end_position])
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            batch_input_ids[i*2, :len(input_ids)] = input_ids
            batch_token_type_ids[i*2, :len(input_ids)] = [0 if k<=input_ids.index(102) else 1 for k in range(len(input_ids))]
            batch_y_start[i*2] = start_position
            batch_y_end[i*2] = end_position
            # negative
            answer_tokens, start_position, end_position = self._get_negative_input_ids(data, question_tokens)
            input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + answer_tokens + ['[SEP]']
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            batch_token_type_ids[i*2+1, :len(input_ids)] = [0 if k<=input_ids.index(102) else 1 for k in range(len(input_ids))]
            batch_input_ids[i*2+1, :len(input_ids)] = input_ids
            batch_y_start[i*2+1] = start_position
            batch_y_end[i*2+1] = end_position

        batch_attention_mask = batch_input_ids > 0

        return torch.from_numpy(batch_input_ids), torch.from_numpy(batch_attention_mask), torch.from_numpy(batch_token_type_ids), torch.LongTensor(batch_y_start), torch.LongTensor(batch_y_end), torch.LongTensor(batch_y)


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


def loss_fn(preds, labels):
    start_preds, end_preds, class_preds = preds
    start_labels, end_labels, class_labels = labels
    
    start_loss = nn.CrossEntropyLoss(ignore_index=-1)(start_preds, start_labels)
    end_loss = nn.CrossEntropyLoss(ignore_index=-1)(end_preds, end_labels)
    class_loss = nn.CrossEntropyLoss()(class_preds, class_labels)
    return start_loss, end_loss, class_loss


def random_sample_negative_candidates(distribution):
    temp = np.random.random()
    value = 0.
    for index in range(len(distribution)):
        value += distribution[index]
        if value > temp:
            break
    return index


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")

    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.device = device

    seed = 1001
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


    # prepare input
    json_dir = '../../input/simplified-nq-train.jsonl'
    max_data = 9999999999

    id_list = []
    data_dict = {}
    with open(json_dir) as f:
        for n, line in tqdm(enumerate(f)):
            if n > max_data:
                break
            data = json.loads(line)

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

            if is_pos and len(data['long_answer_candidates'])>1:
                data_id = data['example_id']
                id_list.append(data_id)

                # uniform sampling
                distribution = np.ones((len(data['long_answer_candidates']),),dtype=np.float32)
                if is_pos:
                    distribution[data['annotations'][0]['long_answer']['candidate_index']] = 0.
                distribution /= len(distribution)
                negative_candidate_index = random_sample_negative_candidates(distribution)

                #
                doc_words = data['document_text'].split()
                # negative
                candidate = data['long_answer_candidates'][negative_candidate_index]
                negative_candidate_words = doc_words[candidate['start_token']:candidate['end_token']]  
                negative_candidate_start = candidate['start_token']
                negative_candidate_end = candidate['end_token']
                # positive
                candidate = data['long_answer_candidates'][annotations['long_answer']['candidate_index']]
                positive_candidate_words = doc_words[candidate['start_token']:candidate['end_token']]
                positive_candidate_start = candidate['start_token']
                positive_candidate_end = candidate['end_token']

                # initialize data_dict
                data_dict[data_id] = {
                                      'question_text': data['question_text'],
                                      'annotations': data['annotations'],  
                                      'positive_text': positive_candidate_words,
                                      'positive_start': positive_candidate_start,  
                                      'positive_end': positive_candidate_end,   
                                      'negative_text': negative_candidate_words,       
                                      'negative_start': negative_candidate_start,  
                                      'negative_end': negative_candidate_end,               
                                     }


    print(len(id_list))
    random.shuffle(id_list)


    # hyperparameters
    max_seq_len = 384
    max_question_len = 64
    learning_rate = 0.00002
    batch_size = 4
    ep = 0


    # build model
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model_path = '../../huggingface_pretrained/bert-base-uncased/'
    config = BertConfig.from_pretrained(model_path)
    config.num_labels = 5
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
    model = BertForQuestionAnswering.from_pretrained(model_path, config=config)

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
    model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )


    # training

    # iterator for training
    train_datagen = TFQADataset(id_list=id_list)
    train_sampler = DistributedSampler(train_datagen)
    train_collate = Collator(data_dict=data_dict, 
                             tokenizer=tokenizer, 
                             max_seq_len=max_seq_len, 
                             max_question_len=max_question_len)
    train_generator = DataLoader(dataset=train_datagen,
                                 sampler=train_sampler,
                                 collate_fn=train_collate,
                                 batch_size=batch_size,
                                 num_workers=3,
                                 pin_memory=True)

    # train
    losses1 = AverageMeter() # start
    losses2 = AverageMeter() # end
    losses3 = AverageMeter() # class
    accuracies1 = AverageMeter() # start
    accuracies2 = AverageMeter() # end
    accuracies3 = AverageMeter() # class
    model.train()
    for j,(batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_y_start, batch_y_end, batch_y) in enumerate(train_generator):
        batch_input_ids = batch_input_ids.cuda()
        batch_attention_mask = batch_attention_mask.cuda()
        batch_token_type_ids = batch_token_type_ids.cuda()
        labels1 = batch_y_start.cuda()
        labels2 = batch_y_end.cuda()
        labels3 = batch_y.cuda()

        logits1, logits2, logits3 = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
        y_true = (batch_y_start, batch_y_end, batch_y)
        loss1, loss2, loss3 = loss_fn((logits1, logits2, logits3), (labels1, labels2, labels3))
        loss = loss1+loss2+loss3
        acc1, n_position1 = get_position_accuracy(logits1, labels1)
        acc2, n_position2 = get_position_accuracy(logits2, labels2)
        acc3, n_position3 = get_position_accuracy(logits3, labels3)

        losses1.update(loss1.item(), n_position1)
        losses2.update(loss2.item(), n_position2)
        losses3.update(loss3.item(), n_position3)
        accuracies1.update(acc1, n_position1)
        accuracies2.update(acc2, n_position2)
        accuracies3.update(acc3, n_position2)

        optimizer.zero_grad()

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()

    if args.local_rank == 0:
        print('epoch: {}, train_loss1: {}, train_loss2: {}, train_loss3: {}, train_acc1: {}, train_acc2: {}, train_acc3: {}'.format(ep,losses1.avg,losses2.avg,losses3.avg,accuracies1.avg,accuracies2.avg,accuracies3.avg), flush=True)

        out_dir = 'weights/epoch0/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        torch.save(model.module.state_dict(), out_dir+'pytorch_model.bin')

if __name__ == "__main__":
    main()
