# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy

class BucketIterator(object):
    def __init__(self, data, batch_size, shuffle=True, max_seq_length=128):
        self.shuffle = shuffle
        # if self.shuffle:
        #     random.shuffle(data)
        #print('data: ',len(data))
        self.batches = self.batch_transfer(data, batch_size)
        self.batch_len = len(self.batches)
        print(self.batch_len)
        self.max_seq_length = max_seq_length
    def batch_transfer(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        batches = []
        for i in range(num_batch):
            batches.append(self.transfer_data(data[i*batch_size : (i+1)*batch_size]))
        return batches

    def transfer_data(self, batch_data):
        batch_questionid = []
        batch_sentences_input_ids = []
        batch_sentences_input_mask = []
        batch_sentences_segment_ids = []
        batch_sentences_valid_ids = []
        batch_labels = []
        batch_labels_weights = []
        batch_length = []
        batch_adj = []
        batch_nodes_mask = []
        for item in batch_data:
            single_batch_sentences_input_ids = []
            single_batch_sentences_input_mask = []
            single_batch_sentences_segment_ids = []
            single_batch_sentences_valid_ids = []

            questionid, sentences,  labels, adj = item['questionid'], item['sentences'][0:20], item['labels'][0:20], item['adj']

            #print('labels: ',labels)
            single_batch_nodes_mask = [1 for i in range(len(sentences))]


            # if len(sentences) < 102:
            #     for id in range(102-len(sentences)):
            #         single_batch_nodes_mask.append(0)
            #         sentence={}
            #         sentence['input_ids'] = [0 for i in range(64)]
            #         sentence['input_mask'] = [0 for i in range(64)]
            #         sentence['segment_ids'] = [0 for i in range(64)]
            #         sentence['valid_ids'] = [0 for i in range(64)]
            #         sentences.append(sentence)
            for sentence in sentences:
                single_batch_sentences_input_ids.append(sentence['input_ids'])
                single_batch_sentences_input_mask.append(sentence['input_mask'])
                single_batch_sentences_segment_ids.append(sentence['segment_ids'])
                single_batch_sentences_valid_ids.append(sentence['valid_ids'])

            batch_questionid.append(questionid)
            if len(single_batch_sentences_input_ids)==102:
                print('input ids: ',len(single_batch_sentences_input_ids))
            batch_sentences_input_ids.append(single_batch_sentences_input_ids)
            batch_sentences_input_mask.append(single_batch_sentences_input_mask)
            batch_sentences_segment_ids.append(single_batch_sentences_segment_ids)
            batch_sentences_valid_ids.append(single_batch_sentences_valid_ids)

            ## node classfication label
            # labels_padding = [0 for i in range(102-len(labels))]
            # labels_weight = [labels.count(label)/(len(labels)*1.0) for label in labels]
            # #print('labels: ',labels)
            # #print('labels weight: ',labels_weight)
            # labels_weight_inverse = [1/i for i in labels_weight]
            # #print('labels weight inverse: ', labels_weight_inverse)
            # final_labels_weight = [(j/max(labels_weight_inverse))*10 for j in labels_weight_inverse]
            # #final_labels_weight = labels_weight_inverse
            # labels_weight_padding = [0 for i in range(102-len(final_labels_weight))]
            # batch_labels.append(labels+labels_padding)
            # batch_labels_weights.append(final_labels_weight+labels_weight_padding)

            ## pointer network label
            # print(labels)
            final_labels_0 = []
            final_labels_1 = []
            final_labels_2 = []
            final_labels_3 = []
            for id_label, label in enumerate(labels):
                if label == 0:
                    final_labels_0.append(id_label)
            for id_label, label in enumerate(labels):
                if label == 2:
                    final_labels_2.append(id_label)
            #random.shuffle(final_labels_2)
            for id_label, label in enumerate(labels):
                if label == 1:
                    final_labels_1.append(id_label)
            for id_label, label in enumerate(labels):
                if label == 3:
                    final_labels_3.append(id_label)
            #random.shuffle(final_labels_3)
            final_labels = final_labels_0+final_labels_2+final_labels_1+final_labels_3
            # print('final labels: ',final_labels)
            #labels_padding = [-1 for i in range(102 - len(labels))]
            if len(final_labels) == 1:
                print(final_labels)
            batch_length.append(len(final_labels))
            #print('final_labels: ',final_labels)
            batch_labels.append(final_labels)
            #print(len(final_labels+labels_padding))
            #print(batch_labels)
            #batch_adj.append(adj)
            batch_adj.append(numpy.pad(adj, ((0,102-len(adj)),(0,102-len(adj))), 'constant'))
            batch_nodes_mask.append(single_batch_nodes_mask)
        if torch.tensor(batch_sentences_input_ids, dtype=torch.long).shape[1]==102:
            print(batch_sentences_input_ids)
        return {'sentences_input_ids':torch.tensor(batch_sentences_input_ids, dtype=torch.long),
                'sentences_input_mask': torch.tensor(batch_sentences_input_mask, dtype=torch.long),
                'sentences_segment_ids': torch.tensor(batch_sentences_segment_ids, dtype=torch.long),
                'sentences_valid_ids': torch.tensor(batch_sentences_valid_ids, dtype=torch.long),
                'labels': torch.tensor(batch_labels, dtype=torch.long),
                'length': torch.tensor(batch_length, dtype=torch.long),
                #'labels_weights': torch.tensor(batch_labels_weights, dtype=torch.float),
                'adj': torch.tensor(batch_adj),
                'nodes_mask': torch.tensor(batch_nodes_mask, dtype=torch.float)
            }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]