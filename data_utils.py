import os
import glob
import pickle
import numpy as np
from torch.utils.data import Dataset

class WorldTreeDataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class WorldTreeDatesetReader:
    @staticmethod
    def __read_file(data_path):
        all_data = []
        for filename in sorted(glob.glob(os.path.join(data_path, '*.txt'))):
            f = open(filename)
            sentences = []
            labels = []
            for line in f:
                if len(line) == 0 or line[0] == "\n":
                    continue
                splits = line.split('\t')
                sentences.append(splits[1])
                labels.append(splits[2].strip())
                if splits[2].strip() == 'question':
                    question_sentence = splits[1]
            for sentence in sentences:
                sentence = question_sentence+' '+sentence
            basename = os.path.splitext(os.path.basename(filename))[0]
            with open(os.path.join(data_path, basename+'.graph'), 'rb') as file:
                adj_matrix = pickle.load(file)
            data = {
                'questionid': basename,
                'sentences': sentences,
                'labels': labels,
                'adj': adj_matrix
            }
            all_data.append(data)
        return all_data

    @staticmethod
    def __create_examples(all_data, tokenizer,max_seq_length):
        label_list = ['question', 'answer', 'gold', 'nongold']
        label_map = {label: i for i, label in enumerate(label_list, 0)}
        #label_map = {'question': 0, 'answer':0, 'gold':0, 'nongold':1}
        max_sentence_lenth_list = []
        for idx, data in enumerate(all_data):
            data['labels'] = [label_map[i] for i in data['labels']]
            sentences_feature = []
            sentence_length_list = []
            for sentence_idx, sentence in enumerate(data['sentences']):
                textlist = sentence.split(' ')
                sentence_length_list.append(len(textlist))
                tokens = []
                valid = []
                for i, word in enumerate(textlist):
                    token = tokenizer.tokenize(word)
                    tokens.extend(token)
                    for m in range(len(token)):
                        if m == 0:
                            valid.append(1)
                        else:
                            valid.append(0)
                if len(tokens) >= max_seq_length - 1:
                    tokens = tokens[0:(max_seq_length - 2)]
                    valid = valid[0:(max_seq_length - 2)]
                ntokens = []
                segment_ids = []
                #label_ids = []
                ntokens.append("[CLS]")
                segment_ids.append(0)
                valid.insert(0, 1)
                for i, token in enumerate(tokens):
                    ntokens.append(token)
                    segment_ids.append(0)
                ntokens.append("[SEP]")
                segment_ids.append(0)
                valid.append(1)
                input_ids = tokenizer.convert_tokens_to_ids(ntokens)
                input_mask = [1] * len(input_ids)
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)
                    valid.append(1)
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                #assert len(label_ids) == max_seq_length
                assert len(valid) == max_seq_length
                # if sentence_idx < 5:
                #     print("tokens: %s" % " ".join(
                #     [str(x) for x in tokens]))
                #     print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                #     print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                #     print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                sentence_feature = {'input_ids':input_ids,
                              'input_mask':input_mask,
                              'segment_ids':segment_ids,
                              'valid_ids':valid}
                sentences_feature.append(sentence_feature)
            data['sentences'] = sentences_feature
            max_sentence_lenth_list.append(np.max(sentence_length_list))
        #print(all_data)
        print('max sentence length: ',np.max(max_sentence_lenth_list))
        return all_data

    def __init__(self, dataset, tokenizer, max_seq_length=128):
        print("preparing {0} dataset ...".format('WorldTree'))
        fname = {
                'train': os.path.join(dataset, 'train'),
                'dev': os.path.join(dataset, 'dev')
        }
        self.train_data = WorldTreeDataset(WorldTreeDatesetReader.__create_examples(WorldTreeDatesetReader.__read_file(fname['train']),tokenizer, max_seq_length))
        self.dev_data = WorldTreeDataset(WorldTreeDatesetReader.__create_examples(WorldTreeDatesetReader.__read_file(fname['dev']),tokenizer,max_seq_length))
