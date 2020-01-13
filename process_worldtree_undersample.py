import re
import os
import glob
import spacy
import pickle
import pandas as pd
import random
import numpy as np
PATH_OUTPUT = '/home/zeyuzhang/PycharmProjects/gcn_eprg/processed_data/worldtree_final/train'
PATH_OUTPUT_UPDOWNSAMPLE_first = '/home/zeyuzhang/PycharmProjects/gcn_eprg/processed_data/worldtree_updownsample'
PATH_OUTPUT_UPDOWNSAMPLE = '/home/zeyuzhang/PycharmProjects/gcn_eprg/processed_data/worldtree_final/train_updownsample'
PATH_LEMMA_FILE = '/home/zeyuzhang/PycharmProjects/lemmatization-en.txt'
def clean_str(string):

    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r",", " , ", string)
    # string = re.sub(r"!", " ! ", string)
    string = re.sub(r";", " ", string)
    # string = re.sub(r"\(", " \( ", string)
    # string = re.sub(r"\)", " \) ", string)
    # string = re.sub(r"\?", " \? ", string)
    # string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def loadLookupLemmatizer(file_name):

    lemmatizerHashmap={}
    with open(file_name, 'r+') as f:
        text = f.readlines()
        for line in text:
            line=line.replace("[\\s]+", "\t").strip()
            split_list = line.lower().split("\t")
            lemma = split_list[0].strip().lower()
            word = split_list[1].strip().lower()
            lemmatizerHashmap[word] = lemma

    return lemmatizerHashmap

def hasoverlap(line_id, line_content, remain_text_lines, remain_line_id_list):
    graph_list = []
    for id, item in enumerate(remain_text_lines):
        if line_content&item != set():
            graph_list.append(line_id+'\t'+remain_line_id_list[id])
    return graph_list

def cleancontent(content, spacy_nlp, lemmatizerHashmap):
    content_posing = ['PROPN', 'VERB', 'NOUN', 'NUM', 'ADJ', 'ADV']
    line_content = clean_str(content)
    line_content_spacy = spacy_nlp(line_content)
    line_tokens = [token for token in line_content_spacy if not token.is_stop and token.pos_ in content_posing]
    line_tokens_lemma = [lemmatizerHashmap[token.text] if token.text in lemmatizerHashmap.keys() else token.lemma_ for token in line_tokens]
    line_tokens_lemma = set(line_tokens_lemma)
    return line_tokens_lemma

def up_downsample(path_output, path_output_updownsample):
    for filename in sorted(glob.glob(os.path.join(path_output, '*.txt'))):
        basename = os.path.basename(filename).split('.')[0]
        #print(basename)
        id_q = []
        id_a = []
        id_gold = []
        id_nongold = []
        with open(filename, 'r') as file:
            for idx, text_line in enumerate(file.readlines()):
                if len(text_line.strip())==0 or text_line[0]=='\n':
                    continue
                if text_line.strip().split('\t')[-1] == 'question':
                    id_q.append('\t'.join(text_line.strip().split('\t')[1:]))
                elif text_line.strip().split('\t')[-1] == 'answer':
                    id_a.append('\t'.join(text_line.strip().split('\t')[1:]))
                elif text_line.strip().split('\t')[-1] == 'gold':
                    id_gold.append('\t'.join(text_line.strip().split('\t')[1:]))
                else:
                    id_nongold.append('\t'.join(text_line.strip().split('\t')[1:]))
        final_id_q = id_q * 4
        final_id_a = id_a * 4
        final_id_gold = id_gold * 3
        final_id_nongold = random.sample(id_nongold, int(0.4*len(id_nongold)))
        #print(final_id_q+final_id_a+final_id_gold+final_id_nongold)
        new_output = final_id_q+final_id_a+final_id_gold+final_id_nongold
        random.shuffle(new_output)
        if len(new_output) > 102:
            print('outlength',basename)
            print(len(new_output))
            print(len(final_id_gold))
        with open(os.path.join(path_output_updownsample, os.path.basename(filename)), 'w') as file_:
            for idx_, item in enumerate(new_output):
                file_.write('{}\t{}'.format(str(idx_), item))
                file_.write('\n')

def build_cites(data_path, spacy_nlp, lemmatizerHashmap):
    for file in sorted(glob.glob(os.path.join(data_path,'*.txt'))):
        graph_list = []
        label_map = {}
        with open(os.path.join(data_path, file), 'r') as file_:
            text_lines = file_.read().split('\n')[:-1]
            line_cleancontent_list = []
            line_id_list = []
            line_label_list = []
            for idx, line in enumerate(text_lines):
                line_id = line.split('\t')[0]
                line_content = line.split('\t')[1]
                line_label = line.split('\t')[2]
                line_cleancontent_list.append(cleancontent(line_content, spacy_nlp, lemmatizerHashmap))
                line_id_list.append(line_id)
                line_label_list.append(line_label)
                label_map[line_id] = line_label
            for idxx in range(len(text_lines)):
                single_graph_list = hasoverlap(line_id_list[idxx], line_cleancontent_list[idxx], line_cleancontent_list[idxx+1:], line_id_list[idxx+1:])
                if single_graph_list == []:
                    continue
                else:
                    #print(single_graph_list)
                    graph_list.extend(single_graph_list)
        double_check = set()
        for item in graph_list:
            item_list = item.split('\t')
            double_check.add(item_list[0])
            double_check.add(item_list[1])
        # for item_ in ['Question', 'Answer'] + [str(i) for i in range(100)]:
        #     if item_ not in double_check:
        #         if label_map[item_] == 'Gold':
        #             print(file)
        #             print(item_)
        #             print(len(double_check))
        graph_list_str = '\n'.join(graph_list)
        with open(os.path.join(data_path, os.path.basename(file)+'.cites'), 'w') as file__:
            file__.write(graph_list_str)

def remove_disconnect_nodes(path_output, path_output_final):
    for filename in sorted(glob.glob(os.path.join(path_output, '*.cites'))):
        basename = os.path.basename(filename).split('.')[0]
        print(basename)
        with open(filename, 'r') as file:
            text = file.read()
            text_lines = text.split('\n')
            nodes = set()
            disconnected_nodes = []
            for item in text_lines:
                item_list = item.split('\t')
                nodes.add(item_list[0])
                nodes.add(item_list[1])
            # for item_ in ['Question', 'Answer'] + [str(i) for i in range(100)]:
            #     if item_ not in nodes:
            #         disconnected_nodes.append(item_)
        with open(os.path.join(path_output_final, basename+'.txt'), 'w') as file_:
            with open(os.path.join(path_output, basename+'.txt'), 'r') as file__:
                text_lines = file__.read().split('\n')[:-1]
                temp_text_lines = []
                line_id_new = 0
                for idx, line in enumerate(text_lines):
                    line_id = line.split('\t')[0]
                    line_content = line.split('\t')[1:]
                    if line_id not in nodes:
                        continue
                    else:
                        file_.write('{}\n'.format('\t'.join([str(line_id_new)]+line_content)))
                        line_id_new += 1

def build_cites_again(data_path, spacy_nlp, lemmatizerHashmap):
    for file in sorted(glob.glob(os.path.join(data_path,'*.txt'))):
        graph_list = []
        label_map = {}
        with open(os.path.join(data_path, file), 'r') as file_:
            text_lines = file_.read().split('\n')[:-1]
            line_cleancontent_list = []
            line_id_list = []
            line_label_list = []
            for idx, line in enumerate(text_lines):
                line_id = line.split('\t')[0]
                line_content = line.split('\t')[1]
                line_label = line.split('\t')[2]
                line_cleancontent_list.append(cleancontent(line_content, spacy_nlp, lemmatizerHashmap))
                line_id_list.append(line_id)
                line_label_list.append(line_label)
                label_map[line_id] = line_label
            for idxx in range(len(text_lines)):
                single_graph_list = hasoverlap(line_id_list[idxx], line_cleancontent_list[idxx], line_cleancontent_list[idxx+1:], line_id_list[idxx+1:])
                if single_graph_list == []:
                    continue
                else:
                    #print(single_graph_list)
                    graph_list.extend(single_graph_list)
        double_check = set()
        for item in graph_list:
            item_list = item.split('\t')
            double_check.add(item_list[0])
            double_check.add(item_list[1])
        for item_ in [str(i) for i in range(len(double_check))]:
            if item_ not in double_check:
                print(file)
                print(item_)
                print(len(double_check))
        graph_list_str = '\n'.join(graph_list)
        with open(os.path.join(data_path, file.split('.')[0]+'.cites'), 'w') as file__:
            file__.write(graph_list_str)

def build_adj_matrix(data_path):
    print(11)
    for filename in sorted(glob.glob(os.path.join(data_path, '*.cites'))):
        basename = os.path.basename(filename).split('.')[0]
        print(basename)
        cites_list = []
        idx_list = set()
        with open(os.path.join(data_path, filename), 'r') as file:
            for line in file.readlines():
                if len(line) == 0 or line[0] == '\n':
                    continue
                splits = line.strip().split('\t')
                cites_list.append((int(splits[0]), int(splits[1])))
                cites_list.append((int(splits[1]), int(splits[0])))
                idx_list.add(int(splits[0]))
                idx_list.add(int(splits[1]))
            cites_dic = {}
            for idx_number in range(len(idx_list)):
                cites_dic[idx_number] = [idx_number]
            for (item0, item1) in cites_list:
                if item0 not in cites_dic.keys():
                    cites_dic[item0] = [item1]
                else:
                    cites_dic[item0].append(item1)
            print(len(idx_list))
            print(len(cites_dic.keys()))
            assert len(cites_dic.keys()) == len(idx_list)
            matrix = np.zeros((len(idx_list), len(idx_list))).astype('float32')
            for key in cites_dic.keys():
                for value in cites_dic[key]:
                    matrix[key][value]=1.0
            #print(np.pad(matrix, ((0,102-len(matrix)),(0,102-len(matrix))), 'constant'))
            #print(matrix)
            rowsum = np.array(matrix.sum(axis = 1,keepdims=True))
            matrix = matrix/rowsum
            # r_mat_inv = sp.diags(r_inv)
            # mx = r_mat_inv.dot(mx)
            #print(matrix)

            with open(os.path.join(data_path, basename+'.graph'), 'wb') as fout:
                pickle.dump(matrix, fout)

def main():
    spacy_nlp = spacy.load('en_core_web_sm', disable=["parser", "ner", "entity_linker", "textcat", "entity_ruler"])
    lemmatizerHashmap = loadLookupLemmatizer(PATH_LEMMA_FILE)
    if not os.path.exists(PATH_OUTPUT_UPDOWNSAMPLE_first):
        os.makedirs(PATH_OUTPUT_UPDOWNSAMPLE_first)
    #up_downsample(PATH_OUTPUT, PATH_OUTPUT_UPDOWNSAMPLE_first)
    #if len(sorted(glob.glob(os.path.join(PATH_OUTPUT_UPDOWNSAMPLE_first,'*.cites')))) == 0:
    #build_cites(PATH_OUTPUT_UPDOWNSAMPLE_first, spacy_nlp, lemmatizerHashmap)
    if not os.path.exists(PATH_OUTPUT_UPDOWNSAMPLE):
        os.makedirs(PATH_OUTPUT_UPDOWNSAMPLE)
    #remove_disconnect_nodes(PATH_OUTPUT_UPDOWNSAMPLE_first, PATH_OUTPUT_UPDOWNSAMPLE)
    build_cites_again(PATH_OUTPUT_UPDOWNSAMPLE, spacy_nlp, lemmatizerHashmap)
    build_adj_matrix(PATH_OUTPUT_UPDOWNSAMPLE)


if __name__ == '__main__':
    main()