import os
import sys
import pickle
import numpy as np
def pickle_load_large_file(filepath):
    max_bytes = 2**31 - 1
    input_size = os.path.getsize(filepath)
    bytes_in = bytearray(0)
    with open(filepath, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    obj = pickle.loads(bytes_in)
    return obj
for i in range(10):
    eval_nodes_mask=list(pickle_load_large_file('epoch'+str(i)+'_eval_nodes_mask.pkl'))
    eval_logits=list(pickle_load_large_file('epoch'+str(i)+'_eval_logits_2'))
    eval_label=list(pickle_load_large_file('epoch'+str(i)+'_eval_label'))
    ##batch loop
    sentences_rank_list = []
    for id, single_eval_nodes_mask in enumerate(eval_nodes_mask):
        #print(eval_logits[id])
        #print(eval_label[id])
        for id_, node_mask in enumerate(list(single_eval_nodes_mask)):
            #print(eval_logits[id])
            nonpad_number = list(node_mask).count(1)
            sentence_rank_list = []
            for id__ in range(2, nonpad_number):
                #print(list(list(eval_logits[id])[id_]))
                gold_score = list(list(list(eval_logits[id])[id_])[id__])[0]
                true_label = list(list(eval_label[id])[id_])[id__]
                sentence_rank_list.append((gold_score, true_label))
            sentences_rank_list.append(sentence_rank_list)
    all_map_list = []
    for sentence in sentences_rank_list:
        sorted_sentence = sorted(sentence, key=lambda x:x[0], reverse=True)
        ranked_location = []
        for idx, item in enumerate(sorted_sentence):
            if item[1] == 0:
                ranked_location.append(idx+1)
        single_map_list = []
        for idx_, item_ in enumerate(ranked_location):
            single_map_list.append((idx_*1.0)/item_)
        single_map = np.mean(single_map_list)
        all_map_list.append(single_map)
    print('map: ', np.mean(all_map_list))
    #print(len(sentences_rank_list))
                # if list(list(eval_logits[id])[id_])[id__] == 0:
                #     TP_FP += 1
                #     if list(list(eval_label[id])[id_])[id__] == 0:
                #         TP += 1
                # if list(list(eval_label[id])[id_])[id__] == 0:
                #     TP_FN += 1
                #     if list(list(eval_logits[id])[id_])[id__] == 0:
                #         TP_ += 1
    # assert TP == TP_
    # precision = (TP*1.0)/TP_FP
    # recall = (TP*1.0)/TP_FN
    # F1 = (2*precision*recall)/(precision+recall)
    # print('precision: ', (TP*1.0)/TP_FP)
    # print('recall: ',(TP*1.0)/TP_FN)
    # print('F1: ',F1)