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
for i in range(44,45):
    eval_argmax=list(pickle_load_large_file('epoch'+str(i)+'_eval_argmax.pkl'))
    eval_label=list(pickle_load_large_file('epoch'+str(i)+'_eval_label.pkl'))
    print(i)
    ##batch loop
    sentences_rank_list = []
    sentences_map = []
    for id, batch_eval_label in enumerate(eval_label):
        #print(eval_logits[id])
        #print(eval_label[id])
        for id_, single_eval_label in enumerate(list(batch_eval_label)):
            #print(eval_logits[id])
            single_eval_label = list(single_eval_label)
            q_location = single_eval_label.index(0)
            a_location = single_eval_label.index(1)
            gold_rows = single_eval_label[q_location+1:a_location]
            nongold_rows = single_eval_label[a_location+1:]
            single_argmax = list(list(eval_argmax[id])[id_])
            print(single_argmax)
            single_argmax_set = list(sorted(set(single_argmax), key=single_argmax.index))
            all_idx = [i for i in range(20)]
            for idx in all_idx:
                if idx not in single_argmax_set:
                    single_argmax_set.append(idx)
            if 0 in single_argmax_set:
                single_argmax_set.remove(0)
            if 1 in single_argmax_set:
                single_argmax_set.remove(1)
            single_map = []
            sort_id = 1.0
            #print(len(single_argmax_set))
            assert len(single_argmax_set) == 18
            for idx__, argmax in enumerate(single_argmax_set):
                if argmax in gold_rows:
                    #print(sort_id/(idx__+1))
                    single_map.append(sort_id/(idx__+1))
                    sort_id+=1
            if len(single_map) != 0:
            #print(np.mean(single_map))
                sentences_map.append(np.mean(single_map))
            #print('label: ',single_eval_label)
    print('MAP: ',np.mean(sentences_map))


    #         for id__ in range(2, nonpad_number):
    #             #print(list(list(eval_logits[id])[id_]))
    #             gold_score = list(list(list(eval_logits[id])[id_])[id__])[0]
    #             true_label = list(list(eval_label[id])[id_])[id__]
    #             sentence_rank_list.append((gold_score, true_label))
    #         sentences_rank_list.append(sentence_rank_list)
    # all_map_list = []
    # for sentence in sentences_rank_list:
    #     sorted_sentence = sorted(sentence, key=lambda x:x[0], reverse=True)
    #     ranked_location = []
    #     for idx, item in enumerate(sorted_sentence):
    #         if item[1] == 0:
    #             ranked_location.append(idx+1)
    #     single_map_list = []
    #     for idx_, item_ in enumerate(ranked_location):
    #         single_map_list.append((idx_*1.0)/item_)
    #     single_map = np.mean(single_map_list)
    #     all_map_list.append(single_map)
    # print('map: ', np.mean(all_map_list))