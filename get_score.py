import os
import sys
import pickle
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
    eval_logits=list(pickle_load_large_file('epoch'+str(i)+'_eval_logits'))
    eval_label=list(pickle_load_large_file('epoch'+str(i)+'_eval_label'))
    total_gold_number = 0
    predicted_gold_number = 0
    TP_FP_0 = 0
    TP_FN_0 = 0
    TP_0 = 0
    TP__0 =0
    TP_FP_1 = 0
    TP_FN_1 = 0
    TP_1 = 0
    TP__1 = 0
    for id, single_eval_nodes_mask in enumerate(eval_nodes_mask):
        #print(eval_logits[id])
        #print(eval_label[id])
        for id_, node_mask in enumerate(list(single_eval_nodes_mask)):
            #print(eval_logits[id])
            nonpad_number = list(node_mask).count(1)
            for id__ in range(nonpad_number):
            #for id__, label in enumerate(eval_label[id][id_]):
                if list(list(eval_logits[id])[id_])[id__] == 0:
                    TP_FP_0 += 1
                    if list(list(eval_label[id])[id_])[id__] == 0:
                        TP_0 += 1
                if list(list(eval_label[id])[id_])[id__] == 0:
                    TP_FN_0 += 1
                    if list(list(eval_logits[id])[id_])[id__] == 0:
                        TP__0 += 1
                if list(list(eval_logits[id])[id_])[id__] == 1:
                    TP_FP_1 += 1
                    if list(list(eval_label[id])[id_])[id__] == 1:
                        TP_1 += 1
                if list(list(eval_label[id])[id_])[id__] == 1:
                    TP_FN_1 += 1
                    if list(list(eval_logits[id])[id_])[id__] == 1:
                        TP__1 += 1
    assert TP_0 == TP__0
    precision_0 = (TP_0*1.0)/TP_FP_0
    recall_0 = (TP_0*1.0)/TP_FN_0
    F1_0 = (2*precision_0*recall_0)/(precision_0+recall_0)
    print('precision_0: ', (TP_0*1.0)/TP_FP_0)
    print('recall_0: ',(TP_0*1.0)/TP_FN_0)
    print('F1_0: ',F1_0)
    assert TP_1 == TP__1
    precision_1 = (TP_1 * 1.0) / TP_FP_1
    recall_1 = (TP_1 * 1.0) / TP_FN_1
    F1_1 = (2 * precision_1 * recall_1) / (precision_1 + recall_1)
    print('precision_1: ', (TP_1 * 1.0) / TP_FP_1)
    print('recall_1: ', (TP_1 * 1.0) / TP_FN_1)
    print('F1_1: ', F1_1)