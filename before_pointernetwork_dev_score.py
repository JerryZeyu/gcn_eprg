import os
import glob
import numpy as np
sentences_map = []
for filename in sorted(glob.glob(os.path.join('/home/zeyuzhang/PycharmProjects/gcn_eprg/processed_data/worldtree_final/dev', '*.txt'))):
    f = open(filename)
    sentences = []
    labels = []
    for line in f:
        if len(line) == 0 or line[0] == "\n":
            continue
        splits = line.split('\t')
        sentences.append(splits[1])
        labels.append(splits[2].strip())
    temp_labels = labels[2:20]
    sorted_id = 1.0
    sentence_map = []
    for idx, label in enumerate(temp_labels):
        if label == 'gold':
            sentence_map.append(sorted_id/(idx+1))
            sorted_id += 1
    if len(sentence_map) != 0:
        sentences_map.append(np.mean(sentence_map))
print('MAP: ',np.mean(sentences_map))