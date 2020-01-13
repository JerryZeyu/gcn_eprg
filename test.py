import numpy as np
import scipy.sparse as sp
idx_features_labels = np.genfromtxt('/home/zeyuzhang/PycharmProjects/gcn_eprg/processed_data/worldtree_final/ACTAAP_2007_7_1.txt', delimiter='\t', dtype=np.dtype(str))

idx = np.array(idx_features_labels[:, 0])
idx_map = {j: i for i, j in enumerate(idx)}
edges_unordered = np.genfromtxt('/home/zeyuzhang/PycharmProjects/gcn_eprg/processed_data/worldtree_final/ACTAAP_2007_7_1.txt.cites',delimiter='\t',
                                dtype=np.dtype(str))
edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                 dtype=np.int32).reshape(edges_unordered.shape)
adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                    shape=(idx.shape[0], idx.shape[0]),
                    dtype=np.float32)
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
adj = adj + sp.eye(adj.shape[0])
print(adj)