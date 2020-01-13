from utils import *
import os
import time
import numpy as np
import os.path as osp
import torch
import pickle
from tqdm import tqdm
import torch.nn.functional as F
from model_class import bertGCN, bertGCN_Pointer
from torch_geometric.datasets import Planetoid
from pytorch_transformers import (WEIGHTS_NAME, AdamW, BertConfig,
                                  BertForTokenClassification, BertTokenizer,
                                  WarmupLinearSchedule)
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
import argparse
import pandas as pd
from data_utils import WorldTreeDatesetReader
from bucket_iterator import BucketIterator

def pickle_dump_large_file(obj, filepath):
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])

def pickle_load_large_file(filepath):
    max_bytes = 2**31 - 1
    input_size = os.path.getsize(filepath)
    bytes_in = bytearray(0)
    with open(filepath, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    obj = pickle.loads(bytes_in)
    return obj

# class KipfGCN(torch.nn.Module):
#     def __init__(self, emb_dim, num_class, params):
#         super(KipfGCN, self).__init__()
#         self.p = params
#         self.emb_dim = emb_dim
#         self.conv1 = GCNConv(self.emb_dim, self.p.gcn_dim, cached=True)
#         self.conv2 = GCNConv(self.p.gcn_dim, num_class, cached=True)
#
#     def forward(self, x, edge_index):
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=self.p.dropout, training=self.training)
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=1)

class AverageMeter(object):
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

class Main(object):

    def add_model(self):
        model = bertGCN_Pointer(self.p, self.p.bert_model, self.device)
        model = torch.nn.DataParallel(model)
        #model = torch.nn.parallel.DistributedDataParallel(model)
        model.to(self.device)
        #model = torch.nn.parallel.DistributedDataParallel(model)
        return model

    def masked_accuracy(self, output, target, mask):
        """Computes a batch accuracy with a mask (for padded sequences) """
        with torch.no_grad():
            masked_output = torch.masked_select(output, mask)
            masked_target = torch.masked_select(target, mask)
            accuracy = masked_output.eq(masked_target).float().mean()

            return accuracy

    def add_optimizer(self, parameters):
        """
        Add optimizer for training variables
        Parameters
        ----------
        parameters:	Model parameters to be learned
        Returns
        -------
        train_op:	Training optimizer
        """
        if self.p.opt == 'adam':
            #return torch.optim.Adam(parameters, lr=0.01, weight_decay=5e-4)
            return AdamW(parameters, lr=self.p.lr, eps=self.p.l2)
        else:
            return AdamW(parameters, lr=self.p.lr, eps=self.p.l2)
            #return torch.optim.SGD(parameters, lr=self.p.lr, weight_decay=self.p.l2)

    def __init__(self, params):
        """
        Constructor for the main function. Loads data and creates computation graph.
        Parameters
        ----------
        params:		Hyperparameters of the model
        Returns
        -------
        """
        self.p = params

        self.p.save_dir = '{}/{}'.format(self.p.model_dir, self.p.name)
        if not osp.exists(self.p.log_dir):    os.system(
            'mkdir -p {}'.format(self.p.log_dir))  # Create log directory if doesn't exist
        if not osp.exists(self.p.save_dir): os.system(
            'mkdir -p {}'.format(self.p.model_dir))  # Create model directory if doesn't exist

        # Get Logger
        self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)
        self.logger.info(vars(self.p))
        print(vars(self.p))

        if self.p.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            n_gpu = torch.cuda.device_count()
            print('n_gpu: ',n_gpu)
            #torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            #torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')
        # tokenizer = BertTokenizer.from_pretrained(self.p.bert_model, do_lower_case=self.p.do_lower_case)
        # worldtree_dataset = WorldTreeDatesetReader(dataset=self.p.dataset, tokenizer=tokenizer, max_seq_length=self.p.max_seq_length)
        # pickle_dump_large_file(worldtree_dataset,'worldtree_dataset_q_128.pkl')
        worldtree_dataset = pickle_load_large_file('worldtree_dataset_q_128.pkl')
        self.train_data_loader = BucketIterator(data=worldtree_dataset.train_data, batch_size=self.p.batch_size, shuffle=True, max_seq_length=self.p.max_seq_length)
        # pickle_dump_large_file(self.train_data_loader, 'train_data_loader_64_2_pointer.pkl')
        # self.train_data_loader = pickle_load_large_file('train_data_loader_64_2_pointer.pkl')
        self.dev_data_loader = BucketIterator(data=worldtree_dataset.dev_data, batch_size=self.p.batch_size, shuffle=False, max_seq_length=self.p.max_seq_length)
        # pickle_dump_large_file(self.dev_data_loader, 'dev_data_loader_64_2_pointer.pkl')
        # self.dev_data_loader = pickle_load_large_file('dev_data_loader_64_2_pointer.pkl')
        self.model = self.add_model()
        self.loss_function = torch.nn.CrossEntropyLoss(reduce=False)
        # if n_gpu > 1:
        #     self.model = torch.nn.DataParallel(self.model)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = self.add_optimizer(optimizer_grouped_parameters)
        #self.optimizer = self.add_optimizer(self.model.parameters())
        num_train_optimization_steps = int(self.train_data_loader.batch_len//4) * self.p.max_epochs
        #warmup_steps = int(0.1 * num_train_optimization_steps)
        self.scheduler = WarmupLinearSchedule(self.optimizer, warmup_steps=0, t_total=num_train_optimization_steps)
    def get_acc(self, logits, y_actual, mask):
        """
        Calculates accuracy
        Parameters
        ----------
        logits:		Output of the model
        y_actual: 	Ground truth label of nodes
        mask: 		Indicates the nodes to be considered for evaluation
        Returns
        -------
        accuracy:	Classification accuracy for labeled nodes
        """

        y_pred = torch.max(logits, dim=1)[1]
        return y_pred.eq(y_actual[mask]).sum().item() / mask.sum().item()

    def evaluate(self, sess, split='valid'):
        """
        Evaluate model on valid/test data
        Parameters
        ----------
        sess:		Session of tensorflow
        split:		Data split to evaluate on
        Returns
        -------
        loss:		Loss over the entire data
        acc:		Overall Accuracy
        """

        feed_dict = self.create_feed_dict(split=split)
        loss, acc = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)

        return loss, acc

    def run_epoch(self, epochs, loss_function, shuffle=True):
        """
        Runs one epoch of training and evaluation on validation set
        Parameters
        ----------
        sess:		Session of tensorflow
        data:		Data to train on
        epoch:		Epoch number
        shuffle:	Shuffle data while before creates batches
        Returns
        -------
        loss:		Loss over the entire data
        Accuracy:	Overall accuracy
        """
        #train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
        global_step = 0
        train_loss = AverageMeter()
        train_accuracy = AverageMeter()
        for epoch in tqdm(range(epochs), desc='Epoch'):
            min_loss = 10
            print('epoch: ',epoch)
            t = time.time()
            #print(len(self.train_data_loader))
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step+=1
                self.model.train()
                # if i_batch == 255:
                #     break
                #print('i_batch: ',i_batch)
                input_ids = sample_batched['sentences_input_ids'].to(self.device)
                input_mask = sample_batched['sentences_input_mask'].to(self.device)
                segment_id = sample_batched['sentences_segment_ids'].to(self.device)
                valid_ids = sample_batched['sentences_valid_ids'].to(self.device)
                adj = sample_batched['adj'].to(self.device)
                labels = sample_batched['labels'].to(self.device)
                length = sample_batched['length'].to(self.device)
                ##add weight
                #labels_weights = sample_batched['labels_weights'].to(self.device)

                nodes_mask = sample_batched['nodes_mask'].to(self.device)
                ## node classfication loss
                #logits = self.model(input_ids,input_mask, segment_id, valid_ids,adj, nodes_mask, length)
                # print('labels shape: ',labels.shape)
                # print('logits shape: ', logits.shape)
                # logits=logits.view(1,-1,2)
                # labels =labels.view(1,-1)
                # print('labels shape: ', labels.shape)
                # print('logits shape: ', logits.shape)
                #print('final labels shape',labels)
                # train_loss = loss_function(logits.squeeze(), labels.squeeze())
                # final_train_loss = torch.mean(train_loss * nodes_mask)
                # #final_train_loss = torch.mean(train_loss * labels_weights.view(1,-1).squeeze())
                # final_train_loss = final_train_loss/8
                # if final_train_loss.item() <= min_loss:
                #     min_loss = final_train_loss.item()
                # print(final_train_loss.item())
                # final_train_loss.backward()

                ##pointer network loss
                log_pointer_score, argmax_pointer, mask = self.model(input_ids, input_mask, segment_id, valid_ids, adj,
                                                                     nodes_mask, length)
                unrolled = log_pointer_score.view(-1, log_pointer_score.size(-1))
                #print('unrolled: ',unrolled.shape)
                #print('labels: ',labels.shape)
                loss = F.nll_loss(unrolled, labels.view(-1), ignore_index=-1)
                assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'
                print(loss.item())
                loss.backward()
                #print(loss.item())
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                train_loss.update(loss.item(), labels.size(0))

                # mask = mask[:, 0, :]
                # train_accuracy.update(self.masked_accuracy(argmax_pointer, labels, mask).item(), mask.int().sum().item())
                if (i_batch +1) % 4 ==0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                if i_batch % 10 == 0:
                    print('Epoch {}: Train [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'
                          .format(epoch, i_batch * self.p.batch_size, self.train_data_loader.batch_len,
                                  100. * i_batch / self.train_data_loader.batch_len, train_loss.avg, train_accuracy.avg))
            if (epoch+1) % 10 == 0:
                torch.save({'epoch':epoch+1,'state_dict': self.model.state_dict(), 'scheduler': self.scheduler,
                            'optimizer': self.optimizer.state_dict()},'model_epoch'+str(epoch+1)+'_'+str(min_loss)+'.pth.tar')
        # loaded_model = torch.load('/home/zeyuzhang/PycharmProjects/gcn_eprg/model_epoch3_0.13608218729496002.pth.tar')
        # self.model.load_state_dict(loaded_model['state_dict'])
        # self.model.to(self.device)
        #self.optimizer.load_state_dict(loaded_model['optimizer'])
            self.model.eval()
            epoch_eval_label = []
            epoch_eval_logits = []
            epoch_eval_argmax = []
            for i_batch_dev, sample_batched_dev in enumerate(self.dev_data_loader):
                print('i batch dev: ',i_batch_dev)
                input_ids_dev = sample_batched_dev['sentences_input_ids'].to(self.device)
                input_mask_dev = sample_batched_dev['sentences_input_mask'].to(self.device)
                segment_id_dev = sample_batched_dev['sentences_segment_ids'].to(self.device)
                valid_ids_dev = sample_batched_dev['sentences_valid_ids'].to(self.device)
                adj_dev = sample_batched_dev['adj'].to(self.device)
                labels_dev = sample_batched_dev['labels'].to(self.device)
                length_dev = sample_batched_dev['length'].to(self.device)
                nodes_mask_dev = sample_batched_dev['nodes_mask'].to(self.device)
                with torch.no_grad():
                    log_pointer_score_dev, argmax_pointer_dev, mask_dev = self.model(input_ids_dev, input_mask_dev, segment_id_dev, valid_ids_dev,
                                                                         adj_dev,
                                                                         nodes_mask_dev, length_dev)
                #print(argmax_pointer_dev.cpu().numpy())
                epoch_eval_logits.append(log_pointer_score_dev.cpu().numpy())
                epoch_eval_label.append(labels_dev.cpu().numpy())
                epoch_eval_argmax.append(argmax_pointer_dev.cpu().numpy())
            pickle_dump_large_file(epoch_eval_logits, 'epoch'+str(epoch)+'_eval_logits.pkl')
            pickle_dump_large_file(epoch_eval_label, 'epoch'+str(epoch)+'_eval_label.pkl')
            pickle_dump_large_file(epoch_eval_argmax, 'epoch' + str(epoch) + '_eval_argmax.pkl')



            # epoch_eval_label = []
            # epoch_eval_logits = []
            # epoch_eval_nodes_mask = []
            # epoch_eval_logits_2 = []
            # for i_batch_dev, sample_batched_dev in enumerate(self.dev_data_loader):
            #     input_ids_dev = sample_batched_dev['sentences_input_ids'].to(self.device)
            #     input_mask_dev = sample_batched_dev['sentences_input_mask'].to(self.device)
            #     segment_id_dev = sample_batched_dev['sentences_segment_ids'].to(self.device)
            #     valid_ids_dev = sample_batched_dev['sentences_valid_ids'].to(self.device)
            #     adj_dev = sample_batched_dev['adj'].to(self.device)
            #     labels_dev = sample_batched_dev['labels'].to(self.device)
            #     nodes_mask_dev = sample_batched_dev['nodes_mask'].to(self.device)
            #     with torch.no_grad():
            #         logits = self.model(input_ids_dev,input_mask_dev, segment_id_dev, valid_ids_dev,adj_dev, nodes_mask_dev)
            #     softmax_logits = F.softmax(logits, dim=2)
            #     logit_max = torch.argmax(softmax_logits, dim=2)
            #     print('labels shape: ',labels_dev.shape)
            #     print('logit max: ', logit_max)
            #     epoch_eval_label.append(labels_dev.cpu().numpy())
            #     epoch_eval_logits.append(logit_max.cpu().numpy())
            #     epoch_eval_nodes_mask.append(nodes_mask_dev.cpu().numpy())
            #     epoch_eval_logits_2.append(softmax_logits.cpu().numpy())
            # pickle_dump_large_file(epoch_eval_nodes_mask, 'epoch'+str(epoch)+'_eval_nodes_mask.pkl')
            # pickle_dump_large_file(epoch_eval_logits, 'epoch'+str(epoch)+'_eval_logits')
            # pickle_dump_large_file(epoch_eval_label, 'epoch'+str(epoch)+'_eval_label')
            # pickle_dump_large_file(epoch_eval_logits_2, 'epoch'+str(epoch)+'_eval_logits_2')
    def fit(self):
        """
        Trains the model and finally evaluates on test
        Parameters
        ----------
        sess:		Tensorflow session object
        Returns
        -------
        """
        self.save_path = os.path.join(self.p.save_dir, 'best_int_avg')

        self.best_val, self.best_test = 0.0, 0.0

        if self.p.restore:
            self.saver.restore(self.save_path)

        self.run_epoch(self.p.max_epochs, self.loss_function)

        #print('Best Valid: {}, Best Test: {}'.format(self.best_val, self.best_test))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='GNN for NLP tutorial - Kipf GCN')

    parser.add_argument('--dataset', dest="dataset", default='/home/zeyuzhang/PycharmProjects/gcn_eprg/processed_data/worldtree_final', help='Dataset to use')
    parser.add_argument('--gpu', dest="gpu", default='0', help='GPU to use')
    parser.add_argument('--name', dest="name", default='test', help='Name of the run')

    parser.add_argument('--lr', dest="lr", default=5e-5, type=float, help='Learning rate')
    parser.add_argument('--max_epochs', dest="max_epochs", default=100, type=int, help='Max epochs')
    parser.add_argument('--l2', dest="l2", default=1e-8, type=float, help='L2 regularization')
    parser.add_argument('--seed', dest="seed", default=1234, type=int, help='Seed for randomization')
    parser.add_argument('--opt', dest="opt", default='adam', help='Optimizer to use for training')
    parser.add_argument('--batch_size', dest="batch_size", default=4, type=int, help='batch size for training')
    #parser.add_argument('--emb_dim', dest="emb_dim", default=128, type=int, help='BERT pre-trained embedding dimension')
    parser.add_argument('--num_class', dest="num_class", default=4, type=int, help='number of classes')

    parser.add_argument("--bert_model", default='bert-base-uncased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--do_lower_case",
                        default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--max_seq_length', dest="max_seq_length", default=128, type=int, help='Max sequenth length for BERT')
    # GCN-related params
    parser.add_argument('--gcn_dim', dest="gcn_dim", default=128, type=int, help='GCN hidden dimension')
    parser.add_argument('--drop', dest="dropout", default=0.5, type=float, help='Dropout for full connected layer')

    parser.add_argument('--restore', dest="restore", action='store_true',
                        help='Restore from the previous best saved model')
    parser.add_argument('--log_dir', dest="log_dir", default='./log/', help='Log directory')
    parser.add_argument('--config_dir', dest="config_dir", default='./config/', help='Config directory')
    parser.add_argument('--model_dir', dest="model_dir", default='./models/', help='Model directory')

    args = parser.parse_args()

    if not args.restore: args.name = args.name + '_' + time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create Model
    model = Main(args)
    model.fit()
    print('Model Trained Successfully!!')