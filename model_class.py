import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from pytorch_transformers import (BertModel, BertTokenizer)
from torch_geometric.nn import GCNConv
from pointer_network import Encoder, Attention, masked_log_softmax, masked_max
#from new_pointer_network import Encoder, Decoder
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #print(self.in_features)
        #print('out: ',self.out_features)
        self.weight = nn.Parameter(torch.zeros(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, text, adj):
        #print('weight: ',self.weight)
        hidden = torch.matmul(text, self.weight)
        #print('weight: ', self.weight)
        # print('weight: ',self.weight.shape)
        #print('hidden: ', hidden)
        # print('hidden: ',hidden.shape)
        # print('adj: ',adj)
        #denom = torch.sum(adj, dim=2, keepdim=True) + 1

        #print('denom: ',denom)
        #print(denom.shape)
        #print(torch.matmul(adj,hidden))
        output = torch.matmul(adj, hidden)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

# class KipfGCN(torch.nn.Module):
#     def __init__(self, emb_dim, num_classes, gcn_dim):
#         super(KipfGCN, self).__init__()
#         self.emb_dim = emb_dim
#         self.conv1 = GCNConv(self.emb_dim, gcn_dim, cached=True)
#         self.conv2 = GCNConv(gcn_dim, num_classes, cached=True)
#
#     def forward(self, x, edge_index):
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=0.3, training=self.training)
#         x = self.conv2(x, edge_index)
#         return x

class bertGCN(nn.Module):
    def __init__(self, params, bert_model, device):
        super(bertGCN, self).__init__()
        self.device = device
        self.p = params
        self.bert_hidden_size = 768
        self.bert = BertModel.from_pretrained(bert_model).to(self.device)
        self.gc1 = GraphConvolution(self.bert_hidden_size, 300)
        self.gc2 = GraphConvolution(300, self.p.gcn_dim)
        #self.gcn = KipfGCN(self.bert_hidden_size, num_classes=4, gcn_dim=self.p.gcn_dim)
        self.fc = nn.Linear(self.p.gcn_dim, 2)
        self.text_embed_dropout = nn.Dropout(0.3)

    def forward(self, input_ids,segment_id, input_mask, valid_ids,adj, nodes_mask):
        #print(input_ids.shape)
        #print(adj.shape)
        #print('nodes_mask: ',nodes_mask)

        out_flag = 0
        for i in range(input_ids.shape[0]):
            flag = 0
            for j in range(math.ceil(input_ids.shape[1]/2)):
                #print('single inputs ids shape: ', input_ids[i][j*2:(j+1)*2].shape)
                # print('input ids: ',input_ids[i][j*2:(j+1)*2])
                # print('segment ids: ', segment_id[i][j*2:(j+1)*2])
                # print('input mask: ', input_mask[i][j*2:(j+1)*2])
                sequence_output = self.bert(input_ids[i][j*2:(j+1)*2], token_type_ids=segment_id[i][j*2:(j+1)*2], attention_mask=input_mask[i][j*2:(j+1)*2], head_mask=None)[0]
                # print('sequence output: ',sequence_output)
                # print(sequence_output.shape)
                #sequence_output_mean = torch.mean(sequence_output, dim=1)
                sequence_output_cls = sequence_output[:,0,:]
                # print('sequence_output_cls: ',sequence_output_cls)
                # print(sequence_output_cls.shape)
                if flag == 0:
                    final_sequence_output = sequence_output_cls
                    flag = 1
                else:
                    final_sequence_output=torch.cat((final_sequence_output, sequence_output_cls),dim=0)
                #print(final_sequence_output.shape)
            if out_flag == 0:
                batch_final_sequence_output = final_sequence_output
                out_flag = 1
            else:
                batch_final_sequence_output = torch.cat((batch_final_sequence_output, final_sequence_output),dim=0)
        embedding = self.text_embed_dropout(batch_final_sequence_output)
        #embedding = batch_final_sequence_output.unsqueeze(0) * nodes_mask.unsqueeze(2)
        ##[102, 768]
        #print(batch_final_sequence_output.shape)
        ##[1,102]
        #print(nodes_mask.shape)
        #print('embedding: ',batch_final_sequence_output)
        # nodes_mask = nodes_mask.squeeze(0).unsqueeze(1)
        # nodes_mask = nodes_mask.expand(-1, 768)
        # embedding = nodes_mask*batch_final_sequence_output
        # print(batch_final_sequence_output)
        # print(batch_final_sequence_output.shape)
        # print(torch.sum(batch_final_sequence_output, dim=1).shape)
        #print('embedding: ',embedding.shape)
        x1 = F.relu(self.gc1(embedding, adj))
        #print('x1: ',x1.shape)
        #x1= x1*nodes_mask.unsqueeze(2)
        x2 = F.relu(self.gc2(x1, adj))
        #print('x2: ',x2.shape)
        #x2=x2*nodes_mask.unsqueeze(2)
        output = self.fc(x2)
        return output

class bertGCN_Pointer(nn.Module):
    def __init__(self, params, bert_model, device):
        super(bertGCN_Pointer, self).__init__()
        self.device = device
        self.p = params
        self.bert_hidden_size = 768
        self.bert = BertModel.from_pretrained(bert_model).to(self.device)
        self.text_embed_dropout = nn.Dropout(0.3)
        #self.gc1 = GraphConvolution(self.bert_hidden_size, 300)
        #self.gc2 = GraphConvolution(300, self.p.gcn_dim)
        #self.gcn = KipfGCN(self.bert_hidden_size, num_classes=4, gcn_dim=self.p.gcn_dim)
        self.hidden_size = 256
        #self.fc = nn.Linear(self.p.gcn_dim, 128)
        self.fc = nn.Linear(768, 256)
        self.bidirectional = False
        self.num_directions = 2 if self.bidirectional else 1
        self.num_layers = 1
        self.batch_first = True
        self.encoder = Encoder(embedding_dim=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                               bidirectional=self.bidirectional, batch_first=self.batch_first)
        self.decoding_rnn = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size)
        self.attn = Attention(hidden_size=self.hidden_size)
        # self.decoder_input0 = nn.Parameter(torch.zeros((10, self.hidden_size)), requires_grad=False)
        # nn.init.uniform_(self.decoder_input0, -1, 1)
        self.max_seq_len = torch.tensor([102],dtype=torch.long).to(self.device)

    def reset_first_hidden(self, batch_size, hidden_size):
        decoder_input0 = nn.Parameter(torch.zeros((batch_size, hidden_size)), requires_grad=False).to(self.device)
        nn.init.uniform_(decoder_input0, -1, 1)
        return decoder_input0
    def forward(self, input_ids,segment_id, input_mask, valid_ids,adj, nodes_mask, input_lengths):
        #print(input_ids.shape)
        #print(adj.shape)
        #print('nodes_mask: ',nodes_mask)

        out_flag = 0
        for i in range(input_ids.shape[0]):
            flag = 0
            for j in range(math.ceil(input_ids.shape[1]/2)):
                #print('single inputs ids shape: ', input_ids[i][j*2:(j+1)*2].shape)
                # print('input ids: ',input_ids[i][j*2:(j+1)*2])
                # print('segment ids: ', segment_id[i][j*2:(j+1)*2])
                # print('input mask: ', input_mask[i][j*2:(j+1)*2])
                sequence_output = self.bert(input_ids[i][j*2:(j+1)*2], token_type_ids=segment_id[i][j*2:(j+1)*2], attention_mask=input_mask[i][j*2:(j+1)*2], head_mask=None)[0]
                # print('sequence output: ',sequence_output)
                # print(sequence_output.shape)
                #sequence_output_mean = torch.mean(sequence_output, dim=1)
                sequence_output_cls = sequence_output[:,0,:]
                # print('sequence_output_cls: ',sequence_output_cls)
                # print(sequence_output_cls.shape)
                if flag == 0:
                    final_sequence_output = sequence_output_cls
                    flag = 1
                else:
                    final_sequence_output=torch.cat((final_sequence_output, sequence_output_cls),dim=0)
                #print(final_sequence_output.shape)
            #print('final_sequence_output: ',final_sequence_output.shape)
            if out_flag == 0:
                batch_final_sequence_output = final_sequence_output.unsqueeze(0)
                out_flag = 1
            else:
                batch_final_sequence_output = torch.cat((batch_final_sequence_output, final_sequence_output.unsqueeze(0)),dim=0)
        #embedding = batch_final_sequence_output.unsqueeze(0) * nodes_mask.unsqueeze(2)
        #embedding = self.text_embed_dropout(embedding)
        #print('input is: ',input_ids.shape[1])
        #print(batch_final_sequence_output.shape)
        #print(nodes_mask.shape)
        embedding = batch_final_sequence_output * nodes_mask.unsqueeze(2)
        #embedding = batch_final_sequence_output.unsqueeze(0) * nodes_mask.unsqueeze(2)

        # x1 = F.relu(self.gc1(embedding, adj))
        # x1 = x1*nodes_mask.unsqueeze(2)
        # x2 = F.relu(self.gc2(x1, adj))
        # x2 = x2
        # embedded = self.fc(x2)
        #embedded = embedded*nodes_mask.unsqueeze(2)
        embedded = self.fc(embedding)
        if self.batch_first:
            batch_size = nodes_mask.size(0)
            max_seq_len = nodes_mask.size(1)
        else:
            batch_size = nodes_mask.size(1)
            max_seq_len = nodes_mask.size(0)
        #print('embedded: ',embedded.shape)
        #print('input_length: ',input_lengths.shape)
        #self.encoder.flatten_parameters()

        #encoder_outputs, encoder_hidden = self.encoder(embedded, input_lengths)
        #print('embedded: ',embedded.shape)
        #print('input length: ',input_lengths.shape)
        encoder_outputs, encoder_hidden = self.encoder(embedded, input_lengths)

        #print('encoder_outputs: ',encoder_outputs)
        #[1,102,256]
        #print('encoder_outputs: ', encoder_outputs.shape)
        #print('encoder_hidden_n: ',encoder_hidden[0])
        #[1,1,256]
        #print('encoder_hidden_n: ', encoder_hidden[0].shape)
        #[1,1,256]
        #print('encoder_hidden_n: ', encoder_hidden[1].shape)
        if self.bidirectional:
            # Optionally, Sum bidirectional RNN outputs
            encoder_outputs = encoder_outputs[:, :, :self.hidden_size] + encoder_outputs[:, :, self.hidden_size:]

        encoder_h_n, encoder_c_n = encoder_hidden
        encoder_h_n = encoder_h_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        encoder_c_n = encoder_c_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)

        # Lets use zeros as an intial input for sorting example
        #decoder_input=self.decoder_input0
        #print('decoder input: ',decoder_input.shape)
        #decoder_input = encoder_outputs.new_zeros(torch.Size((batch_size, self.hidden_size)))
        decoder_input = self.reset_first_hidden(batch_size, self.hidden_size)
        if decoder_input.size(0) == 1:
            decoder_hidden = (encoder_h_n[-1, 0, :, :].squeeze().unsqueeze(0), encoder_c_n[-1, 0, :, :].squeeze().unsqueeze(0))
        else:
            decoder_hidden = (encoder_h_n[-1, 0, :, :].squeeze(), encoder_c_n[-1, 0, :, :].squeeze())
        #print('decoder_hidden shape: ',decoder_hidden[0].shape)
        #[1,256]
        #print('decoder_input: ',decoder_input.shape)
        #print('decoder_input: ', decoder_input)
        # [1,256]
        #print('decoder_hidden: ',decoder_hidden[0].shape)
        #print('decoder_hidden: ', decoder_hidden[0])
        # [1,256]
        # print('decoder_hidden: ', decoder_hidden[1].shape)
        #print('decoder_hidden: ', decoder_hidden[1])
        #print('input length: ',input_lengths)
        range_tensor = torch.arange(max_seq_len, device=input_lengths.device, dtype=input_lengths.dtype).expand(
            batch_size, max_seq_len, max_seq_len)
        #print('range tensor: ',range_tensor)
        each_len_tensor = input_lengths.view(-1, 1, 1).expand(batch_size, max_seq_len, max_seq_len)
        #print('each_len_tensor: ',each_len_tensor)
        row_mask_tensor = (range_tensor < each_len_tensor)
        #print('row mask tensor: ', row_mask_tensor.shape)
        #print('row mask tensor: ',row_mask_tensor)
        col_mask_tensor = row_mask_tensor.transpose(1, 2)
        #print('col mask tensor: ', col_mask_tensor.shape)
        #print('col mask tensor: ',col_mask_tensor)
        mask_tensor = row_mask_tensor * col_mask_tensor
        #print('mask tensor: ', mask_tensor.shape)
        #print('mask tensor: ',mask_tensor)

        pointer_log_scores = []
        pointer_argmaxs = []

        for i in range(max_seq_len):
            # We will simply mask out when calculating attention or max (and loss later)
            # not all input and hiddens, just for simplicity
            sub_mask = mask_tensor[:, i, :].float()

            # h, c: (batch_size, hidden_size)
            #self.decoding_rnn.flatten_parameters()
            #print('decoder input: ',decoder_input.shape)
            #print('decoder hideen: ',decoder_hidden[0].shape)
            #print('decoder hideen: ', decoder_hidden[1].shape)
            h_i, c_i = self.decoding_rnn(decoder_input, decoder_hidden)
            # print('decoder hi: ',h_i)
            # print('decoder hi: ', h_i.shape)
            # print('decoder ci: ', c_i)
            # print('decoder ci: ', c_i)

            # next hidden
            decoder_hidden = (h_i, c_i)

            # Get a pointer distribution over the encoder outputs using attention
            # (batch_size, max_seq_len)
            # print('h_i: ',h_i.shape)
            # print('encoder_outputs: ',encoder_outputs.shape)
            # print('sub_mask: ',sub_mask.shape)
            log_pointer_score = self.attn(h_i, encoder_outputs, sub_mask)
            pointer_log_scores.append(log_pointer_score)

            # Get the indices of maximum pointer
            _, masked_argmax = masked_max(log_pointer_score, sub_mask, dim=1, keepdim=True)

            pointer_argmaxs.append(masked_argmax)
            index_tensor = masked_argmax.unsqueeze(-1).expand(batch_size, 1, self.hidden_size)

            # (batch_size, hidden_size)
            decoder_input = torch.gather(encoder_outputs, dim=1, index=index_tensor).squeeze(1)

        pointer_log_scores = torch.stack(pointer_log_scores, 1)
        pointer_argmaxs = torch.cat(pointer_argmaxs, 1)
        #print('pointer log scores: ',pointer_log_scores.shape)
        return pointer_log_scores, pointer_argmaxs, mask_tensor

