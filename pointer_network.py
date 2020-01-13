import torch
import torch.nn as nn

def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)
def masked_max(vector: torch.Tensor, mask: torch.Tensor, dim: int, keepdim: bool = False, min_val: float = -1e7) -> (torch.Tensor, torch.Tensor):
    one_minus_mask = (1.0 - mask).bool()
    replaced_vector = vector.masked_fill(one_minus_mask, min_val)
    max_value, max_index = replaced_vector.max(dim=dim, keepdim=keepdim)
    return max_value, max_index

class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layers=1, batch_first=True, bidirectional=True):
        super(Encoder, self).__init__()

        self.batch_first = batch_first
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                           batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, embedded_inputs, input_lengths):
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded_inputs, input_lengths, batch_first=self.batch_first)
        # Forward pass through RNN
        self.rnn.flatten_parameters()
        outputs, hidden = self.rnn(packed)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=self.batch_first)
        # Return output and final hidden state
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.vt = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs, mask):
        # (batch_size, max_seq_len, hidden_size)
        encoder_transform = self.W1(encoder_outputs)

        # (batch_size, 1 (unsqueezed), hidden_size)
        decoder_transform = self.W2(decoder_state).unsqueeze(1)

        # 1st line of Eq.(3) in the paper
        # (batch_size, max_seq_len, 1) => (batch_size, max_seq_len)
        u_i = self.vt(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)

        # softmax with only valid inputs, excluding zero padded parts
        # log-softmax for a better numerical stability
        #print('u_i: ',u_i.shape)
        #print('mask: ',mask.shape)
        log_score = masked_log_softmax(u_i, mask, dim=-1)

        return log_score