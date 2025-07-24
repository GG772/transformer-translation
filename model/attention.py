import torch
import torch.nn as nn
import torch.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_head = n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_final = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        _, _, d_model = q.shape
        assert(d_model == self.d_model)
        n_d = self.d_model // self.n_head
        # broadcasting weights 
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # now split into different attention heads
        # put n_d in the last dimension since we want to take the softmax of the features
        q = q.view(q.shape[0], q.shape[1], self.n_head, n_d).permute(0, 2, 1, 3)
        k = k.view(k.shape[0], k.shape[1], self.n_head, n_d).permute(0, 2, 1, 3)
        v = v.view(v.shape[0], v.shape[1], self.n_head, n_d).permute(0, 2, 1, 3)

        # now, put each into a attention head
        # score has shape (batch, n_heads, len_q, len_k)
        score = q @ k.transpose(2, 3) / math.sqrt(n_d)
        if mask is not None:
            # mask = torch.tril(torch.ones(seq_len, seq_len, dtype=bool))
            score = score.masked_fill(mask == 0, -1e9)
        # score has shape (batch, n_heads, seq_len, n_d)
        score = self.softmax(score) @ v

        # to view after permute, we have to make the tensor contiguous in memory
        # d_model = self.n_head * self.n_d
        score = score.permute(0, 2, 1, 3).contiguous().view(score.shape[0], -1, d_model)

        return self.w_final(score)

if __name__ == '__main__':
    # for testing
    # (batch, seq_len, d_model)
    X = torch.randn(128, 64, 512) 

    d_model = 512
    n_head = 8
    attention = MultiHeadAttention(d_model, n_head)
    res = attention(X, X, X)
    print(res, res.shape)

    has_nan = res.isnan().any()

    print(has_nan)  # Outputs: tensor(True)

