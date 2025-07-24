import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super().__init__(vocab_size, d_model)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        self.encoding = torch.zeros(seq_len, d_model)
        self.encoding.requires_grad = False

        pos = torch.arange(0, seq_len)
        pos = pos.float().unsqueeze(1)

        ind = torch.arange(0, d_model, 2)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (ind / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (ind / d_model)))

    def forward(self, x):
        # x has shape (batch, seq_len)
        seq_len = x.shape[1]
        return self.encoding[:seq_len, :]

    
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-10):
        super().__init__()

        # gamma and beta are learnable parameters that shifts normalization
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

        self.eps = eps

    def forward(self, x):
        # shape of x: (batch, seq_len, d_model)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        var = torch.clamp(var, min=self.eps)


        norm = (x - mean) / torch.sqrt(var + self.eps)
        shifted_norm = self.gamma * norm + self.beta
        return shifted_norm

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        res = self.fc1(x)

        res = F.relu(res)
        res = self.dropout(res)
        res = self.fc2(res)
        
        return res
    
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, seq_len, drop_prob):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.drop_prob = drop_prob

        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(d_model, seq_len)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        res = self.drop_out(tok_emb + pos_emb)
        return res
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.drop1 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, dropout=drop_prob)
        self.norm2 = LayerNorm(d_model)
        self.drop2 = nn.Dropout(drop_prob)

    def forward(self, x, mask=None):
        # self attention
        # for residual connection
        _x = x
        x = self.attention(x, x, x, mask)

        x = self.drop1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.drop2(x)
        x = self.norm2(x + _x)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super().__init__()  

        # masked attention
        self.attention1 = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        # cross attention
        self.attention2 = MultiHeadAttention(d_model, n_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, dropout=drop_prob)
        self.dropout3 = nn.Dropout(drop_prob)
        self.norm3 = LayerNorm(d_model)

    def forward(self, dec, enc, pad_mask, mask):
        # enc are key, value matrices from encoder
        # pad_mask: mask for padding
        # mask: regular lower_triangular masking

        # first residual connection
        x = dec
        _x = x
        x = self.attention1(x, x, x, mask=mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # second residual connection
        if enc is not None:
            _x = x
            # cross attention
            x = self.attention2(x, enc, enc, mask=pad_mask)
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # last residual connections
        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + _x)

        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model, ffn_hidden, n_head, n_layer, drop_prob):
        super().__init__()
        self.emb = TransformerEmbedding(vocab_size, d_model, seq_len, drop_prob=drop_prob)
        
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, ffn_hidden, n_head, drop_prob) for _ in range(n_layer)]
        )
    
    def forward(self, x, pad_mask):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, mask=pad_mask)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model, ffn_hidden, n_head, n_layer, drop_prob):
        super().__init__()
        self.emb = TransformerEmbedding(vocab_size, d_model, seq_len, drop_prob=drop_prob)
        
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, ffn_hidden, n_head, drop_prob) for _ in range(n_layer)]
        )

        self.fc = nn.Linear(d_model, vocab_size)
        # self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, dec, enc, pad_mask, mask):
        x = dec
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, enc, pad_mask, mask)

        x = self.fc(x)
        # x = self.softmax(x)

        return x    

class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, 
                 enc_vocab_size, dec_vocab_size, seq_len, 
                 d_model, n_head, ffn_hidden, n_layer, drop_prob):
        super().__init__()
        self.encoder = Encoder(enc_vocab_size, seq_len, d_model, ffn_hidden, n_head, n_layer, drop_prob)
        self.decoder = Decoder(dec_vocab_size, seq_len, d_model, ffn_hidden, n_head, n_layer, drop_prob)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_causal_mask(self, q, k):
        # q has shape (batch, len_q)
        # k has shape (batch, len_k)
        # 2D dimension since q, k are lists that store indices of token
        # and we don't bother the features of each token

        len_q, len_k = q.size(1), k.size(1)

        # score has shape (batch, n_heads, len_q, len_k) since score = QK^T
        # 4 dimensions since we are using multi-head
        # our mask need to have the same shape as score
        mask = torch.tril(torch.ones((len_q, len_k))).type(torch.BoolTensor)
        return mask
    
    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):
        # q has shape (batch, len_q)
        # k has shape (batch, len_k)
        # 2D dimension since q, k are lists that store indices of token
        # and we don't bother the features of each token

        # in self attention, q = k
        # in cross attention, q is from decoder and k is from encoder

        len_q, len_k = q.size(1), k.size(1)
        # ne : not equal

        # in this line, q has shape (batch_size, 1, len_q, 1)
        q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        # in this line, q became (batch_size, 1, len_q, len_k)
        q = q.repeat(1, 1, 1, len_k)

        # in this line, k has shape (batch_size, 1, 1, len_k)
        k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        # in this line, k has shape (batch_size, 1, len_q, len_k)
        k = k.repeat(1, 1, len_q, 1)

        mask = q & k

        # mask has shape (batch_size, 1, len_q, len_k)
        # since score has shape (batch_size, n_heads, len_q, len_k)
        # we can utilize broadcasting
        return mask
    
    def forward(self, src, trg):
        # src, trg has shape (batch, len_src) and (batch, len_trg)
        # they are then transformed into (batch, len_src, d_model) and (batch, len_trg, d_model)
        # by embedding layers
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        trg_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx) * self.make_causal_mask(trg, trg)

        # cross attention
        src_trg_mask = self.make_pad_mask(trg, src, self.trg_pad_idx, self.src_pad_idx)

        enc = self.encoder(src, src_mask)
        dec = self.decoder(trg, enc, trg_mask, src_trg_mask)

        return dec


