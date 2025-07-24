from torch.utils.data import Dataset, DataLoader
from .tokenizer import tokenizer
import torch

class CustomDataset(Dataset):
    def __init__(self, src_dir, tgt_dir, max_length=256):
        self.src_dir = src_dir
        self.tgt_dir = tgt_dir
        self.seq_len = max_length

        self.bos_token = torch.tensor([tokenizer.convert_tokens_to_ids("[BOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer.convert_tokens_to_ids("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer.convert_tokens_to_ids("[PAD]")], dtype=torch.int64)

        self.src, self.tgt = list(), list()
        with open(src_dir, "r", encoding="utf-8") as f:
            self.src = f.readlines()

        with open(tgt_dir, "r", encoding="utf-8") as f:
            self.tgt = f.readlines()

        self.src, self.tgt = zip(*[
            (x, y) for x, y in zip(self.src, self.tgt) if len(x) <= 250 and len(y) <= 250
        ])
        self.src = list(self.src)
        self.tgt = list(self.tgt)
        assert(len(self.src) == len(self.tgt))
        
    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        # For encoder-decoder training, we typically want:
        #   - encoder_input_ids
        #   - decoder_input_ids
        #   - labels (target shifted by 1 if teacher forcing is used)
        
        src_text = self.src[idx]
        tgt_text = self.tgt[idx]

        src_tokenized = tokenizer.encode(src_text, add_special_tokens=False)
        tgt_tokenized = tokenizer.encode(tgt_text, add_special_tokens=False)

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(src_tokenized) - 2  # We will add <s> and </s>
        # We will only add <s> on decoder input, and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(tgt_tokenized) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError(f"Sentence is too long, src/tgt len are {len(src_tokenized), len(tgt_tokenized)}")

        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.bos_token,
                torch.tensor(src_tokenized, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        decoder_input = torch.cat(
            [
                self.bos_token,
                torch.tensor(tgt_tokenized, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        labels = torch.cat(
            [
                torch.tensor(tgt_tokenized, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        return {
            "encoder_input": encoder_input, # (seq_len,)
            "decoder_input": decoder_input, # (seq_len,)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "labels": labels, # (seq_len,)
            "src_text": src_text,
            "tgt_text": tgt_text
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

