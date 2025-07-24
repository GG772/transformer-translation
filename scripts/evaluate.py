from model.model import Transformer
from utils.dataloader import CustomDataset
from torch.utils.data import DataLoader
from .train_config import config
import torch
import torch.nn as nn
from utils.tokenizer import tokenizer
from tqdm import tqdm
from utils.dataloader import causal_mask

def greedy_decode(model, source, source_mask, tokenizer, max_len):
    bos_idx = tokenizer.convert_tokens_to_ids('[BOS]')
    eos_idx = tokenizer.convert_tokens_to_ids('[EOS]')

    encoder_output = model.encoder(source, source_mask)
    # Initialize the decoder input with bos token
    # (batch, seq_len)
    decoder_input = torch.empty(1, 1).fill_(bos_idx).type_as(source)
    while True:
        if decoder_input.size(1) >= max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).unsqueeze(0)

        out = model.decoder(decoder_input, encoder_output, source_mask, decoder_mask)

        # get the next token
        # prob has shape (1, vocab_size)
        prob = out[:, -1, :]
        # Select max prob in the vocab_size dimension
        _, next_word = torch.max(prob, dim=1)
        # concat in seq_len dimension
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item())], dim=1)

        if next_word.item() == eos_idx:
            break
    return decoder_input.squeeze(0)

def evaluate(model, dataset, tokenizer, max_len, num_examples=10):
    model.eval()
    count = 0

    console_width = 80

    with torch.no_grad():
        for batch in dataset:
            count += 1
            encoder_input = batch['encoder_input'] # (batch, seq_len)
            encoder_mask = batch['encoder_mask'] # (batch, 1, 1, seq_len)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation" 
            model_output = greedy_decode(model, encoder_input, encoder_mask, tokenizer, max_len)

            # use index 0 since each batch only has one sentence
            source_text = batch['src_text'][0] 
            target_text = batch['tgt_text'][0]
            # detach() tells pytorch to not keep track of gradient of model_output
            model_out_text = tokenizer.decode(model_output.detach().numpy())

            # 
            print("-"*console_width)
            print(f"SOURCE: {source_text}")
            print(f"TARGET: {target_text}")
            print(f"PREDICTED: {model_out_text}")

            if count == num_examples:
                break



if __name__ == '__main__':

    eval_src_dir = "/Users/georgewu/Desktop/Transformer/transformer-translation/data/val_src.txt"
    eval_tgt_dir = "/Users/georgewu/Desktop/Transformer/transformer-translation/data/val_tgt.txt"

    eval_data = CustomDataset(eval_src_dir, eval_tgt_dir)
    eval_loader = DataLoader(
        eval_data, 
        batch_size=1, 
        shuffle=config["shuffle"]
    )

    pad_idx = tokenizer.pad_token_id
    vocab_size = len(tokenizer)
    max_len = config["max_len"]

    # d_model, n_head, ffn_hidden, n_layer, drop_prob
    model = Transformer(
        pad_idx, pad_idx, 
        vocab_size, vocab_size, 
        max_len, config["d_model"],
        config["n_head"],
        config["ffn_hidden"],
        config["n_layer"],
        config["drop_prob"]
    )

    state = torch.load("/Users/georgewu/Desktop/Transformer/transformer-translation/checkpoints/model.pth")
    model.load_state_dict(state['model_state_dict'])

    print("Setup successful!")

    evaluate(model, eval_loader, tokenizer, config["max_len"])


