from model.model import Transformer
from utils.dataloader import CustomDataset
from torch.utils.data import DataLoader
from .train_config import config
import torch
import torch.nn as nn
from utils.tokenizer import tokenizer
from tqdm import tqdm


def train(model, optim, loss_fn, config, dataset):
    torch.autograd.set_detect_anomaly(True)
    model.train()
    epochs = config["epoch"]
    for epoch in range(epochs):
        dataset = tqdm(dataset, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in dataset:
            optim.zero_grad()

            encoder_input = batch["encoder_input"] # (Batch, seq_len)
            decoder_input = batch["decoder_input"] # (Batch, seq_len)

            label = batch["labels"] # (Batch, seq_len)

            output = model(encoder_input, decoder_input) # (Batch, seq_len, tgt_vocab_size)

            # output now has shape (Batch * seq_len, tgt_vocab_size)
            output = output.view(-1, output.size(-1))
            # label now has shape (Batch * seq_len)
            label = label.view(-1)

            # Debug: Check for NaN in model output
            if torch.isnan(output).any():
                print("NaN detected in model output!")
                print(output)
                break

            loss = loss_fn(output, label)

            # update loss in tqdm progress bar
            dataset.set_postfix({f"loss": f"{loss.item():6.3f}"})
            
            loss.backward()
            optim.step()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict()
        }, '/Users/georgewu/Desktop/Transformer/transformer-translation/checkpoints/model.pth')
        print("Successfully saved model to checkpoints/model.pth")
    print("training successful!")

if __name__ == '__main__':
    train_src_dir = "/Users/georgewu/Desktop/Transformer/transformer-translation/data/train_src.txt"
    train_tgt_dir = "/Users/georgewu/Desktop/Transformer/transformer-translation/data/train_tgt.txt"

    train_data = CustomDataset(train_src_dir, train_tgt_dir)

    train_loader = DataLoader(
        train_data, 
        batch_size=config["batch_size"], 
        shuffle=config["shuffle"]
    )

    print("Data Loaded Successfully!")
    print("Num samples: ", len(train_loader))

    pad_idx = tokenizer.pad_token_id
    vocab_size = len(tokenizer)
    max_len = config["max_len"]

    print("pad idx: ", pad_idx)
    print("vocab size:", vocab_size)
    print("seq_len (max_len): ", max_len)

    for batch in train_loader:
        print("Shape of input token: ", batch["encoder_input"].shape)
        print("Shape of target tokens: ", batch["labels"].shape)
        print("First (train, sample) tuple: ")
        print("train sample: ", batch["encoder_input"][0])
        print("target sequence: ", batch["labels"][0])
        break
    

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
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)

    train(model, optimizer, loss_fn, config, train_loader)