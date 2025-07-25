from model.model import Transformer
from utils.dataloader import CustomDataset
from torch.utils.data import DataLoader
from .train_config import config
import torch
import torch.nn as nn
from utils.tokenizer import tokenizer
from tqdm import tqdm
from pathlib import Path


def setup():
    """
    Modular function to handle data loading, model and optimizer initialization,
    and loading from a checkpoint if it exists.
    """
    # 1. --- Define Relative Paths ---
    # Assumes this script is in a 'scripts' directory, so we go up two levels
    # to the project root 'transformer-translation/'.
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    train_src_dir = PROJECT_ROOT / "data" / "train_src.txt"
    train_tgt_dir = PROJECT_ROOT / "data" / "train_tgt.txt"
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "model.pth"

    # Ensure the checkpoints directory exists
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    train_data = CustomDataset(train_src_dir, train_tgt_dir)
    train_loader = DataLoader(
        train_data, 
        batch_size=config["batch_size"], 
        shuffle=config["shuffle"]
    )

    print("Data Loaded Successfully!")
    print("Num batches: ", len(train_loader))

    # 3. --- Model Initialization ---
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
        print("train sample: ", batch["encoder_input"][0].shape)
        print("target sequence: ", batch["labels"][0].shape)
        break

    model = Transformer(
        src_pad_idx=pad_idx,
        trg_pad_idx=pad_idx, 
        enc_vocab_size=vocab_size,
        dec_vocab_size=vocab_size, 
        seq_len=max_len,
        d_model=config["d_model"],
        n_head=config["n_head"],
        ffn_hidden=config["ffn_hidden"],
        n_layer=config["n_layer"],
        drop_prob=config["drop_prob"]
    )
    
    # 4. --- Optimizer and Loss Function ---
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)

    # 5. --- Checkpoint Loading ---
    start_epoch = 0
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # We start from the epoch AFTER the one that was saved
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("No checkpoint found, starting training from scratch.")

    return model, optimizer, loss_fn, train_loader, config, checkpoint_path, start_epoch


def train(model, optim, loss_fn, config, dataset, checkpoint_path, start_epoch=0):
    # torch.autograd.set_detect_anomaly(True)
    model.train()
    epochs = config["epoch"]
    for epoch in range(start_epoch, epochs):
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
                return

            loss = loss_fn(output, label)

            # update loss in tqdm progress bar
            dataset.set_postfix({f"loss": f"{loss.item():6.3f}"})
            
            loss.backward()
            optim.step()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict()
        }, checkpoint_path)
        print(f"Successfully saved model checkpoint to {checkpoint_path}")
    print("Training successful!")

if __name__ == '__main__':

    model, optim, loss_fn, dataset, config, checkpoint_path, start_epoch = setup()

    train(model, optim, loss_fn, config, dataset, checkpoint_path, start_epoch=start_epoch)