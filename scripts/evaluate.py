import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# Assuming these imports are correctly set up in your project structure
from model.model import Transformer
from utils.dataloader import CustomDataset, causal_mask
from .train_config import config # Assuming a shared config for model params
from utils.tokenizer import tokenizer

def setup_evaluation():
    """
    Modular function to handle data loading, model initialization,
    and loading a trained model from a checkpoint.
    """
    # 1. --- Define Relative Paths ---
    # Assumes this script is in a 'scripts' directory, so we go up two levels
    # to the project root 'transformer-translation/'.
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    eval_src_dir = PROJECT_ROOT / "data" / "val_src.txt"
    eval_tgt_dir = PROJECT_ROOT / "data" / "val_tgt.txt"
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "model.pth"

    # --- Error handling for checkpoint ---
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please train the model first.")

    # 2. --- Dataloader Setup ---
    # For evaluation, batch size must be 1 to process one sentence at a time.
    eval_data = CustomDataset(str(eval_src_dir), str(eval_tgt_dir))
    eval_loader = DataLoader(
        eval_data, 
        batch_size=1, 
        shuffle=False # No need to shuffle for evaluation
    )
    print("Validation data loaded successfully!")

    # 3. --- Model Initialization ---
    pad_idx = tokenizer.pad_token_id
    vocab_size = len(tokenizer)
    max_len = config["max_len"]

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

    # 4. --- Load Trained Model State ---
    print(f"Loading trained model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model state loaded successfully.")

    return model, eval_loader, tokenizer, config

def greedy_decode(model, source, source_mask, tokenizer_instance, max_len):
    """
    Performs greedy decoding for one sentence.
    """
    bos_token_id = tokenizer_instance.bos_token_id
    eos_token_id = tokenizer_instance.eos_token_id

    # Get the device from the model
    device = next(model.parameters()).device

    # Move source tensors to the correct device
    source = source.to(device)
    source_mask = source_mask.to(device)
    
    # Pre-compute the encoder output
    encoder_output = model.encoder(source, source_mask)

    # Initialize decoder input with the BOS token
    decoder_input = torch.empty(1, 1).fill_(bos_token_id).type_as(source).to(device)

    while True:
        # Stop if the generated sequence is too long
        if decoder_input.size(1) >= max_len:
            break

        # Create a causal mask for the decoder input
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        
        # Get the model's raw output (logits)
        out = model.decoder(decoder_input, encoder_output, source_mask, decoder_mask)
        
        # Get the logits for the very last token
        # prob has shape (batch, d_model)
        # since batch = 1 in evaluation, prob has shape (1, d_model)
        prob = out[:, -1, :]
        
        # Select the token with the highest probability (greedy search)
        _, next_word = torch.max(prob, dim=1)
        
        # Append the new token to the decoder input sequence
        # Concat along the seq_len dimension
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], 
            dim=1
        )
        
        # Stop if the EOS token is generated
        if next_word.item() == eos_token_id:
            break
            
    return decoder_input.squeeze(0)

def evaluate(model, dataset, tokenizer_instance, config, num_examples=10):
    """
    The main evaluation loop.
    """
    model.eval() # Set the model to evaluation mode
    count = 0
    console_width = 80

    with torch.no_grad(): # Disable gradient calculation for inference
        for batch in tqdm(dataset, desc="Evaluating"):
            count += 1
            if count > num_examples:
                break

            encoder_input = batch['encoder_input'] # (1, seq_len)
            encoder_mask = batch['encoder_mask']   # (1, 1, 1, seq_len)

            # Ensure batch size is 1 for evaluation
            assert encoder_input.size(0) == 1, "Batch size must be 1 for evaluation" 

            model_output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_instance, config["max_len"])

            # Decode the text from token IDs
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_instance.decode(model_output.detach().cpu().numpy())

            # Print the results
            print("-" * console_width)
            print(f"SOURCE:    {source_text}")
            print(f"TARGET:    {target_text}")
            print(f"PREDICTED: {model_out_text}")

    print(f"\nEvaluation finished. Displayed {num_examples} examples.")

if __name__ == '__main__':

    # Setup returns all necessary objects for evaluation
    model, eval_loader, tokenizer_instance, config_params = setup_evaluation()
    
    # Pass the objects to the evaluate function
    evaluate(model, eval_loader, tokenizer_instance, config_params)


