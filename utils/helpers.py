import json
import os
from pathlib import Path

def load_data(data_dir, usage: str):
    assert usage in ["train", "val", "test"], "Usage must be one of train, val, test"
    assert os.path.exists(data_dir), "Data directory does not exist"
    src_file = open(f"transformer-translation/data/{usage}_src.txt", "w", encoding="utf-8")
    tgt_file = open(f"transformer-translation/data/{usage}_tgt.txt", "w", encoding="utf-8")

    for file in os.listdir(data_dir):
        if file.endswith(".json"):
            with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
                data = json.load(f)

                for item in data:
                    # item is a dictionary
                    src = item["with_punctuation"].strip()
                    tgt = item["translation"].strip()

                    if src and tgt:
                        src_file.write(src + "\n")
                        tgt_file.write(tgt + "\n")

    src_file.close()
    tgt_file.close()
    print(f"Completed! Data saved to {usage}_src.txt and {usage}_tgt.txt")

def train_val_split(source_dir, split_index=4000):
    assert os.path.exists(source_dir), "Source directory does not exist"
    train_src_file = open("transformer-translation/data/train_src.txt", "w", encoding="utf-8")
    train_tgt_file = open("transformer-translation/data/train_tgt.txt", "w", encoding="utf-8")

    val_src_file = open("transformer-translation/data/val_src.txt", "w", encoding="utf-8")
    val_tgt_file = open("transformer-translation/data/val_tgt.txt", "w", encoding="utf-8")

    for i, file in enumerate(sorted(os.listdir(source_dir))):
        if file.endswith(".json"):
            with open(os.path.join(source_dir, file), "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    src = item["with_punctuation"]
                    tgt = item["translation"]
                    if i < split_index:
                        train_src_file.write(src + "\n")
                        train_tgt_file.write(tgt + "\n")
                    else:
                        val_src_file.write(src + "\n")
                        val_tgt_file.write(tgt + "\n")
    
    train_src_file.close()
    train_tgt_file.close()
    val_src_file.close()
    val_tgt_file.close()
    print("Success!")

if __name__ == "__main__":
    # generate testing data
    # test_dir = "/Users/georgewu/Desktop/Transformer/transformer-translation/data/Date0524"
    # load_data(test_dir, "test")

    # generate train and test data
    dir = "/Users/georgewu/Desktop/Transformer/transformer-translation/data/Date0525"
    train_dir = "/Users/georgewu/Desktop/Transformer/transformer-translation/data/"
    train_val_split(dir)

