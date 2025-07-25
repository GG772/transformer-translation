import json
import os
from pathlib import Path

def process_raw_json(source_dir: Path, output_dir: Path, usage: str):
    """
    Reads all .json files from a source directory, extracts text pairs,
    and writes them to source and target .txt files.

    Args:
        source_dir: The directory containing the raw .json files.
        output_dir: The directory where the output files will be saved (e.g., '.../data/').
        usage: A string to name the output files (e.g., 'train', 'val', 'test').
    """
    assert usage in ["train", "val", "test"], "Usage must be one of 'train', 'val', or 'test'"
    if not source_dir.exists():
        print(f"Warning: Source directory not found at {source_dir}. Skipping.")
        return

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    src_output_path = output_dir / f"{usage}_src.txt"
    tgt_output_path = output_dir / f"{usage}_tgt.txt"

    print(f"Processing files from {source_dir}...")
    # Use 'with' to ensure files are closed automatically
    with open(src_output_path, "w", encoding="utf-8") as src_file, \
         open(tgt_output_path, "w", encoding="utf-8") as tgt_file:

        # Iterate through all files in the source directory
        for filename in sorted(os.listdir(source_dir)):
            if filename.endswith(".json"):
                file_path = source_dir / filename
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        src = item.get("with_punctuation", "").strip()
                        tgt = item.get("translation", "").strip()
                        if src and tgt:
                            src_file.write(src + "\n")
                            tgt_file.write(tgt + "\n")
    
    print(f"Completed! Data saved to {src_output_path.name} and {tgt_output_path.name}")


def create_train_val_split(source_dir: Path, output_dir: Path, split_file_count=4000):
    """
    Splits .json files from a source directory into training and validation sets
    based on the file's sort order index.

    Args:
        source_dir: The directory containing the .json files to split.
        output_dir: The directory where train and val .txt files will be saved.
        split_file_count: The number of files to include in the training set.
    """
    if not source_dir.exists():
        print(f"Warning: Source directory not found at {source_dir}. Skipping split.")
        return

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    train_src_path = output_dir / "train_src.txt"
    train_tgt_path = output_dir / "train_tgt.txt"
    val_src_path = output_dir / "val_src.txt"
    val_tgt_path = output_dir / "val_tgt.txt"
    
    print(f"Splitting files from {source_dir} into training and validation sets...")
    
    json_files = sorted([f for f in os.listdir(source_dir) if f.endswith(".json")])

    # Use 'with' to manage all four output files
    with open(train_src_path, "w", encoding="utf-8") as train_src_file, \
         open(train_tgt_path, "w", encoding="utf-8") as train_tgt_file, \
         open(val_src_path, "w", encoding="utf-8") as val_src_file, \
         open(val_tgt_path, "w", encoding="utf-8") as val_tgt_file:

        for i, filename in enumerate(json_files):
            file_path = source_dir / filename
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    src = item.get("with_punctuation", "").strip()
                    tgt = item.get("translation", "").strip()
                    
                    if not (src and tgt):
                        continue

                    # Decide whether to write to train or val files based on file index
                    if i < split_file_count:
                        train_src_file.write(src + "\n")
                        train_tgt_file.write(tgt + "\n")
                    else:
                        val_src_file.write(src + "\n")
                        val_tgt_file.write(tgt + "\n")

    print("Train/validation split successful!")


def main():
    """
    Main function to define paths and run the data processing scripts.
    """
    # Define project root relative to this script's location.
    # Assumes this script is in a 'utils' or 'scripts' directory.
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data"

    # --- Task 1: Generate testing data from a specific folder ---
    # Define the source directory for test data relative to the main data folder
    test_source_dir = DATA_DIR / "Date0524"
    process_raw_json(source_dir=test_source_dir, output_dir=DATA_DIR, usage="test")

    # --- Task 2: Generate training and validation data from another folder ---
    # Define the source directory for the train/val split
    train_val_source_dir = DATA_DIR / "Date0525"
    create_train_val_split(source_dir=train_val_source_dir, output_dir=DATA_DIR)


if __name__ == "__main__":
    main()
