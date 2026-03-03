import glob
import os
import re
import zipfile
from typing import Optional

from utils import Tokenizer


def consolidate_source_texts() -> None:
    """Combines all source text files into one based on their prefix index.

    Finds all .txt files in data/source_texts/, sorts them numerically
    by their index prefix, and concatenates them with two newlines
    as a separator. Automatically unzips data.zip if source_texts is missing.
    """
    source_dir: str = "data/source_texts"
    output_file: str = "data/combined_star_wars.txt"
    zip_file: str = "data.zip"

    if not os.path.exists(source_dir):
        if os.path.exists(zip_file):
            print(f">>> {source_dir} not found. Unzipping {zip_file}...")
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(".")
            print(">>> Unzip complete.")
        else:
            print(f"Error: Neither {source_dir} nor {zip_file} found.")
            return

    files: list[str] = glob.glob(os.path.join(source_dir, "*.txt"))

    def get_index(f: str) -> int:
        match: Optional[re.Match] = re.search(r"(\d+)_", os.path.basename(f))
        return int(match.group(1)) if match else 999

    sorted_files: list[str] = sorted(files, key=get_index)

    print(f"Consolidating {len(sorted_files)} source files into {output_file}...")
    combined_content: list[str] = []
    for f_path in sorted_files:
        with open(f_path, "r", encoding="utf-8") as f:
            content: str = f.read().strip()
            ascii_content: str = content.encode("ascii", "ignore").decode("ascii")
            combined_content.append(ascii_content)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(combined_content))

    print("Consolidation complete!")


def main() -> None:
    """Consolidates source texts and builds a unified vocabulary."""
    consolidate_source_texts()

    source_file: str = "data/combined_star_wars.txt"
    print(f"Generating unified vocabulary from {source_file}...")

    with open(source_file, "r", encoding="utf-8") as f:
        text: str = f.read()

    tokenizer = Tokenizer.from_corpus(text)
    os.makedirs("models", exist_ok=True)
    tokenizer.save("models/unified_vocab.pt")
    print(f"Unified vocabulary saved! Size: {tokenizer.vocab_size} words.")


if __name__ == "__main__":
    main()
