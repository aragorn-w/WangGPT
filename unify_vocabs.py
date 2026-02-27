import glob
import os
import re

import torch

from utils import clean_tokenization


def consolidate_source_texts():
    """Combines all source text files into one based on their prefix index.

    Finds all .txt files in data/source_texts/, sorts them numerically
    by their index prefix, and concatenates them with two newlines
    as a separator.
    """

    source_dir = "data/source_texts"
    output_file = "data/combined_star_wars.txt"

    if not os.path.exists(source_dir):
        print(f"Error: Source directory {source_dir} not found.")
        return

    # Get all .txt files and sort by integer prefix
    files = glob.glob(os.path.join(source_dir, "*.txt"))

    def get_index(f):
        match = re.search(r"(\d+)_", os.path.basename(f))
        return int(match.group(1)) if match else 999

    sorted_files = sorted(files, key=get_index)

    print(f"Consolidating {len(sorted_files)} source files into {output_file}...")
    combined_content = []
    for f_path in sorted_files:
        with open(f_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            # Remove all non-ASCII characters
            ascii_content = content.encode("ascii", "ignore").decode("ascii")
            combined_content.append(ascii_content)

    # Join with two newlines
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(combined_content))

    print(f"Consolidation complete!")


def main():
    """Orchestrates the data consolidation and vocabulary generation.

    First combines the source text files, then generates a unified vocabulary
    from the combined output to ensure cross-model consistency.
    """

    # Consolidate files
    consolidate_source_texts()

    # Build vocabulary
    source_file = "data/combined_star_wars.txt"
    print(f"Generating unified vocabulary from {source_file}...")

    with open(source_file, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = clean_tokenization(text)
    vocab = sorted(list(set(tokens)))

    torch.save(vocab, "models/unified_vocab.pt")
    print(f"Unified vocabulary saved! Size: {len(vocab)} words.")


if __name__ == "__main__":
    main()
