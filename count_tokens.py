import glob
import os

from utils import clean_tokenization


def count_tokens_in_file(file_path: str):
    """Calculates the number of cleaned tokens in a given text file.

    This function reads the specified file, applies the project-standard
    Spacy-based tokenization (filtering for alpha characters, removing
    stop words and punctuation), and returns the total count.

    Args:
    - file_path: The absolute or relative path to the .txt file.

    Returns:
    - The integer count of valid tokens.

    Raises:
    - FileNotFoundError: If the file_path is invalid.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = clean_tokenization(text)
    return len(tokens)


def main():
    """Finds all data files and reports their token counts.

    Identifies any .txt files in the data/ directory and prints a
    formatted summary of their token counts for use in training
    estimation and dataset analysis.
    """

    data_files = sorted(glob.glob("data/*.txt"))

    if not data_files:
        print("No data files found in data/ directory.")
        return

    print("\n{: <20} | {: <12}".format("File Name", "Token Count"))
    print("-" * 35)

    total_tokens = 0
    for file_path in data_files:
        try:
            name = os.path.basename(file_path)
            count = count_tokens_in_file(file_path)
            print("{: <20} | {: <12,}".format(name, count))
            total_tokens += count
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if len(data_files) > 1:
        print("-" * 35)
        print("{: <20} | {: <12,}\n".format("TOTAL", total_tokens))
    else:
        print()


if __name__ == "__main__":
    main()
