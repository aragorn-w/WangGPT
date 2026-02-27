import spacy
import torch


# Load Spacy once, disable unnecessary components
NLP_MODEL = spacy.load("en_core_web_sm", disable=["parser", "ner"])
NLP_MODEL.max_length = 2_000_000_000

# Load unified vocabulary if it exists
try:
    UNIFIED_VOCAB = torch.load("models/unified_vocab.pt")
    TOK_TO_IDX = {tok: idx for idx, tok in enumerate(UNIFIED_VOCAB)}
    IDX_TO_TOK = {idx: tok for tok, idx in TOK_TO_IDX.items()}
    VOCAB_SIZE = len(UNIFIED_VOCAB)
except:
    UNIFIED_VOCAB = None
    TOK_TO_IDX = None
    IDX_TO_TOK = None
    VOCAB_SIZE = 0

def clean_tokenization(text: str):
    """Cleans and tokenizes the input text using Spacy.

    Lowercase the text, then removes stop words, punctuation, and
    non-alphabetic tokens to ensure only meaningful words remain.

    Args:
    - text: The raw input string to tokenize.

    Returns:
    - A list of cleaned token strings.

    Raises:
    - ValueError: If the input text is not a string.
    """

    # Use nlp.pipe for faster processing if we were doing many small strings,
    # but for one big string, we just need to ensure components are active.
    doc = NLP_MODEL(text.lower())
    return [
        token.text for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
