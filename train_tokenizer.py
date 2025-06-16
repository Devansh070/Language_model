# train_tokenizer.py
from tokenizers import ByteLevelBPETokenizer
from pathlib import Path
import requests

texts = []

# Download multiple books from Project Gutenberg for a larger dataset
urls = [
    "https://www.gutenberg.org/files/11/11-0.txt",     # Alice in Wonderland
    "https://www.gutenberg.org/files/1342/1342-0.txt", # Pride and Prejudice
    "https://www.gutenberg.org/files/84/84-0.txt",     # Frankenstein
    "https://www.gutenberg.org/files/1661/1661-0.txt", # Sherlock Holmes
    "https://www.gutenberg.org/files/2701/2701-0.txt", # Moby Dick
    "https://www.gutenberg.org/files/98/98-0.txt",     # A Tale of Two Cities
    "https://www.gutenberg.org/files/5200/5200-0.txt", # Metamorphosis
    "https://www.gutenberg.org/files/2600/2600-0.txt", # War and Peace
    "https://www.gutenberg.org/files/74/74-0.txt",     # The Adventures of Tom Sawyer
    "https://www.gutenberg.org/files/1400/1400-0.txt", # Great Expectations
]

for url in urls:
    print(f"Downloading {url} ...")
    response = requests.get(url)
    if response.status_code == 200:
        book_texts = [line.strip() for line in response.text.split('\n') if line.strip()]
        texts.extend(book_texts)
        print(f"Added {len(book_texts)} lines.")
    else:
        print(f"Failed to download {url}")

print(f"Total lines collected: {len(texts)}")

# Count total number of training tokens (words/wordpieces)
total_tokens = sum(len(line.split()) for line in texts)
print(f"Total number of training tokens (approximate, whitespace split): {total_tokens}")

# Save all texts to a temporary file for training
corpus_path = "corpus.txt"
with open(corpus_path, "w", encoding="utf-8") as f:
    for line in texts:
        f.write(line + "\n")

# Train a Byte Pair Encoding (BPE) tokenizer
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=corpus_path,
    vocab_size=10000,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)

# Save the tokenizer
save_dir = "my-10k-bpe-tokenizer"
Path(save_dir).mkdir(exist_ok=True)
tokenizer.save_model(save_dir)

# Save as HuggingFace tokenizer JSON for compatibility
tokenizer_json_path = str(Path(save_dir) / "tokenizer.json")
tokenizer.save(tokenizer_json_path)
print(f"Saved HuggingFace-compatible tokenizer.json to {tokenizer_json_path}")

print(f"BPE tokenizer trained and saved to {save_dir}/")
print(f"Number of tokens in vocab: {tokenizer.get_vocab_size()}")