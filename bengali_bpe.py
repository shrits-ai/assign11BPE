import re
from collections import Counter, defaultdict
import pandas as pd
import json
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_and_clean_dataset(file_path):
    print("Loading and cleaning dataset...")
    data = pd.read_csv(file_path)

    # Combine relevant columns into one and deduplicate rows
    data['FullText'] = data[['Topics', 'Question-Title', 'Questions', 'Answers']].fillna('').agg(' '.join, axis=1)
    data.drop_duplicates(subset=['FullText'], inplace=True)

    # Clean text using regex
    data['CleanedText'] = data['FullText'].apply(lambda text: re.sub(r'[^ঀ-৿0-9।,!?\s]', '', text))
    data['CleanedText'] = data['CleanedText'].apply(lambda text: re.sub(r'\s+', ' ', text).strip())

    # Deduplicate near-duplicates based on cosine similarity
    def remove_near_duplicates(texts, threshold=0.9):
        print("Removing near-duplicates...")
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)

        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        unique_texts = []
        to_remove = set()

        for i, text in enumerate(texts):
            if i in to_remove:
                continue

            unique_texts.append(text)
            for j in range(i + 1, len(texts)):
                if cosine_sim[i, j] > threshold:
                    to_remove.add(j)

        return unique_texts

    cleaned_texts = data['CleanedText'].tolist()
    cleaned_texts = remove_near_duplicates(cleaned_texts)

    # Filter dataset
    data_cleaned = data[data['CleanedText'].isin(cleaned_texts)]
    return data_cleaned['CleanedText'].tolist()


# BPE Tokenizer Implementation
class BPETokenizer:
    def __init__(self, vocab_size, penalty_factor=5):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.penalty_dict = {}
        self.penalty_factor = penalty_factor

    def get_stats(self, tokenized_corpus):
        pairs = defaultdict(int)
        for word, freq in tokenized_corpus.items():
            tokens = word.split()
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pairs[pair] += freq

        # Apply penalties to over-merged pairs
        for pair in self.penalty_dict:
            if pair in pairs:
                pairs[pair] -= self.penalty_factor
                if pairs[pair] <= 0:
                    del pairs[pair]

        return pairs

    def merge_vocab(self, pair, corpus):
        """Merge the most frequent pair in the corpus."""
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        new_corpus = {}
        changes_made = False  # Track if the merge makes changes

        for word, freq in corpus.items():
            if bigram in word:
                changes_made = True
                new_word = word.replace(bigram, replacement)
                new_corpus[new_word] = freq
            else:
                new_corpus[word] = freq

        # If no changes were made, log and skip this merge
        if not changes_made:
            print(f"Merge of {pair} made no changes to the corpus. Skipping...")

        return new_corpus, changes_made


    def train(self, corpus):
        print("Starting BPE training...")

        tokenized_corpus = {' '.join(word): freq for word, freq in Counter(corpus).items()}
        seen_tokens = set()  # Track tokens already in the vocabulary
        merged_pairs = set()  # Track already merged pairs
        ineffective_pairs = set()  # Track pairs that made no changes

        with tqdm(total=self.vocab_size, desc="Training BPE", unit="token") as pbar:
            while len(self.vocab) < self.vocab_size:
                pairs = self.get_stats(tokenized_corpus)

                # Check if no pairs are available to merge
                if not pairs:
                    print("No more pairs to merge. Stopping early.")
                    break

                # Skip pairs that are ineffective or already merged
                pairs = {pair: freq for pair, freq in pairs.items() if pair not in ineffective_pairs and pair not in merged_pairs}

                # If no valid pairs remain, exit the loop
                if not pairs:
                    print("No valid pairs to merge. Stopping early.")
                    break

                # Find the most frequent pair
                best_pair = max(pairs, key=pairs.get)

                merged_token = ''.join(best_pair)

                # Skip if the token is already in the vocabulary
                if merged_token in self.vocab:
                    if merged_token not in seen_tokens:
                        print(f"Token '{merged_token}' already in vocabulary. Skipping...")
                        seen_tokens.add(merged_token)
                    continue

                # Merge the most frequent pair and update the corpus
                tokenized_corpus, changes_made = self.merge_vocab(best_pair, tokenized_corpus)

                if not changes_made:
                    print(f"Merge of {best_pair} made no changes to the corpus. Skipping...")
                    ineffective_pairs.add(best_pair)  # Add to blacklist
                    continue

                # Add the merged token to the vocabulary
                self.vocab[merged_token] = len(self.vocab)
                merged_pairs.add(best_pair)  # Add to merged pairs
                pbar.update(1)

                # Monitor compression ratio dynamically
                if len(self.vocab) % 100 == 0:
                    compression_ratio = calculate_compression_ratio(corpus, self)
                    if compression_ratio >= 3.5 and len(self.vocab) >= self.vocab_size:
                        print(f"Final Compression Ratio: {compression_ratio:.2f}")
                        break

        print("BPE training completed.")
        return self.vocab




    def encode(self, text):
        for token in sorted(self.vocab, key=len, reverse=True):
            text = text.replace(token, f' {token} ')
        return text.split()


def calculate_compression_ratio(corpus, tokenizer):
    original_length = sum(len(text) for text in corpus)
    encoded_length = sum(len(tokenizer.encode(text)) for text in corpus)
    return original_length / encoded_length


def save_vocabulary(vocab, file_path):
    print(f"Saving vocabulary to {file_path}...")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
    print("Vocabulary saved.")

def save_cleaned_dataset(cleaned_texts, save_path):
    """Save the cleaned dataset to a file."""
    print(f"Saving cleaned dataset to {save_path}...")
    with open(save_path, 'w', encoding='utf-8') as f:
        for text in cleaned_texts:
            f.write(text + '\n')
    print("Cleaned dataset saved successfully.")

def load_cleaned_dataset(file_path):
    """Load the cleaned dataset from a file."""
    print(f"Loading cleaned dataset from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        cleaned_texts = [line.strip() for line in f]
    print(f"Loaded {len(cleaned_texts)} lines from the cleaned dataset.")
    return cleaned_texts

if __name__ == "__main__":
    # Load and preprocess dataset
    file_path = 'BengaliEmpatheticConversationsCorpus.csv'
    cleaned_file_path = 'cleaned_dataset.txt'

    # Check if the cleaned dataset already exists
    try:
        # Load cleaned dataset if it exists
        corpus = load_cleaned_dataset(cleaned_file_path)
    except FileNotFoundError:
        # Clean the original dataset and save it
        corpus = load_and_clean_dataset(file_path)
        save_cleaned_dataset(corpus, cleaned_file_path)

    # Train BPE Tokenizer
    bpe_tokenizer = BPETokenizer(vocab_size=5000, penalty_factor=5)
    vocab = bpe_tokenizer.train(corpus)

    # Calculate compression ratio
    compression_ratio = calculate_compression_ratio(corpus, bpe_tokenizer)
    print(f"Final Compression Ratio: {compression_ratio:.2f}")

    # Save the vocabulary
    vocab_path = 'bpe_vocab_5000.json'
    save_vocabulary(vocab, vocab_path)

    print("Vocabulary Size:", len(vocab))
