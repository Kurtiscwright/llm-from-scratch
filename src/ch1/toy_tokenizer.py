import os
import requests
import re

# Setup file path for data folder.
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '../..')
file_path = os.path.join(data_dir, 'data/', 'the-verdict.txt')

# Will be refenced during initial setup
verdict_file_path = os.path.abspath(file_path)

# Pull down sample text if it's not already present
if not os.path.exists(verdict_file_path):
    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(verdict_file_path, "wb") as f:
        f.write(response.content)

# Initial look at what text is apart of this example
with open(verdict_file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of characters:", len(raw_text))
print(raw_text[:99])

# Start toy tokenizier via simple regex
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])
print(len(preprocessed))

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

vocab = {token:integer for integer,token in enumerate(all_words)}

# Let's see the first 50 words captured by the toy tokenizer
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

# Create a more formal class to represent the tokenizer class
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

tokenizer.decode(ids)
tokenizer.decode(tokenizer.encode(text))

# Now let's extend with special characters to allow multiple documents of test data and unknown words.
all_tokens = sorted(list(preprocessed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

vocab = {token:integer for integer,token in enumerate(all_tokens)}
# Should now be 1132
len(vocab.items())

# Let's validate that the end of text and unknown characters are at the end
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

# Now an updated Tokenizer class to properly use the unknown character
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items() }

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

# Let's test the updated tokenizerV2


tokenizer = SimpleTokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1, text2))

print(text)

tokenizer.encode(text)
tokenizer.decode(tokenizer.encode(text))
