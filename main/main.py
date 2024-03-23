# python == 3.11.7

import re
import nltk
import torch
from pathlib import Path
# Define the device if using GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import pipeline
# from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')

tokenizer = AutoTokenizer.from_pretrained(Path("/home/bikas/Text-Summarization/main/summary/tokenizer"))
model = "/home/bikas/Text-Summarization/main/summary/pegasus-samsum-model"
def remove_spaces_before_punctuation(text):
    pattern = re.compile(r'(\s+)([.,;!?])')
    result = pattern.sub(r'\2', text)
    result = re.sub(r'\[|\]', '', result)
    return result


def replace_pronouns(text):
    # Replace "they" with "he" or "she" based on context
    text = re.sub(r'\bthey\b', 'He/She', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(are|have|were)\b', lambda x: {'are': 'is', 'have': 'has', 'were': 'was'}[x.group()], text)
    return text


def clean_and_lemmatize(text):
    # Remove digits, symbols, punctuation marks, and newline characters
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s,-]', '', text.replace('\n', ''))
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    # Lemmatize each token and join back into a sentence
    lemmatized_text = ' '.join([lemmatizer.lemmatize(token) for token in tokens])
    return lemmatized_text


# input_text = input("\nGive me Your Full Month Log hour Task Description within 5500 Characters: \n")
input_text = """

"""

# Tokenize the input text
tokens_org_text = tokenizer.tokenize(input_text)
# print(tokens_org_text)
# Get the length of the tokenized sequence
sequence_length_org_text = len(tokens_org_text)

# print(sequence_length_org_text)

input_text = clean_and_lemmatize(input_text)
# Tokenize the input text
tokens = tokenizer.tokenize(input_text)

# Get the length of the tokenized sequence
sequence_length = len(tokens)
# print(sequence_length)


if sequence_length >= 1024:
    print(f"Your Text token length is {sequence_length_org_text} Which exceeds 1023 tokens")
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(model).to(device)
   
    gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}

    pipe = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)
    # Example usage
    text = pipe(input_text, **gen_kwargs)[0]["summary_text"]
    output_text = replace_pronouns(remove_spaces_before_punctuation(text))
    print("Summary: ", output_text)
    # Clear the GPU cache
    torch.cuda.empty_cache()
