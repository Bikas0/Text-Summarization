import re
import nltk
from transformers import pipeline
from transformers import AutoTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')


def remove_spaces_before_punctuation(text):
    pattern = re.compile(r'(\s+)([.,;!?])')
    result = pattern.sub(r'\2', text)
    result = re.sub(r'\[|\]', '', result)
    return result


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
input_text = """Set up the Python development environment by installing Anaconda Python distribution and Visual Studio Code. Ensure seamless integration between Anaconda and VS Code for efficient coding, debugging, and project management.
Studied NLP Basic, Topics - Tokenization using Spacy, Language Processing Pipeline, Stemming, Lemmatization and POS Tagging Get the overview and understand the implementation of Twitter Sentiment Analysis using NLTK. Clearing concepts related to Bangla NLP and trying to bridging my knowledge gap.
Explored Text to Sequence and Word Embedding techniques, implemented LSTM-CNN for Bangla Sentiment Analysis, achieving an accuracy of around 84% (with room for improvement). Commenced the review of Transformer Models for potential advancements in the task.
Enhanced sentiment analysis by augmenting Bangla text using Token Replacement. Assisted Nazmul in setting up CUDA Toolkit on Server PC for optimized GPU
Achieved a groundbreaking ~94% validation accuracy in sentiment analysis using a CNN-LSTM model on a dataset expanded from 6.6k to 16k texts using Text Augmentation. This accomplishment underscores the effectiveness of our approach, marking a significant leap in sentiment analysis accuracy.
Worked on FlutterML Dataset rearrangement process. Discovered another sentiment analysis dataset of 100k+ text Trained, and accuracy ~ 90% (Model is biased, as the dataset is imbalanced) Tried Model training with weight balance, but still performance is not satisfactory. Researched on AI and Health
Get an overview of Text summarization, studied attention model, transformers (T5, BART, PEGASUS) 2. Text summarization tested with crosSum (Buet CSENLP) pretrained model implementation. 3. Take a look on Google TAPAS Table QNA work process, analyzed with a few questions inputs
Enhanced text summarization using fine-tuned Google Pegasus on the Samsum dataset, elevating the ROUGE score from 0.266 to 0.439. Set up server PC environment for the full-scale dataset training. Studied basic workflow of the Langchain, Streamlit framework.
Prepared last day's code. Tried to fine-tune pre-trained Google Pegasus model on Samsum dataset, reduced loss from 3.43 to 1.09. But after 6 epochs train failed due to low CUDA memory. Created dataset summary sample, and evaluated by Kabir vai (Flutter Team). Tutorial watching - Docker, Streamlit
"""
Org_len = len(input_text)
input_text = clean_and_lemmatize(input_text)

if len(input_text) > 5500:
    print(f"Your Text characters number is {Org_len} Which exceeds 5500 Characters")
else:
    tokenizer = AutoTokenizer.from_pretrained("summary/tokenizer")
    # model = "pegasus-samsum-model"
    gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}

    pipe = pipeline("summarization", model="summary/pegasus-samsum-model", tokenizer=tokenizer)
    # Example usage
    text = pipe(input_text, **gen_kwargs)[0]["summary_text"]
    output_text = remove_spaces_before_punctuation(text)
    print("\nSummary: ", output_text)
