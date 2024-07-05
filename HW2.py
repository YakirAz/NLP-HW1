import pandas as pd
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
import time
import requests
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re  # Added import for regex
from gensim.models import Word2Vec  # Added import for Word2Vec

# Download necessary NLTK datasets
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Load spaCy model
nlp = spacy.load('en_core_web_md')  # Changed to 'en_core_web_md' for GloVe embeddings

# Q4 - Load the dataset
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
df.columns = ['label', 'message']

# Q5 - Print basic statistics
total_messages = df.shape[0]
spam_count = df[df['label'] == 'spam'].shape[0]
ham_count = df[df['label'] == 'ham'].shape[0]
df['word_count'] = df['message'].apply(lambda x: len(x.split()))
average_words_per_message = df['word_count'].mean()

print(f'Total number of SMS messages: {total_messages}')
print(f'Number of spam messages: {spam_count}')
print(f'Number of ham messages: {ham_count}')
print(f'Average number of words per message: {average_words_per_message:.2f}')

# Text processing
stop_words = set(stopwords.words('english'))
additional_stopwords = {'and', 'or', 'but', 'if', 'while', 'a', 'an', 'the'}

def preprocess_message(message):
    message = message.lower()
    words = word_tokenize(message)
    words = [word for word in words if word not in stop_words and word not in additional_stopwords and word.isalnum()]
    return words

df['words'] = df['message'].apply(preprocess_message)
all_words = [word for words in df['words'] for word in words]

# Most frequent words
most_common_words = Counter(all_words).most_common(5)
print('5 most frequent words:')
for word, count in most_common_words:
    print(f'{word}: {count}')

# Words that appear only once
word_counts = Counter(all_words)
words_once = [word for word, count in word_counts.items() if count == 1]
print(f'Number of words that only appear once: {len(words_once)}')

# Q6 - Tokenize using NLTK
start_time = time.time()
df['nltk_tokens'] = df['message'].apply(word_tokenize)
nltk_time = time.time() - start_time

print(f"NLTK tokenization time: {nltk_time:.4f} seconds")

# Tokenize using spaCy
start_time = time.time()
df['spacy_tokens'] = df['message'].apply(lambda x: [token.text for token in nlp(x)])
spacy_time = time.time() - start_time

print(f"spaCy tokenization time: {spacy_time:.4f} seconds")

# Q7 - Lemmatize using spaCy
def spacy_lemmatize(doc):
    return [token.lemma_ for token in nlp(doc)]

start_time = time.time()
df['spacy_lemmas'] = df['message'].apply(spacy_lemmatize)
spacy_lemma_time = time.time() - start_time

print(f"spaCy lemmatization time: {spacy_lemma_time:.4f} seconds")

# Q8 - Stem using NLTK
stemmer = PorterStemmer()

def nltk_stem(words):
    return [stemmer.stem(word) for word in words]

start_time = time.time()
df['nltk_stems'] = df['words'].apply(nltk_stem)
nltk_stem_time = time.time() - start_time

print(f"NLTK stemming time: {nltk_stem_time:.4f} seconds")

# Display a few examples of lemmatized messages using spaCy
print("\nSample lemmatized messages using spaCy:")
print(df['spacy_lemmas'].head())

# Display a few examples of stemmed messages using NLTK
print("\nSample stemmed messages using NLTK:")
print(df['nltk_stems'].head())

# Q9 - Lemmatize using NLTK
lemmatizer = WordNetLemmatizer()

def nltk_lemmatize(words):
    return [lemmatizer.lemmatize(word) for word in words]

start_time = time.time()
df['nltk_lemmas'] = df['words'].apply(nltk_lemmatize)
nltk_lemma_time = time.time() - start_time

print(f"NLTK lemmatization time: {nltk_lemma_time:.4f} seconds")

# Display a few examples of lemmatized messages using NLTK
print("\nSample lemmatized messages using NLTK:")
print(df['nltk_lemmas'].head())

# Q10 - Stem using spaCy
def spacy_stem(doc):
    return [token.lemma_ for token in nlp(doc)]

start_time = time.time()
df['spacy_stems'] = df['message'].apply(spacy_stem)
spacy_stem_time = time.time() - start_time

print(f"spaCy stemming time: {spacy_stem_time:.4f} seconds")

# Display a few examples of stemmed messages using spaCy
print("\nSample stemmed messages using spaCy:")
print(df['spacy_stems'].head())

# Q11 - Scrape text data from a public internet page using BeautifulSoup
web_url = 'https://en.wikipedia.org/wiki/Natural_language_processing'
response = requests.get(web_url)
if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    web_text = soup.get_text(separator=' ', strip=True)  # Added parameters for better readability
else:
    print(f"Failed to fetch data from {web_url}")
    web_text = ""

# Print scraped text data
print("\nText data from the public internet page:")
print(web_text)

# Q12 - Perform tokenization, lemmatization, and stemming on the scraped text
web_tokens = word_tokenize(web_text)
print("\nTokens from scraped text:")
print(web_tokens)

# Lemmatization using spaCy
web_lemas = spacy_lemmatize(web_text)
print("\nLemmas from scraped text using spaCy:")
print(web_lemas)

# Stemming using NLTK
web_stems = nltk_stem(web_tokens)
print("\nStems from scraped text using NLTK:")
print(web_stems)

# Q13 - Print word statistics on the scraped data before and after text processing
print("\nWord statistics on the scraped data before text processing:")
print(f"Total number of words: {len(web_tokens)}")
print(f"Number of unique words: {len(set(web_tokens))}")

print("\nWord statistics on the scraped data after text processing:")
print(f"Total number of words after processing: {len(web_lemas)}")
print(f"Number of unique words after processing: {len(set(web_lemas))}")

# Q14 - Read WhatsApp data from file
with open('whatsapp.txt', 'r', encoding='utf-8') as file:
    whatsapp_data = file.read()

# Q15 - Preprocess WhatsApp data
whatsapp_words = preprocess_message(whatsapp_data)

# Q16 - Calculate word statistics for WhatsApp data after preprocessing
whatsapp_word_counts = Counter(whatsapp_words)
most_common_whatsapp_words = whatsapp_word_counts.most_common(5)

print("\nWord statistics for WhatsApp data after preprocessing:")
print(f"Total number of words: {len(whatsapp_words)}")
print(f"Number of unique words: {len(set(whatsapp_words))}")
print("5 most common words:", most_common_whatsapp_words)

# Homework 2:

# White Space Tokenizer
df['whitespace_tokens'] = df['message'].apply(lambda x: x.split())

# Regex Tokenizer
def regex_tokenizer(text):
    return re.findall(r'\w+', text)

df['regex_tokens'] = df['message'].apply(regex_tokenizer)

# Word Tokenizer
df['word_tokens'] = df['message'].apply(word_tokenize)

# Sentence Tokenizer
df['sentence_tokens'] = df['message'].apply(sent_tokenize)

# Feature Extraction - BOW
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(df['message'])
print("\nBag of Words (BOW) feature extraction:")
print(X_bow.toarray())

# Feature Extraction - TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(df['message'])
print("\nTF-IDF feature extraction:")
print(X_tfidf.toarray())

# Feature Extraction - Word2Vec
sentences = df['words'].tolist()
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

def get_word_vector(words):
    vectors = []
    for word in words:
        if word in word2vec_model.wv:
            vectors.append(word2vec_model.wv[word])
    if vectors:
        return sum(vectors) / len(vectors)
    else:
        return None

df['word2vec'] = df['words'].apply(get_word_vector)

# Using spaCy GloVe embeddings
df['glove'] = df['message'].apply(lambda x: nlp(x).vector)

# CYK Parsing
from nltk import CFG
from nltk.parse import ChartParser

grammar = CFG.fromstring("""
    S -> NP VP
    NP -> DT NN
    VP -> VBZ NP | VBZ NN
    DT -> 'the' | 'a'
    NN -> 'cat' | 'dog' | 'man' | 'park'
    VBZ -> 'sees' | 'likes'
""")

parser = ChartParser(grammar)

sentences = [
    "the cat sees the dog",
    "a man likes the park",
    "the dog sees a cat",
    "the man likes the dog",
    "a cat sees a man"
]

for sentence in sentences:
    tokens = sentence.split()
    parses = list(parser.parse(tokens))
    for tree in parses:
        print(tree)
