import nltk
import emoji
from gensim.parsing.preprocessing import remove_stopwords
import cleantext

from cleantext import clean
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize

import re

def all(text):
    text = text.lower()

    text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)

    text_tokens = word_tokenize(text)
    text = [word for word in text_tokens if not word in stopwords.words()]

    string_encode = text.encode("ascii", "ignore")
    text = string_encode.decode()

    text = re.sub('\r', ' ', text)
    text = re.sub('&quot;|"|“|”', '', text)

    text = clean(text, no_emoji=True)

    return text


def normalise(text):
    #actual = actual.apply(lambda x:x.str.lower())
    text = text.lower()
    return text

def remove_punct_url_at(text):
    text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
    return text
    
def remove_stopword_nltk(text):
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    return tokens_without_sw

def remove_non_ascii(text):
    string_encode = text.encode("ascii", "ignore")
    string_decode = string_encode.decode()
    return string_decode

## Remove special character (\r)
def remove_slashR(text):
    return re.sub('\r', ' ', text)

def remove_special_char(text):
    return re.sub('&quot;|"|“|”', '', text)

# def remove_emoji(text):
#     return emoji.get_emoji_regexp().sub('', text)

def remove_emoji(text):
    text = clean(text, no_emoji=True)
    return text

def remove_multiple_space(text):
    return re.sub('\s+', ' ', text).strip()

# Replace newline
def remove_newline(text):
    return re.sub(r'\n', ' ', text)

# Replace apostrophe's special characters to original apostrophe
def replace_apostrophes(text):
    return re.sub('&#39;|’|´|‘', "'", text)

def remove_multiple_comma(text):
    return re.sub(r'[,]{2,}',',', text)

def remove_multiple_dot(text):
    return re.sub(r'[.]{3,}','', text)

def tokenise(text):
    text.split()
    return text 

    # def remove_stopword_gensim(text):
#     filtered_sentence = remove_stopwords(text)
#     return filtered_sentence

# def unicode_problem(text):
#     return re.sub(r'[\u0080]','',text).strip()

# def preprocess_tweet_fn(text):
#     p.set_options(p.OPT.URL, p.OPT.HASHTAG, p.OPT.MENTION)
#     return p.tokenize(text) 

## Convert traditional Chinese characters to Simplified Chinese characters
# def convert_Tra_Simp_Chi(text):
#     return HanziConv.toSimplified(text)
