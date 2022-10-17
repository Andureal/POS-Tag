import nltk
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import pprint, time

import sklearn
import sklearn_crfsuite
import scipy.stats
import math, string, re
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from itertools import chain
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn_crfsuite import CRF
from nltk.tokenize import word_tokenize
from nltk.tag.util import untag

import ast
from ast import literal_eval

import jieba 
from hanziconv import HanziConv

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import Tokenizer
from operator import add


def convert_string2_list(text):
    return ast.literal_eval(str(text))

def features(sentence, index):
    # """ sentence: [w1, w2, ...], index: the index of the word """
    
    feature_set =  {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'prefix-4': sentence[index][:4],
        'prefix-5': sentence[index][:5],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'suffix-4': sentence[index][-4:],
        'suffix-5': sentence[index][-5:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'prev2_word': '' if index == 0 else sentence[index - 2],
        'next2_word': '' if index == len(sentence) - 2 or index == len(sentence) - 1  else sentence[index + 2],
        'prev3_word': '' if index == 0 else sentence[index - 3],
        'next3_word': '' if index == len(sentence) - 2 or index == len(sentence) - 1  or index == len(sentence) - 3  else sentence[index + 3],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:],
        'natural_number': (re.findall(r'^[0-9]+', sentence[index])),
        'initcaps' : (re.findall(r'^[A-Z]\w+', sentence[index])),
        'initcapsalpha': (re.findall(r'^[A-Z][a-z]\w+', sentence[index])),
        'word.stemmed': re.sub(r'(.{2,}?)([aeiougyn]+$)',r'\1', sentence[index].lower()),
        'word.ispunctuation': (sentence[index] in string.punctuation)
    }
    
    if index <= 0:
        feature_set['BOS'] = True
    
    if index > len(sentence)-1:
        feature_set['EOS'] = True
        
    return feature_set


def transform_to_dataset(tagged_sentences):
    X, y = [], []
 
    for tagged in tagged_sentences:
        X.append([features(untag(tagged), index) for index in range(len(tagged))])
        y.append([tag for _, tag in tagged])
 
    return X, y

def pos_tag1(sentence, model):
    sentence = sentence_splitter(sentence)
    sentence_features = [features(sentence, index) for index in range(len(sentence))]
    return list(zip(sentence, model.predict([sentence_features])[0]))

def sentence_splitter(sentence):
    result = []
    sents = word_tokenize(sentence)
    for s in sents:
        if re.findall(r'[\u4e00-\u9fff]+', s):
            s = HanziConv.toSimplified(s)
            result = result + list(jieba.cut(s, cut_all=False))
        else:
            result.append(s)
    return result
