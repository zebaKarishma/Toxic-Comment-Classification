import numpy as np
import pandas as pd
from keras.layers import Dense, Input, LSTM, Bidirectional, Conv1D
from keras.layers import Dropout, Embedding
from keras.preprocessing import text, sequence
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D
from keras.models import Model
import re
from langdetect import detect
from google.cloud import translate_v2 as translate
from tqdm import tqdm

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text



EMBEDDING_FILE = 'glove.840B.300d.txt'
train_x = pd.read_csv('train.csv').fillna(' ')
test_x = pd.read_csv('test.csv').fillna(' ')


train_x['comment_text'] = train_x['comment_text'].map(lambda com : clean_text(com))
test_x['comment_text'] = test_x['comment_text'].map(lambda com : clean_text(com))

translate_client = translate.Client()

for i in tqdm(range(0, 153163)):
    try:
        if detect(test_x['comment_text'][i]) != 'en':
            temp = translate_client.translate(test_x['comment_text'][i],target_language='en')
            test_x['comment_text'][i] = temp['translatedText']
    except:
        temp = 0

for i in tqdm(range(0, 159570)):
    try:
        if detect(train_x['comment_text'][i]) != 'en':
            temp = translate_client.translate(train_x['comment_text'][i],target_language='en')
            train_x['comment_text'][i] = temp['translatedText']
    except:
        temp = 0


test_x.to_csv('test_translated.csv', index=False)
train_x.to_csv('train_translated.csv', index=False)
