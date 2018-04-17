import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pandas.compat import cStringIO

# LOAD DATA
df = pd.read_csv("train.csv", delimiter=',')

' Remove comments from below code to get questions from user '
"""que1=input("1:")
que2=input("2:")

que3=[[que1,que2]]
df = pd.DataFrame(que3,columns=['question1','question2']) """

# encode questions to unicode
df['question1'] = df['question1'].apply(lambda x: str(str(x).encode("utf-8")))
df['question2'] = df['question2'].apply(lambda x: str(str(x).encode("utf-8")))

# TRAIN GLOVE

import gensim

if os.path.exists('data/word2vectors.mdl'):
    model = gensim.models.KeyedVectors.load_word2vec_format('data/word2vectors.bin', binary=True)
    # trim memory
    model.init_sims(replace=True)
    # creta a dict
    w2v = dict(zip(model.index2word, model.syn0))
    print("Number of tokens in Word2Vec:", len(w2v.keys()))
else:
    questions = list(df['question1']) + list(df['question2'])

    # tokenize
    c = 0
    for question in tqdm(questions):
        questions[c] = list(gensim.utils.tokenize(question, deacc=True, lower=True))
        c += 1

    # train model
    model = gensim.models.Word2Vec(questions, size=300, workers=16, iter=10, negative=20)

    # trim memory
    model.init_sims(replace=True)

    # creta a dict
    w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    print("Number of tokens in Word2Vec:", len(w2v.keys()))

    # save model
    model.wv.save('data/word2vectors.mdl')
    model.wv.save_word2vec_format('data/word2vectors.bin', binary=True)
    del questions

# EXTRACT FEATURES
from utils import MeanEmbeddingVectorizer, TfidfEmbeddingVectorizer

if os.path.exists('data/Features.pkl'):
    df = pd.read_pickle('data/Features.pkl')
else:
    # gather all questions
    questions = list(df['question1']) + list(df['question2'])
    # print(questions)
    # tokenize questions
    c = 0
    for question in tqdm(questions):
        # print(c)
        questions[c] = list(gensim.utils.tokenize(question, deacc=True))
        c += 1

    me = TfidfEmbeddingVectorizer(w2v)
    me.fit(questions)
    # exctract word2vec vectors
    vecs1 = me.transform(df['question1'])
    df['q1_feats'] = list(vecs1)

    vecs2 = me.transform(df['question2'])
    df['q2_feats'] = list(vecs2)

    # save features
    pd.to_pickle(df, 'data/Features.pkl')
