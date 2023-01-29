import os
import sys
import re as re
import string
import yaml

import multiprocessing
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from gensim.parsing.porter import PorterStemmer

import random as rn
import pandas as pd
import csv

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score, accuracy_score, precision_recall_fscore_support
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
from tensorflow import keras
from keras import backend as K
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Embedding, LSTM, Bidirectional, Activation, LeakyReLU
from keras.models import model_from_yaml
from keras.utils import np_utils
from keras_self_attention import SeqSelfAttention



tf.disable_v2_behavior()
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 16} ) 
sess = tf.Session(config=config) 
K.set_session(sess)


csv.field_size_limit(100000000)
sys.setrecursionlimit(1000000)



n_epoch = 30

vocab_dim = 100
maxlen = 100
n_exposures = 10
window_size = 5
batch_size = 24
input_length = 100
cpu_count = multiprocessing.cpu_count()

test_list = []

path = 'input/message'
model_location = 'model'  +'/model_'+ sys.argv[1]
embedding_location = 'embedding' + '/Word2vec_model_' + sys.argv[1] + '.pkl'

result_list = []

def loadfile():

    data_full=pd.read_csv(path  + '_' + sys.argv[1] + '.csv', usecols=[0,1,2,3,4,5,6,7], engine='python')
    dataset = data_full.values
    classes = dataset[:, 5]
    data=data_full['Feature'].values.tolist()
    combined = data
    combined_full = data_full.values.tolist()
    y = list(classes)
    test_block_list = []
    train_block_list = []
    for x in x_test:
        test_list.append(x[0])
        test_block_list.append(x[4])
    x_test = np.array(test_block_list)
    for x in x_train:
        train_block_list.append(x[4])
    x_train = np.array(train_block_list)
    x_train = x_train.reshape(-1, 1)
    return combined,y, x_train, x_val, x_test, y_train, y_val,  y_test



def word2vec_train(combined):

    model = Word2Vec(size=vocab_dim, 
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count, sg=1,
                     )
    model.build_vocab(combined)
    model.save(embedding_location)
    index_dict, word_vectors,combined = create_dictionaries(model=model,combined=combined)
    return   index_dict, word_vectors,combined


def tokenizer(text):
    newText = []
    for doc in text:
        docText = []
        for word in str(doc).replace("'", "").replace("[", "").replace("]", "").replace(",", "").replace('"', "").split(' '):
            docText.append(word)         
        newText.append(docText)
    return newText
    


def input_transform(words):
    model=Word2Vec.load(embedding_location)
    _, _,dictionaries=create_dictionaries(model,words)
    return dictionaries



def create_dictionaries(model=None,
                        combined=None):
    from keras.preprocessing import sequence
    
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}
        w2vec = {word: model.wv[word] for word in w2indx.keys()}

        def parse_dataset(combined):
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=maxlen)
        return w2indx, w2vec,combined





def get_data(index_dict,word_vectors,combined):

    n_symbols = len(index_dict) + 1  
    embedding_weights = np.zeros((n_symbols, vocab_dim))
    for word, index in index_dict.items():
        embedding_weights[index, :] = word_vectors[word]


    return n_symbols,embedding_weights



def train_model_dl(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):
    
    oversample = ADASYN()
    #print("ADASYN oversampling enabled")
    x_train, y_train = oversample.fit_resample(x_train, y_train)

    


    model = Sequential()  
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))  
    model.add(Bidirectional(LSTM(128,activation='sigmoid')))
    model.add(Dropout(0.2))
    #model.add(Dense(64, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid')) #Output layer

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',metrics=['accuracy'])

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch,verbose=1)
    score = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    np.set_printoptions(threshold=sys.maxsize)  
    predicted = model.predict(x_test)
    #print(predicted)
    label_predicted = []
    for p in predicted:
        if p>=0.5:
            label_predicted.append(1)
        else:
            label_predicted.append(0)
    pos_label=1
    val_baccuracy = balanced_accuracy_score(y_test, label_predicted)
    val_precision_suff = precision_score(y_test, label_predicted, labels=[pos_label],pos_label=1, average ='binary')
    val_recall_suff = recall_score(y_test, label_predicted, labels=[pos_label],pos_label=1, average ='binary')
    val_f1_suff = f1_score(y_test, label_predicted, labels=[pos_label], pos_label=1, average='binary')
    val_auc = roc_auc_score(y_test, label_predicted)


    pos_label=0
    val_precision_insuff = precision_score(y_test, label_predicted, labels=[pos_label],pos_label=0, average ='binary')
    val_recall_insuff = recall_score(y_test, label_predicted, labels=[pos_label],pos_label=0, average ='binary')
    val_f1_insuff = f1_score(y_test, label_predicted, labels=[pos_label], pos_label=0, average='binary')

    result_list.append([val_baccuracy,val_precision_suff,val_recall_suff,val_f1_suff,val_precision_insuff,val_recall_insuff,val_f1_insuff])

    return [val_baccuracy, val_precision_suff, val_recall_suff, val_f1_suff, val_precision_insuff, val_recall_insuff, val_f1_insuff]
        

def train_model_ml(x_train, x_test, y_train, y_test):

    oversample = ADASYN()
    x_train, y_train = oversample.fit_resample(x_train, y_train)
    clf = RandomForestClassifier().fit(x_train,y_train)
    #clf = DecisionTreeClassifier.fit(x_train,y_train)
    #clf = LogisticRegression.fit(x_train,y_train)
    #clf = LinearSVC.fit(x_train,y_train)

    label_predicted = clf.predict(x_test)

    pos_label=1
    val_baccuracy = balanced_accuracy_score(y_test, label_predicted)
    val_precision_suff = precision_score(y_test, label_predicted, labels=[pos_label],pos_label=1, average ='binary')
    val_recall_suff = recall_score(y_test, label_predicted, labels=[pos_label],pos_label=1, average ='binary')
    val_f1_suff = f1_score(y_test, label_predicted, labels=[pos_label], pos_label=1, average='binary')


    pos_label=0
    val_precision_insuff = precision_score(y_test, label_predicted, labels=[pos_label],pos_label=0, average ='binary')
    val_recall_insuff = recall_score(y_test, label_predicted, labels=[pos_label],pos_label=0, average ='binary')
    val_f1_insuff = f1_score(y_test, label_predicted, labels=[pos_label], pos_label=0, average='binary')

    
    result_list.append([val_baccuracy,val_precision_suff,val_recall_suff,val_f1_suff,val_precision_insuff,val_recall_insuff,val_f1_insuff])
    return [val_baccuracy, val_precision_suff, val_recall_suff, val_f1_suff, val_precision_insuff, val_recall_insuff, val_f1_insuff]



    
def train(x_train,x_test,y_train,y_test, fold_no):
    x_train = tokenizer (x_train)
    x_test = tokenizer (x_test)
    index_dict, word_vectors,combined=word2vec_train(x_train)
    x_train = input_transform(x_train)
    x_test = input_transform(x_test)
    n_symbols,embedding_weights=get_data(index_dict, word_vectors,combined)
    print('Fold #: ', fold_no)
    result = train_model_dl(n_symbols,embedding_weights,x_train,y_train, x_test,y_test)
    #result = train_model_ml(x_train, x_test, y_train, y_test)
    return result




def StratifiedKFold_train():
    data_full=pd.read_csv(path  + '_' + sys.argv[1] + '.csv', usecols=[0,1,2,3,4,5,6,7], engine='python') 
    dataset = data_full.values
    classes = dataset[:, 5]
    data=data_full['Feature'].values.tolist()

    X = np.array(data)
    Y = list(classes)

    skf = StratifiedKFold(n_splits=10,shuffle=True)
    fold_no = 0
    
    for train_index,test_index in skf.split(X,Y):
        numpy_y = np.array(Y)
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = numpy_y[train_index], numpy_y[test_index]
        train(x_train, x_test, y_train, y_test, fold_no)
        fold_no += 1
    balanced_accuracy = 0.0
    precision_suff = 0.0
    recall_suff = 0.0
    f1_suff = 0.0
    precision_insuff = 0.0
    recall_insuff = 0.0
    f1_insuff = 0.0
    for r in result_list:
        balanced_accuracy += r[0]
        precision_suff += r[1]
        recall_suff += r[2]
        f1_suff += r[3]
        precision_insuff += r[4]
        recall_insuff += r[5]
        f1_insuff += r[6]
    avg_balanced_accuracy = balanced_accuracy/fold_no
    avg_precision_suff = precision_suff/fold_no
    avg_recall_suff = recall_suff/fold_no
    avg_f1_suff = f1_suff/fold_no
    avg_precision_insuff = precision_insuff/fold_no
    avg_recall_insuff = recall_insuff/fold_no
    avg_f1_insuff = f1_insuff/fold_no
    print("Avg Balanced Accuracy: ", avg_balanced_accuracy)
    print("Avg Precision - Adequate: ", avg_precision_suff)
    print("Avg Recall - Adequate: ", avg_recall_suff)
    print("Avg F1 - Adequate: ", avg_f1_suff)
    print("Avg Precision - Inadequate: ", avg_precision_insuff)
    print("Avg Recall - Inadequate: ", avg_recall_insuff)
    print("Avg F1 - Inadequate: ", avg_f1_insuff)



if __name__=='__main__':
    StratifiedKFold_train()




