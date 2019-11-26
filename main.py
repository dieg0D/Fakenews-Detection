from keras.models import Sequential
from keras import layers
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from math import sqrt
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Cosntantes que usaremos em nosso código
MAX_SEQUENCE_LENGTH = 5000
MAX_NUM_WORDS = 25000
EMBEDDING_DIM = 300
TEST_SPLIT = 0.3
TEXT_DATA = '/home/diego/Documents/unb/pw/Fakenews-Detection/data/fake_or_real_news.csv'


# Leitura da base de dados e Limpeza dos dados (remover colunas de texto em branco, etc...)
df = pd.read_csv(TEXT_DATA)
df.drop(labels=['id', 'title'], axis='columns', inplace=True)
mask = list(df['text'].apply(lambda x: len(x) > 0))
df = df[mask]


# Como utilizamos aprendiagem supervisionada separamos em dois dataframes um com os textos e outro com as labels (FAKE/REAL)
texts = df['text']
labels = df['label']



#Criação dos tokens que iremos enviar para a rede convolucional
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
data = pad_sequences(sequences,
                     maxlen=MAX_SEQUENCE_LENGTH,
                     padding='pre',
                     truncating='pre')

# Separação dos dados para traino e teste
x_train, x_test, y_train, y_test = train_test_split(data,
                                                  labels.apply(
                                                      lambda x: 0 if x == 'FAKE' else 1),
                                                  test_size=TEST_SPLIT)


# Construção da 1D convnet utilizando global maxpooling
model = Sequential(
    [
        # part 1: word and sequence processing
        layers.Embedding(num_words,
                         EMBEDDING_DIM,
                         input_length=MAX_SEQUENCE_LENGTH,
                         trainable=True),
        layers.Conv1D(128, 5, activation='relu'),
        layers.GlobalMaxPooling1D(),

        # part 2: classification
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()

# Treinamento do modelo
history = model.fit(x_train,
                    y_train,
                    batch_size=128,
                    epochs=10,
                    validation_data=(x_test, y_test))

# Salvando o modelo para podermos reutilziar mais tarde 
model.save("./model.h5")
