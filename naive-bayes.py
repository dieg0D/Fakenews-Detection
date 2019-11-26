from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from math import sqrt
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MAX_SEQUENCE_LENGTH = 5000
MAX_NUM_WORDS = 25000
EMBEDDING_DIM = 300
TEST_SPLIT = 0.3

TEXT_DATA = '/home/diego/Documents/unb/pw/Fakenews-Detection/data/fake_or_real_news.csv'

# define a function that allows us to evaluate our models


def evaluate_model(predict_fun, X_train, y_train, X_test, y_test):
    '''
    evaluate the model, both training and testing errors are reported
    '''
    # training error
    y_predict_train = predict_fun(X_train)
    train_acc = accuracy_score(y_train, y_predict_train)

    # testing error
    y_predict_test = predict_fun(X_test)
    test_acc = accuracy_score(y_test, y_predict_test)

    return train_acc, test_acc

# estimate 95% confidence interval on error

# https://stats.stackexchange.com/questions/247551/how-to-determine-the-confidence-of-a-neural-network-prediction
# towards bottom of the page.


def error_conf(error, n):
    term = 1.96*sqrt((error*(1-error))/n)
    lb = error - term
    ub = error + term

    return lb, ub


# read in our data and preprocess it

df = pd.read_csv(TEXT_DATA)
df.drop(labels=['id', 'title'], axis='columns', inplace=True)
# only select stories with lengths gt 0 -- there are some texts with len = 0
mask = list(df['text'].apply(lambda x: len(x) > 0))
df = df[mask]


# prepare text samples and their labels

texts = df['text']
labels = df['label']


# set up vector models for training and testing


# data vectorizer
vectorizer = CountVectorizer(analyzer="word",
                             binary=True,
                             min_df=2,
                             stop_words='english')
docarray = vectorizer.fit_transform(texts).toarray()
docterm = pd.DataFrame(docarray, columns=vectorizer.get_feature_names())


# create training and test data

docterm_train, docterm_test, y_train, y_test = train_test_split(
    docterm, labels, test_size=TEST_SPLIT)


# Naive Bayes Model

model = MultinomialNB()
model.fit(docterm_train, y_train)

# # evaluate model

train_acc, test_acc = evaluate_model(
    model.predict, docterm_train, y_train, docterm_test, y_test)
print("Training Accuracy: {:.2f}%".format(train_acc*100))
print("Testing Accuracy: {:.2f}%".format(test_acc*100))

# estimate 95% confidence interval

n = docterm_test.shape[0]
lb, ub = error_conf(1-test_acc, n)

print(
    "95% confidence interval: {:.2f}%-{:.2f}%".format((1-ub)*100, (1-lb)*100))
