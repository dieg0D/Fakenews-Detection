from django.db import models

# Import ML
from keras.models import Sequential
from keras import layers
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from math import sqrt
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




MAX_SEQUENCE_LENGTH = 5000
MAX_NUM_WORDS = 25000
EMBEDDING_DIM = 300
TEST_SPLIT = 0.2


# Create your models here.
class FormNews (models.Model):
    veiculo = models.URLField()
    titulo = models.CharField(max_length=200)
    texto = models.TextField()
    predicao = models.FloatField(blank=True)

    def predict(self):
        return 0.0


    def save(self, *args, **kwargs):
        self.predicao = self.predict()
        super().save(*args, **kwargs)