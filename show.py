# Code made by Mateus Machado for Intelivix analysis
# Importing neccessary libraries
import os

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from pandas import DataFrame

# Marcelo gave us a dataset, but we don't know how bamuch balanced it is.
# First, get the dataset as Pandas' Dataframe to be easier to work
df = pd.read_csv('./data/train.tsv', sep='\t', header=0)
# Then let's get only the classes
feels = np.array(df['Sentimento'])
feelings = [0, 0, 0, 0, 0]
for f in feels:
    feelings[f] += 1
# Let's plot a beautiful bar graph to see the data balancing.
plt.figure(num=1, figsize=(9, 3))
plt.bar(['negativo', 'um pouco negativo', 'neutro',
         'um pouco positivo', 'positivo'], feelings)
plt.show()
# As we can see, it's a dataset far from be balanced. But why exactly?
# If you check the data, it gives first the full sentence, then derivates it.
# So let's check how really balanced are the sentences.
# First, we get only the full sentences rows.
sentences = [0]
ids = []
last = len(sentences)
for i, idsent in enumerate(df.IdSentenca):
    if idsent > sentences[-1]:
        sentences.append(idsent)
        ids.append(i)

feelings = [0, 0, 0, 0, 0]
for i in ids:
    feelings[df.iloc[i]['Sentimento']] += 1

# So let's plot another bar graph to check if it's a real tricky dataset.
plt.figure(num=2, figsize=(9, 3))
plt.bar(['negativo', 'um pouco negativo', 'neutro',
         'um pouco positivo', 'positivo'], feelings)
plt.show()

# Turns out it's an imbalanced and inconstant dataset. It has 8530 sentences, and not 8544. So let's balance it.
# There are some ways to balance a dataset, including some machine learning algorithms (LVQs)
# But today, we'll do the basics, taking samples of the majorities.
# Instead of duplicating sentences, We'll choose some derivated sentences classified as minorities to increment the minorities classes.
