import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

df = pd.read_csv('./data/train.tsv', sep='\t', header=0)
feels = np.array(df['Sentimento'])
feelings = [0, 0, 0, 0, 0]
for f in feels:
    feelings[f] += 1

plt.figure(num=1, figsize=(9, 3))
plt.bar(['negativo', 'um pouco negativo', 'neutro',
         'um pouco positivo', 'positivo'], feelings)
plt.show()

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

# Vamos ver a veradeira distribuição de classes em relação às sentenças
plt.figure(num=2, figsize=(9, 3))
plt.bar(['negativo', 'um pouco negativo', 'neutro',
         'um pouco positivo', 'positivo'], feelings)
plt.show()

count_vect = CountVectorizer()
X_counts = count_vect.fit_transform(df.Texto)
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)
X_train = X_tfidf[:-1000]
X_test = X_tfidf[-1000:]
Y_train = df.Sentimento[:-1000]
Y_test = df.Sentimento[-1000:]

MNB = MultinomialNB()
MNB.fit(X_train,Y_train)

print("Acurácia Multinomial Naive Bayes",MNB.score(X_test, Y_test)*100)

SVC = LinearSVC()
SVC.fit(X_train,Y_train)

print("Acurácia Linear Support Vector Classification",SVC.score(X_test, Y_test)*100)

MLP = Sequential()
MLP.add(Dense(units=32, activation='relu', input_dim=X_train.shape[1]))
MLP.add(Dropout(0.15))
MLP.add(Dense(units=64, activation='relu'))
MLP.add(Dropout(0.15))
MLP.add(Dense(units=128, activation='relu'))
MLP.add(Dropout(0.15))
MLP.add(Dense(units=5, activation='softmax'))
MLP.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])

Y = []
for label in df.Sentimento[:-1000]:
    l = [0 for _ in range(5)]
    l[label] = 1
    Y.append(l)
MLP.fit(X_train, np.array(Y), epochs=500, batch_size=128)
y_test = []
for label in Y_test:
    l = [0 for _ in range(5)]
    l[label] = 1
    y_test.append(l)
loss_and_metrics = MLP.evaluate(X_test, np.array(y_test), batch_size=128)

print("Acurácia Multilayer Perceptron",loss_and_metrics[1]*100)

# Separo por classes
feels = np.array(df['Sentimento'])
feelings = [0, 0, 0, 0, 0]
for f in feels:
    feelings[f] += 1

df_class_0 = df[df['Sentimento'] == 0]
df_class_1 = df[df['Sentimento'] == 1]
df_class_2 = df[df['Sentimento'] == 2]
df_class_3 = df[df['Sentimento'] == 3]
df_class_4 = df[df['Sentimento'] == 4]

# Faço o undersample das classes majoritárias
df_class_1 = df_class_1.sample(feelings[0])
df_class_2 = df_class_2.sample(feelings[0])
df_class_3 = df_class_3.sample(feelings[0])
df_class_4 = df_class_4.sample(feelings[0])

# Crio um novo DataFrame para classes balanceadas
balanced = pd.concat([df_class_0, df_class_1, df_class_2, df_class_3, df_class_4], axis=0)
balanced = balanced.sample(frac=1).reset_index(drop=True)
feels = np.array(balanced['Sentimento'])
feelings = [0, 0, 0, 0, 0]
for f in feels:
    feelings[f] += 1
plt.figure(num=3, figsize=(9, 3))
plt.bar(['negativo', 'um pouco negativo', 'neutro',
         'um pouco positivo', 'positivo'], feelings)
plt.show()

X_counts = count_vect.fit_transform(balanced.Texto)
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)
X_train = X_tfidf[:-1000]
X_test = X_tfidf[-1000:]
Y_train = balanced.Sentimento[:-1000]
Y_test = balanced.Sentimento[-1000:]

MNB = MultinomialNB()
MNB.fit(X_train,Y_train)

print("Acurácia Multinomial Naive Bayes",MNB.score(X_test, Y_test)*100)

SVC = LinearSVC()
SVC.fit(X_train,Y_train)

print("Acurácia Linear Support Vector Classification",SVC.score(X_test, Y_test)*100)

MLP = Sequential()
MLP.add(Dense(units=32, activation='relu', input_dim=X_train.shape[1]))
MLP.add(Dropout(0.15))
MLP.add(Dense(units=64, activation='relu'))
MLP.add(Dropout(0.15))
MLP.add(Dense(units=128, activation='relu'))
MLP.add(Dropout(0.15))
MLP.add(Dense(units=5, activation='softmax'))
MLP.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])

Y = []
for label in balanced.Sentimento[:-1000]:
    l = [0 for _ in range(5)]
    l[label] = 1
    Y.append(l)
MLP.fit(X_train, np.array(Y), epochs=500, batch_size=128)

y_test = []
for label in Y_test:
    l = [0 for _ in range(5)]
    l[label] = 1
    y_test.append(l)
loss_and_metrics = MLP.evaluate(X_test, np.array(y_test), batch_size=128)

print("Acurácia Multilayer Perceptron",loss_and_metrics[1]*100)