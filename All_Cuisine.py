# -*- coding: utf-8 -*-
"""
Text classification for restaurant cuisine type identification
Author: Shahreen Muntaha Nawfee
Created on Wed May  15 00:01:25 2019
"""

""" Modules imported for the application"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB #for machine larning
from sklearn.pipeline import Pipeline

""" Extract the required dataset
"""
#load the csv file
cuisine = pd.read_csv("Trainingset.csv")

#print the column containing business name only
res = cuisine['business_n']
#print(res)

#print the column containing cuisine type only
type = cuisine['Cuisine-type']
#print(type)


""" Text transformation
"""
#Assign each word an integer id
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(res)

#transform vectorize data to array
b = X_train_counts.toarray()
#print(b)

#get feature name 
a = count_vect.get_feature_names()
#print(a)

#count no.of times each word occurs in each document
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#print(X_train_tfidf.shape) #gives dimension of the term matrix


"""Classifying the text using Naive Bayes method
"""
#train the NB classifier with the training data
clf = MultinomialNB().fit(X_train_tfidf, type)

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
('clf', MultinomialNB())])
text_clf = text_clf.fit(res, type)
#print(text_clf)

""" Run the classifier on test dataset
"""
#test the performance of the NB classifier on test set
cuisine_test = pd.read_csv("Testset.csv")
predicted = text_clf.predict(res)
print(predicted)


""" Accuracy assessment of the classifier
"""
Accuracy= np.mean(predicted == type)
print(Accuracy)

