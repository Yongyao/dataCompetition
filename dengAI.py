#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 21:44:05 2017

@author: yjiang
"""

import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.preprocessing import Imputer


# load dataset
dataframe = pandas.read_csv("data/dengue_features_train.csv")
test = pandas.read_csv("data/dengue_features_test.csv")

dataset = dataframe.values
test_dataset = test.values

# split into input (X) and output (Y) variables
X = dataset[:,4:24]
Y = dataset[:,24]

test_X = test_dataset[:,4:24]

# impute missing values
imputer = Imputer()
X = imputer.fit_transform(X)
test_X = imputer.fit_transform(test_X)

scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
test_X = scaler.transform(test_X)

# define base model
def baseline_model():
    model = Sequential()
    model.add(Dense(23, input_dim=20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(15, kernel_initializer='normal', activation='relu'))
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=50, batch_size=50, verbose=0)

estimator.fit(X, Y)
prediction = estimator.predict(test_X)
pandas.concat([pandas.DataFrame(test_dataset[:,0:3]),pandas.DataFrame(prediction.astype(int))], 
               axis=1).to_csv('data/test_result.csv', index=False, 
               header=['city', 'year', 'weekofyear', 'total_cases'])
#==============================================================================
# output = np.column_stack((test_dataset[:,0:4], prediction))
# np.savetxt("data/test_result.csv", output, delimiter=",")
#==============================================================================

#==============================================================================
# from sklearn.metrics import mean_absolute_error
# score = mean_absolute_error(Y_test, clf.predict(X_test))
#==============================================================================
#==============================================================================
# kfold = KFold(n_splits=4, random_state=seed)
# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
#==============================================================================
