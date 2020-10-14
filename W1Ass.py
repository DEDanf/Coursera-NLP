import nltk
from os import getcwd
from utils import process_tweet, build_freqs
from functions import sigmoid, gradientDescent, extract_features, predict_tweet, test_logistic_regression

import numpy as np
import pandas as pd

from nltk.corpus import twitter_samples

all_pos = twitter_samples.strings('positive_tweets.json')
all_neg = twitter_samples.strings('negative_tweets.json')

tweets = all_pos+all_neg
labels = np.append(np.ones((len(all_pos))), np.zeros((len(all_neg))))

train_pos = all_pos[:4000]
test_pos = all_pos[4000:]
train_neg = all_neg[:4000]
test_neg = all_pos[4000:]

train_x = train_pos+train_neg
test_x = test_pos+test_neg

train_y = np.append(np.ones((len(train_pos),1)), np.zeros((len(train_neg),1)), axis=0)
test_y = np.append(np.ones((len(test_pos),1)), np.zeros((len(test_neg),1)), axis=0)

freqs = build_freqs(train_x, train_y)

X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :]= extract_features(train_x[i], freqs)

# training labels corresponding to X
Y = train_y

# Apply gradient descent
J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)
print(f"The cost after training is {J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")


tmp_accuracy = test_logistic_regression(test_x, test_y, freqs, theta)
print(f"Logistic regression model's accuracy = {tmp_accuracy:.4f}")



