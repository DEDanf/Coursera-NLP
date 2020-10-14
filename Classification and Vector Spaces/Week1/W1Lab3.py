import nltk
from nltk.corpus import twitter_samples

from os import getcwd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils import process_tweet,build_freqs

def neg(theta, pos):
    return(-theta[0]-pos*theta[1])/theta[2]

def direction(theta,pos):
    return pos*theta[2]/theta[1]

all_pos = twitter_samples.strings('positive_tweets.json')
all_neg = twitter_samples.strings('negative_tweets.json')

tweets = all_pos+all_neg

labels = np.append(np.ones((len(all_pos))), np.zeros((len(all_neg))))

#split data
train_pos = all_pos[:4000]
train_neg = all_neg[:4000]

train_x = train_pos+train_neg

data = pd.read_csv('logistic_features.csv')

X = data[['bias', 'positive', 'negative']].values
Y = data['sentiment'].values

theta = [7e-08, 0.0005239, -0.00055517]

fig,ax = plt.subplots(figsize=(8,8))

colors = ['red' , 'green']

ax.scatter(X[:,1], X[:,2], c=[colors[int(k)]for k in Y], s=0.1)
plt.xlabel('Pos')
plt.ylabel('Neg')

maxpos = np.max(X[:,1])

offset = 5000

ax.plot([0,maxpos],[neg(theta,0),neg(theta,maxpos)],color = 'gray')

ax.arrow(offset,neg(theta,offset),offset,direction(theta,offset),head_width=500,head_length=500,fc='g',ec='g')
ax.arrow(offset,neg(theta,offset),-offset,-direction(theta,offset),head_width=500,head_length=500,fc='r',ec='r')

plt.show()
