import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import numpy as np
from utils import process_tweet,build_freqs

all_pos = twitter_samples.strings('positive_tweets.json')
all_neg = twitter_samples.strings('negative_tweets.json')

tweets = all_pos+all_neg

#construct output labels

labels = np.append(np.ones((len(all_pos))), np.zeros((len(all_neg))))

freqs = build_freqs(tweets,labels)

print(freqs[('happi', 1)])
print(freqs[('happi', 0)])

#display info

keys = ['happi', 'merri', ':)', ':(', 'sad', 'bad']

data = []

for word in keys:

    pos = 0
    neg = 0

    if(word,1) in freqs:
        pos = freqs[(word,1)]

    if(word,0) in freqs:
        neg = freqs[(word,0)]

    data.append([word, pos, neg])

#plot info

fig, ax = plt.subplots(figsize=(8,8))

x = np.log([x[1] + 1 for x in data])

y = np.log([x[2] + 1 for x in data])

ax.scatter(x,y)

plt.xlabel("log +ve")
plt.ylabel("log -ve")

for i in range(0, len(data)):

    ax.annotate(data[i][0], (x[i], y[i]), fontsize=12)

ax.plot([0,9], [0,9], color='red')
plt.show()