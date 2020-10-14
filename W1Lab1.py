import nltk
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import matplotlib.pyplot as plt
import random
import re
import string
from utils import process_tweet
from utils import build_freqs


#Download and initiate relevant arrays of twitter sample

#nltk.download('twitter_samples')
#nltk.download('stopwords')

all_pos = twitter_samples.strings('positive_tweets.json')
all_neg = twitter_samples.strings('negative_tweets.json')

#Study Data with pie chart and vals

#print(len(all_pos))
#print(len(all_neg))
#print(type(all_pos))
#print(type(all_neg[0]))

#fig = plt.figure(figsize=(5,5))
#labels = 'Positives', 'Negatives'
#sizes = [len(all_pos), len(all_neg)]

#plt.pie(sizes,labels=labels,autopct='%1.1f%%',shadow=True,startangle=90)
#plt.axis('equal')
#plt.show()

#1. Remove hyperlinks, twitter marks and styles

tweet = all_pos[2277]
print(tweet)

tweet = re.sub(r'^RT[\s]+', '', tweet)
print(tweet)

tweet = re.sub(r'https?:\/\/.*[\r\n]*','',tweet)
print(tweet)

tweet = re.sub(r'#','',tweet)
print(tweet)

#2. Tokenize string

tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True)
tweet = tokenizer.tokenize(tweet)
print(tweet)

#3. Remove stop words and punctuation

stopwords_english = stopwords.words('english')
#print(string.punctuation)

clean_tweet = []

for word in tweet:
    if(word not in stopwords_english and word not in string.punctuation):
        clean_tweet.append(word)

print(clean_tweet)

#4. Stemming

stemmer = PorterStemmer()

tweets_stem = []

for word in clean_tweet:
    stem = stemmer.stem(word)
    tweets_stem.append(stem)

print(tweets_stem)

#processing function

tweet2 = all_pos[2277]

tweet2_stem = process_tweet(tweet2)

print(tweet2_stem)


