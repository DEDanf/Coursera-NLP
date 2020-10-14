import numpy as np
from utils import process_tweet, build_freqs

def sigmoid(z):

    return 1/(1+np.exp(-z))

def gradientDescent(x, y, theta, alpha, num_iters):

    m = len(x)
    for i in range(0, num_iters):

        z = np.dot(x,theta)
        h = sigmoid(z)
        J = (-1*(np.dot(np.transpose(y),np.log(h))+np.dot(np.transpose(1-y),np.log(1-h))))/m
        theta = theta - (alpha/m)*(np.dot(np.transpose(x),(h-y)))
    
    J = float(J)
    return J, theta

def extract_features(tweet, freqs):

    word_l = process_tweet(tweet)   
    x = np.zeros((1, 3))   
    x[0,0] = 1 
    
    for word in word_l:

        x[0,1] += freqs.get((word,1.0),0)
        x[0,2] += freqs.get((word,0.0),0)

    assert(x.shape == (1, 3))
    return x

def predict_tweet(tweet, freqs, theta):
    
    x = extract_features(tweet, freqs)
    y_pred = sigmoid(np.dot(x,theta))
    
    return y_pred

def test_logistic_regression(test_x, test_y, freqs, theta):

    y_hat = []
    
    for tweet in test_x:

        y_pred = predict_tweet(tweet, freqs, theta)
        
        if y_pred > 0.5:
            y_hat.append(1)
        else:
            y_hat.append(0)

    accuracy = (y_hat==np.squeeze(test_y)).sum()/len(test_x)

    return accuracy

