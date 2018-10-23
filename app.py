import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
from flask import Flask, render_template, request, Response, stream_with_context, jsonify,flash,Markup
stemmer = LancasterStemmer()
from nltk import word_tokenize

app = Flask(__name__)
# restore all of our data structures
import pickle
data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']
# sentence="can you tell me my ticket status"
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net, tensorboard_dir="tflearn_logs")
# import our chat-bot intents file

with open('intents.json') as json_data:
    intents = json.load(json_data)
# load our saved model
model.load('model.tflearn')
def clean_up_sentence(sentence):
    # tokenize the pattern
    nltk.download('punkt')
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))
ERROR_THRESHOLD = 0.25
@app.route("/")
def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    print(return_list)
    return json.dumps(return_list)

def response(sentence, show_details=False):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # a random response from the intent
                    print (random.choice(i['responses']))
                return (random.choice(i['responses']))
                results.pop(0)

# if __name__ == "__main__":
    # print(classify("call customer support agent"))
    # print(classify("hi"))
    # print(classify("need help from it agent"))
    # print(classify("can you tell me my ticket status"))
    # print(classify("can you raise a ticket for me"))
    # print(classify("raise a ticket"))
    # print(classify("ticket status"))
    # print(classify("sadsadas@asdsa.com"))
    # print(classify("sfsdf"))
    # print(classify("asff"))
    # print(classify("wqee@gmail.com"))
    # print(classify("djs@yahoo.com"))
    # print(classify("want to raise a ticket"))
    # print(classify("tell my ticket status"))
if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8080))
#     app.run(host='0.0.0.0', port=port,debug=True)
    print(classify("call customer support agent"))
    
