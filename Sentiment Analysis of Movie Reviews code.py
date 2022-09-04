#importing necessary modules
import tensorflow as tf
import numpy as np
import keras
from keras.preprocessing.text import text_to_word_sequence
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense


vocab_size = 88584

#pad sequences function
def pad_sequences(data_list, maxlen):
    returned = []
    for data in data_list:
        if len(data)<maxlen:
            for i in range(maxlen - len(data)):
                data = [0] + data
            returned.append(data)
        elif len(data)>maxlen:
            for i in range(len(data)-1, maxlen-1, -1):
                data.pop(i)
            returned.append(data)
        else:
            returned.append(data)
    return np.array(returned, dtype="int32")

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)
#print(train_data[1]) # stored as integers, with each integer being linked to a word
# data is different lengths so you need to pad the data so they are all same length
# pad_sequences adds 0s to the start to make the length 250, and trims it if longer than 250
maxlen = 250
#pad the training data and test data
train_data = pad_sequences(train_data, maxlen)
test_data = pad_sequences(test_data, maxlen)
"""
#create sequential model
model = Sequential()
#embedding the words, creating vector of shape 32 that will show whether word is positive or negative
model.add(Embedding(vocab_size, 32))
#LSTM layer which can access inputs from any timestep in the past, leaving previous inputs to not dissapear over time
model.add(LSTM(32))
model.add(Dense(1, activation="sigmoid"))
model.summary()

model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=['acc'])
history = model.fit(train_data, train_labels, epochs=10)
results = model.evaluate(test_data, test_labels)
model.save("Sentiment Analysis of Movie Reviews")
"""

model = keras.models.load_model("Sentiment Analysis of Movie Reviews")

word_index = imdb.get_word_index()

def encode_text(text):
  tokens = keras.preprocessing.text.text_to_word_sequence(text)
  tokens = [word_index[word] if word in word_index else 0 for word in tokens]
  return pad_sequences([tokens], maxlen)[0]
text = 'this was a good movie. very good. great. the best. excellent.'
def predict(text):
    encoded = encode_text(text)
    pred = np.zeros((1,250))
    pred[0] = encoded
    result = model.predict(pred)
    if result[0] >0.5:
        percentage = (result[0]-0.5)*200
        print(f"I am {float(percentage)}% sure that it is positive")
    else:
        percentage = ((1-result[0]) - 0.5)*200
        print(f"I am {float(percentage)}% sure that it is negative")

#enter reviews in the list, then run and it will tell you whether it was positive or negative
list = ["The movie was incredible! I loved it."]
for review in list:
    predict(review)