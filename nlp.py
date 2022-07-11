# Communication between natural (human) languages and computer languages
# ex: spellcheck / autocomplete
# how computers can understand and/or process natural/human languages

# RNN (Recurrent Neural Networks)
# * Sentiment Analysis
# * Character Generation

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import utils
import tensorflow as tf 
import os
import numpy as np 

VOCAB_SIZE = 88584
MAXLEN = 250
BATCH_SIZE= 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)

# preprocessing
train_data = utils.pad_sequences(train_data, MAXLEN)
test_data = utils.pad_sequences(test_data, MAXLEN)

# model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

# Training
model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=['acc'])
history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)

# evaluating
results = model.evaluate(test_data, test_labels)
print(results)

# Predictions
word_index = imdb.get_word_index()

def encode_text(text):
    tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return utils.pad_sequences([tokens], MAXLEN)[0]

# text = "that move was just amazing, so amazing"
# encoded = encode_text(text)
# print(encoded)
# reverse_word_index = {value: key for (key, value) in word_index.items()}

def decode_integers(integers):
    PAD = 0
    text= ""
    for num in integers:
        if num != PAD:
            text += reverse_word_index[num] + " "
    return text[:-1]

def predict(text):
    encoded_text = encode_text(text)
    pred = np.zeros((1, 250))
    pred[0] = encoded_text
    result = model.predict(pred)
    print(result[0])

positive_review = "That movie was so awsome! I really loved it and would watch it again because it was amazingly great" 
predict(positive_review)

negative_review = "That movie sucked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched"
predict(negative_review)