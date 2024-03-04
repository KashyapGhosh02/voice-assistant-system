#importing libaies 
import json
import numpy as np
import os
import logging
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, GlobalAveragePooling1D

import tensorflow as tf  # Add this line to import TensorFlow as tf

# from keras import __version__
# tf.keras.__version__ = __version__

# from rl.agents import DQNAgent
# from rl.policy import BoltzmannQPolicy
# from rl.memory import SequentialMemory

with open(r'Data\intents.json') as file:
    data = json.load(file)

# training_sentences = []
# training_labels = []
# labels = []
# #responses = []

# for intent in data['intents']:
#     for pattern in intent['patterns']:
#         training_sentences.append(pattern)
#         training_labels.append(intent['tag'])
#     #responses.append(intent['responses'])

#     if intent['tag'] not in labels:
#         labels.append(intent['tag'])

# num_classes = len(labels)
# lbl_encoder = LabelEncoder()
# lbl_encoder.fit(training_labels)
# training_labels = lbl_encoder.transform(training_labels)
# vocab_size = 1000
# embedding_dim = 16
# max_len = 20
# oov_token = "<OOV>"

# tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
# tokenizer.fit_on_texts(training_sentences)
# word_index = tokenizer.word_index
# sequences = tokenizer.texts_to_sequences(training_sentences)
# padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)
# model = Sequential()
# model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
# model.add(GlobalAveragePooling1D())
# model.add(Dense(16, activation='relu'))
# #model.add(Dense(16, activation='relu'))
# model.add(Dense(num_classes, activation='softmax'))

# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer='adam', metrics=['accuracy'])

# model.summary()
# epochs = 500
# history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)
# # to save the trained model
# accuracy = history.history['accuracy'][-1]  # Get the accuracy from the last epoch
# print(f"Final Accuracy: {accuracy * 100:.2f}%")
# try:
#     tf.keras.models.save_model(model, r"C:\Users\kashy\Documents\testing_project\Data\chat_model")
# except Exception as e:
#     print(f"Error during model saving: {e}")


# import pickle

# # to save the fitted tokenizer
# with open(r'C:\Users\kashy\Documents\testing_project\Data\tokenizer.pickle', 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # to save the fitted label encoder
# with open(r'C:\Users\kashy\Documents\testing_project\Data\label_encoder.pickle', 'wb') as ecn_file:
#     pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
    
# import tensorflow as tf
# # print(f"TensorFlow version: {tf.__version__}")
# # print(f"Keras version: {tf.keras.__version__}")

