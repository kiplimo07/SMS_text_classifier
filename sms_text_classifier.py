# import libraries
try:
  # %tensorflow_version only exists in Colab.
  !pip install tf-nightly
except Exception:
  pass
import tensorflow as tf
import pandas as pd
from tensorflow import keras
!pip install tensorflow-datasets
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# get data files
!wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
!wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

# Load and preprocess the data
train_data = pd.read_csv(train_file_path, sep='\t', header=None, names=['label', 'message'])
test_data = pd.read_csv(test_file_path, sep='\t', header=None, names=['label', 'message'])

# Convert labels to numerical values: 'ham' -> 0, 'spam' -> 1
train_data['label'] = train_data['label'].map({'ham': 0, 'spam': 1})
test_data['label'] = test_data['label'].map({'ham': 0, 'spam': 1})

# Tokenize the text
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_data['message'].values)
train_sequences = tokenizer.texts_to_sequences(train_data['message'].values)
test_sequences = tokenizer.texts_to_sequences(test_data['message'].values)

# Pad the sequences to ensure uniform input size
train_padded = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, padding='post', maxlen=100)
test_padded = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, padding='post', maxlen=100)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 32, input_length=100),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_padded, train_data['label'].values, epochs=15, validation_data=(test_padded, test_data['label'].values))

# function to predict messages based on model
# (should return list containing prediction and label, ex. [0.008318834938108921, 'ham'])
def predict_message(pred_text):
      # Preprocess the input text
    sequence = tokenizer.texts_to_sequences([pred_text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, padding='post', maxlen=100)
    
    # Predict using the model
    prediction = model.predict(padded_sequence)[0][0]
    label = 'spam' if prediction > 0.5 else 'ham'
    
    return [prediction, label]
#return (prediction)

pred_text = "how are you doing today?"

prediction = predict_message(pred_text)
print(prediction)

# Run this cell to test your function and model. Do not modify contents.
def test_predictions():
  test_messages = ["how are you doing today",
                   "sale today! to stop texts call 98912460324",
                   "i dont want to go. can we try it a different day? available sat",
                   "our new mobile video service is live. just install on your phone to start watching.",
                   "you have won £1000 cash! call to claim your prize.",
                   "i'll bring it tomorrow. don't forget the milk.",
                   "wow, is your arm alright. that happened to me one time too"
                  ]

  test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
  passed = True

  for msg, ans in zip(test_messages, test_answers):
    prediction = predict_message(msg)
    if prediction[1] != ans:
      passed = False

  if passed:
    print("You passed the challenge. Great job!")
  else:
    print("You haven't passed yet. Keep trying.")

test_predictions()
