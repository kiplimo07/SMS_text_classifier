# SMS Spam Classifier

## Overview

This repository contains a machine learning model that classifies SMS messages into two categories: "ham" (normal messages from friends) and "spam" (advertisements or messages from companies). The model is built using TensorFlow and Keras, and it is trained on the SMS Spam Collection dataset.

## Features

- **Text Classification**: Classifies SMS messages as either "ham" or "spam".
- **Neural Network**: Utilizes a deep learning model with embedding and LSTM layers.
- **Data Preprocessing**: Includes tokenization and padding of text data.
- **Evaluation**: Provides accuracy metrics to evaluate model performance.

## Getting Started

### Prerequisites

Ensure you have the following Python packages installed:

- `tensorflow`
- `pandas`
- `numpy`
- `matplotlib`
- `tensorflow-datasets`

You can install these packages using pip:

```bash
pip install tensorflow tf-nightly pandas numpy matplotlib tensorflow-datasets
```

### Data

The dataset used in this project is the [SMS Spam Collection dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset). It is divided into training and validation sets.

- **Training Data**: [train-data.tsv](https://cdn.freecodecamp.org/project-data/sms/train-data.tsv)
- **Validation Data**: [valid-data.tsv](https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv)

### Model Architecture

The model is built with the following architecture:

1. **Embedding Layer**: Converts words into dense vectors of fixed size.
2. **LSTM Layer**: Captures temporal dependencies in the text data.
3. **Global Average Pooling Layer**: Reduces the dimensionality of the LSTM output.
4. **Dense Layers**: Two dense layers with ReLU and sigmoid activation functions for classification.

### Training

The model is trained with the following configuration:

- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 15

To train the model, run the following code:

```python
# Build and compile the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 32, input_length=100),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(train_padded, train_data['label'].values, epochs=15, validation_data=(test_padded, test_data['label'].values))
```

### Prediction

To classify a new SMS message, use the `predict_message` function:

```python
def predict_message(pred_text):
    sequence = tokenizer.texts_to_sequences([pred_text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, padding='post', maxlen=100)
    prediction_prob = model.predict(padded_sequence)[0][0]
    label = 'spam' if prediction_prob > 0.5 else 'ham'
    return [prediction_prob, label]
```

### Testing

The `test_predictions` function is used to verify the accuracy of the model with predefined test messages:

```python
def test_predictions():
    test_messages = [
        "how are you doing today",
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
```

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
