import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import random

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Load intents file
with open('intents.json', 'r', encoding='utf-8') as file:
    data_file = file.read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))

        # Add to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word, remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Print statistics
print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

# Save words and classes to pickle files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle features and convert to np.array
random.shuffle(training)
training = np.array(training)

# Create training and testing lists
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")

# Create model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),)))
model.add(BatchNormalization())  # Add Batch Normalization
model.add(LeakyReLU(alpha=0.1))  # Use LeakyReLU
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(BatchNormalization())  # Add Batch Normalization
model.add(LeakyReLU(alpha=0.1))  # Use LeakyReLU
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model with Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Early Stopping Callback
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

# Fit the model with Early Stopping
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1, callbacks=[early_stopping])

# Save the model
model.save('chatbot_model.h5')

print("Model created and saved")
