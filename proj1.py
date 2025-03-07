import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec

csv_path = r"IMDB Dataset.csv"  # Adjust if needed
df = pd.read_csv(csv_path)

def clean_text(text):
    text = re.sub(r"<[^>]*>", "", text)   # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Keep alphanumeric and space
    text = text.lower()                          # lowercase
    return text

df["cleaned"] = df["review"].apply(clean_text)

# Encode sentiment as 0/1
df["label"] = df["sentiment"].apply(lambda x: 1 if x=="positive" else 0)

# For example, 80% train, 20% test
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(
    df[["cleaned", "label"]],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

vocab_size = 10000       # Maximum vocabulary
max_seq_len = 200        # Max words per review
oov_token_str = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token_str)
tokenizer.fit_on_texts(train_data["cleaned"].values)

train_sequences = tokenizer.texts_to_sequences(train_data["cleaned"].values)
test_sequences  = tokenizer.texts_to_sequences(test_data["cleaned"].values)

X_train = pad_sequences(train_sequences, maxlen=max_seq_len, padding='post')
X_test  = pad_sequences(test_sequences,  maxlen=max_seq_len, padding='post')

y_train = train_data["label"].values
y_test  = test_data["label"].values

sentences_for_w2v = [text.split() for text in train_data["cleaned"].values]
w2v_dim = 100  # Embedding dimension (could be 300 if you want)
w2v_model = Word2Vec(
    sentences=sentences_for_w2v,
    vector_size=w2v_dim,
    window=7,
    min_count=1,
    workers=10
)
word_vectors = w2v_model.wv

embedding_matrix = np.zeros((vocab_size, w2v_dim))

for word, idx in tokenizer.word_index.items():
    if idx < vocab_size:
        if word in word_vectors:
            embedding_matrix[idx] = word_vectors[word]

model = models.Sequential()
model.add(layers.Embedding(
    input_dim=vocab_size,
    output_dim=w2v_dim,
    weights=[embedding_matrix],
    input_length=max_seq_len,
    trainable=False
))
model.add(layers.LSTM(128, return_sequences=True))
model.add(layers.Dropout(0.3))
model.add(layers.LSTM(64))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

epochs = 5
batch_size = 64
history = model.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2
)
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
