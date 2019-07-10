from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from keras.preprocessing import text, sequence
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

colnames = ["product/productId",
            "review/userId",
            "review/helpfulness",
            "review/score",
            "review/time",
            "review/summary",
            "review/text"]
df = pd.read_csv("data/finemuged.csv", encoding='latin1', header=None,
                 names=colnames, quotechar="\"").sample(100000)

def one_hot(x, maxi=5):
    arr = [0]*maxi
    arr[x-1] = 1
    return arr

t = text.Tokenizer(10000)
X = df["review/text"].values
t.fit_on_texts(X)
X = t.texts_to_sequences(X)
X = sequence.pad_sequences(X, value=0, padding='post', maxlen=256)
y = df["review/score"].astype(int).values.reshape(-1) - 1
y = np.eye(5)[y]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)
# exit(0)


sequence_length = 256
vocabulary_size = 10000
embedding_dim = 300
filter_sizes = [3,4,5]
num_filters = 512
drop = 0.5

epochs = 5
batch_size = 1024

# this returns a tensor
print("Creating Model...")
inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim,
                      input_length=sequence_length)(inputs)
reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim),
                padding='valid', kernel_initializer='normal',
                activation='relu')(reshape)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim),
                padding='valid', kernel_initializer='normal',
                activation='relu')(reshape)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim),
                padding='valid', kernel_initializer='normal',
                activation='relu')(reshape)

maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1),
                      strides=(1,1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1),
                      strides=(1,1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1),
                      strides=(1,1), padding='valid')(conv_2)

concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)
output = Dense(5, activation='softmax')(dropout)

# this creates a model that includes
model = Model(inputs=inputs, outputs=output)

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
print("Traning Model...")
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
          validation_data=(X_test, y_test))

