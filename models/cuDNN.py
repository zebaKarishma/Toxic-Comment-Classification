import numpy as np
import pandas as pd
import os
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, CuDNNGRU, Conv1D
from keras.preprocessing import text, sequence
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
print(tf.__version__)
tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
)
EMBEDDING_FILES = [
    'crawl-300d-2M.vec',
    'glove.840B.300d.txt'
]

BATCH_SIZE = 512
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
EPOCHS = 4
MAX_LEN = 220


TEXT_COLUMN = 'comment_text'
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
submission = pd.read_csv("sample_submission.csv")

y = train_df[list_classes].values
x_train = train_df[TEXT_COLUMN].astype(str)
y_train = y
x_test = test_df[TEXT_COLUMN].astype(str)

embedding_matrix = np.load('embedding_matrix_big.npy')
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)


def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            pass
    return embedding_matrix

def build_model(embedding_matrix):
    words = Input(shape=(None,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    #x = SpatialDropout1D(0.2)(x)

    x1 = SpatialDropout1D(0.2)(x)

    x = Bidirectional(CuDNNGRU(LSTM_UNITS, return_sequences = True))(x1)
    x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)
    
    y = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences = True))(x1)
    y = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(y)


    avg_pool1 = GlobalAveragePooling1D()(x)
    max_pool1 = GlobalMaxPooling1D()(x)
   
    avg_pool2 = GlobalAveragePooling1D()(y)
    max_pool2 = GlobalMaxPooling1D()(y)
    
   

    x = concatenate([avg_pool1, max_pool1, avg_pool2, max_pool2])


    x = Dense(6, activation = "sigmoid")(x)

    model = Model(inputs = words, outputs = x)

    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    return model

tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE)
tokenizer.fit_on_texts(list(x_train) + list(x_test))

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

X_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, test_size = 0.1)
EPOCHS = 5
SEEDS = 10

pred = 0

for ii in range(SEEDS):
    model = build_model(embedding_matrix)
    for global_epoch in range(EPOCHS):
        print(global_epoch)
        model.fit(
                    X_train,
                    Y_train,
                    validation_data = (X_valid, Y_valid),
                    batch_size=128,
                    epochs=1,
                    verbose=2,
                    callbacks=[
                        LearningRateScheduler(lambda _: 1e-3 * (0.55 ** global_epoch))
                    ]
                )
        val_preds_3 = model.predict(X_valid)
        AUC = 0
        for i in range(6):
             AUC += roc_auc_score(Y_valid[:,i], val_preds_3[:,i])/6.
        print(AUC)

    pred += model.predict(x_test, batch_size = 1024, verbose = 1)/SEEDS
    model.save_weights('model_weights_'+str(ii)+'.h5')
    os.system('gzip '+'model_weights_'+str(ii)+'.h5')

submission[list_classes] = (pred)
submission.to_csv("submission_highacc.csv", index = False)