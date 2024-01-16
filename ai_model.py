
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
from dataset import *

def create_model(sequence_length, num_genes,num_features):
    model = Sequential()
    model.add(LSTM(128, input_shape=(sequence_length, num_features)))
    model.add(Dense(num_genes, activation='softmax'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    
    X_train, y_train = generate_data(5, 1000)
    
    model.fit(X_train, y_train, epochs=10, batch_size=3)
    
    return model
        