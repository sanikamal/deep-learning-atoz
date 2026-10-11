# Long short-term memory recurrent neural network for sequence classification.
# =========================================================================== 
# The model expects 100 time steps of one feature as input. The model has a single
# LSTM hidden layer to extract features from the sequence, followed by a fully connected
# layer to interpret the LSTM output, followed by an output layer for making binary predictions.

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
# from tensorflow.keras.utils import plot_model

visible = Input(shape=(100,1))
hidden1 = LSTM(10)(visible)
hidden2 = Dense(10, activation= 'relu' )(hidden1)
output = Dense(1, activation= 'sigmoid' )(hidden2)
model = Model(inputs=visible, outputs=output)

# summarize layers
model.summary()

# plot graph
# plot_model(model, to_file= ' lstm.png ' )