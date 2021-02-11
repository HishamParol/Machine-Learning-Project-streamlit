from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
import streamlit as st


def tfneuralnetwork(ip_dim, op_dim, hlayers, ds_name, input_data, output_data, batch_size, epochs, val_split):
	model = Sequential()
	model.add(Dense(hlayers, input_dim=ip_dim, activation='relu'))
	model.add(Dense(op_dim, activation='relu'))
	tensorboard = TensorBoard(log_dir="logs\{}".format(ds_name))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(input_data, output_data, batch_size=batch_size, epochs=epochs, validation_split=val_split, callbacks=[tensorboard])
	return model