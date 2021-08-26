import numpy as np
import os

from skimage import transform

from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import resnet50
from tensorflow.keras.utils import plot_model


def preprocess_image(image, size):
	img = transform.resize(image, (size, size))
	return img


def create_model(num_classes, img_size):
	model = Sequential()
	model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(img_size, img_size, 3)))
	model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))	
	model.add(Dropout(0.2))	

	model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
	model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))	
	model.add(Dropout(0.2))	

	model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
	model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))	
	model.add(Dropout(0.2))	

	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	model.summary()	
	plot_model(model, to_file='training_model.png', show_shapes=True)

	return model

def create_resnet50(num_classes, img_size):
	base_model = resnet50.ResNet50(weights= None, include_top=False, input_shape= (img_size, img_size, 3))
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dropout(0.7)(x)
	predictions = Dense(num_classes, activation= 'softmax')(x)
	model = Model(inputs = base_model.input, outputs = predictions)
	return model
