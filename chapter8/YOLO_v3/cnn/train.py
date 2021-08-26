import numpy as np
import glob
import os,sys
from skimage import io
from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import SGD

from util import preprocess_image, create_model

def get_label_from_image_path(image_path, data_path):
	path = image_path.replace(data_path, "");
	paths = path.split("/")
	label = int(paths[0])
	return label


def get_training_data(data_path, num_classes, img_size):
	images = []
	labels = []

	all_image_paths = glob.glob(os.path.join(data_path, '*/*.ppm'))
	np.random.shuffle(all_image_paths)
	print(data_path)
	i = 0
	for image_path in all_image_paths:
		try:
			img = preprocess_image(io.imread(image_path), img_size)
			label = get_label_from_image_path(image_path, data_path)
			images.append(img)
			labels.append(label)
			print("load images: {}".format(i))
			i = i+1
		except(IOError, OSError):
			print("failed to process {}".format(image_path))


	X = np.array(images, dtype='float32')
	y = np.eye(num_classes, dtype='uint8')[labels]

	return X, y


NUM_CLASSES = 43
IMG_SIZE = 48

TRAINING_DATA_PATH = "./GTSRB/Final_Training/Images/"

model = create_model(NUM_CLASSES, IMG_SIZE)
X, y = get_training_data(TRAINING_DATA_PATH, NUM_CLASSES, IMG_SIZE)

learning_rate = 0.01
sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

batch_size = 32
epochs = 30

history = model.fit(X, y,
					batch_size=batch_size,
			        epochs=epochs,
			        validation_split=0.2,
			        shuffle=True)
model.save(sys.argv[1])


