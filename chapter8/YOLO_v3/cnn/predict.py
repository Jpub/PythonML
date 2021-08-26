import numpy as np
import os

import pandas as pd
from skimage import io, color, exposure, transform
from sklearn.model_selection import train_test_split

from util import create_model, preprocess_image, create_resnet50

NUM_CLASSES = 43
IMG_SIZE = 48

DATA_PATH = "./GTSRB/Final_Test/Images/"

def get_test_data(csv_path, data_path):
	test = pd.read_csv(csv_path, sep=';')
	X_test = []
	y_test = []

	i=0
	for file_name, class_id in zip(list(test['Filename']),list(test['ClassId'])):
	    img_path = os.path.join(data_path,file_name)
	    X_test.append(preprocess_image(io.imread(img_path), IMG_SIZE))
	    y_test.append(class_id)
	    i = i+1
	    print('loaded image {}'.format(i))
	    
	X_test = np.array(X_test)
	y_test = np.array(y_test)

	return X_test, y_test


print('start')
# model = create_model(NUM_CLASSES, IMG_SIZE)
model = create_resnet50(NUM_CLASSES, IMG_SIZE)
# model.load_weights('gtsrb_cnn_1.h5')
weight_file = 'gtsrb_cnn_augmentation.h5'
weight_file = 'gtsrb_resnet.h5'
model.load_weights(weight_file)

test_x, test_y = get_test_data('./GTSRB/GT-final_test.csv', DATA_PATH)

correct_ans = 0.0
for i in range(len(test_x)):
	x = test_x[i]
	y = test_y[i]
	y_pred =  np.argmax(model.predict([[x]]))
	if y_pred == y:
		correct_ans = correct_ans + 1.0

#y_pred = model.predict_classes(test_x)
#acc = np.sum(y_pred==test_y)/np.size(y_pred)
acc = correct_ans / float(len(test_y))
print("Test accuracy = {}".format(acc))

