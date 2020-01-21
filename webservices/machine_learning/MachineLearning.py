from PIL import Image
import os
import numpy as np

# Logging
import logging
formatter = logging.Formatter(fmt = '%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)


# Dask distributed computing
#import dask.distributed
import joblib

# Scikit-learn machine learning
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn import model_selection, neural_network
from dask_ml.model_selection import GridSearchCV

#DASK_IP_ADRESS = "127.0.0.1:59085"

class MachineLearning():

	# reads images and stores them
	def __init__(self, input_folder, model_folder):
		self.input_folder = input_folder
		self.model_folder = model_folder
		self.imgs, self.labels = self.read_images(input_folder, 240)
		self.default = "svm"
	
	# reads images from a directory and resizes them
	# returns the list of images and list of labels
	def read_images(self, directory, img_size = 0):
		list_img = []
		labels = []
		logging.info('read_images')
	
		try:
			for name in os.listdir(directory + 'yes'):
				if name == "Thumbs.db":
					continue
				img = Image.open(directory + 'yes/'+ name)
				if img_size != 0:
					img = img.resize((img_size, img_size))
				img = img.convert('L').convert('RGB')
				list_img.append(np.asarray(img).flatten())
				labels.append(1)
				
			for name in os.listdir(directory + 'no'):
				if name == "Thumbs.db":
					continue
				img = Image.open(directory + 'no/'+ name)
				if img_size != 0:
					img = img.resize((img_size, img_size))
				img = img.convert('L').convert('RGB')
				list_img.append(np.asarray(img).flatten())
				labels.append(0)
				
		except Exception as err:
			logging.error("Error in read_images")
			logging.error(err)
			list_img = []
			labels = []

		logging.info("Finished reading images")

		return list_img, labels
		
	# returns the untrained model for a given algorithm
	def get_model(self, algorithm, params):
		if (algorithm == "knn"):
			return KNeighborsClassifier(**params, n_jobs=-1)
		elif (algorithm == "svm"):
			return SVC(**params, gamma='auto', random_state=0, probability=True)
		elif (algorithm == "gbc"):
			return GradientBoostingClassifier(**params)
		elif (algorithm == "rfc"):
			return RandomForestClassifier(**params, n_estimators = 500)
		elif (algorithm == "nn"):
			return neural_network.MLPClassifier(**params)
		else:
			return self.get_model(self.default, params)

	# returns a set of the "best" parameters for a given algorithm
	def get_params(self, algorithm):
		if (algorithm == "knn"):
			return  {'n_neighbors': '3'}
		elif (algorithm == "svm"):
			return  {
						'kernel': 'poly',
						'C': 10**-4
					}
		elif (algorithm == "gbc"):
			return  {
						'n_estimators': 10
					}
		elif (algorithm == "rfc"):
			return  {
						'max_depth': 8,
						'max_features': "auto",
						'criterion': "gini"
					}
		elif (algorithm == "nn"):
			return  {
						'hidden_layer_sizes': tuple([64 for _ in range(10)])
					}
		else:
			return self.get_params(self.default)
		
	# trains a model using the best parameters and returns the score
	def train(self, algorithm, imgs, labels):
		params = self.get_params(algorithm)
		model = self.get_model(algorithm, params)
		logging.info("Training %s with the following parameters:" %(algorithm) )
		logging.info(params)
		img_train, img_test, lbl_train, lbl_test = train_test_split(self.imgs, self.labels, test_size = 0.2)

		model.fit(img_train, lbl_train)
		score_train = model.score(img_train, lbl_train)
		score_test = model.score(img_test, lbl_test)
		logging.info("Training complete, saving model %s to file" %(algorithm) )
		
		# saving the model to file
		joblib.dump(model, str(self.model_folder) + str(algorithm) + ".model")
		
		logging.info("Score on training set: %.4f, score on test set: %.4f" %(score_train, score_test) )

		return (score_train, score_test)