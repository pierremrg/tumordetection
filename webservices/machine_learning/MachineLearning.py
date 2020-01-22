from PIL import Image
import os
import numpy as np
import time

# Logging
import logging
formatter = logging.Formatter(fmt = '%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


# Dask distributed computing
#import dask.distributed
import joblib
from dask.distributed import Client, wait
import dask

# Scikit-learn machine learning
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn import model_selection, neural_network

from hdfs import InsecureClient
DASK_IP_ADRESS = "192.168.1.4:8786"

class MachineLearning():

	# reads images and stores them
	def __init__(self, input_folder, model_folder, img_size = 240):
		self.input_folder = input_folder
		self.model_folder = model_folder
		self.hdfs_client = InsecureClient('http://192.168.1.4:9870', user='hadoop')
		self.imgs, self.labels = self.read_images(input_folder, 240)
		self.default = "svm"
	
	# reads images from a directory and resizes them
	# returns the list of images and list of labels
	def read_images(self, directory, img_size = 0):
		list_img = []
		labels = []
		logging.info('read_images')
	
		try:
			for name in self.hdfs_client.list('/' + directory + 'yes'):
				if name == "Thumbs.db":
					continue
				with self.hdfs_client.read('/' + directory + 'yes/' + name) as reader:
					img = Image.open(reader)
				if img_size != 0:
					img = img.resize((img_size, img_size))
				img = img.convert('L').convert('RGB')
				list_img.append(np.asarray(img).flatten())
				labels.append(1)
				
			for name in self.hdfs_client.list('/' + directory + 'no'):
				if name == "Thumbs.db":
					continue
				with self.hdfs_client.read('/' + directory + 'no/' + name) as reader:
					img = Image.open(reader)
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
			return  {'n_neighbors': 9}
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

		dask_client = Client(DASK_IP_ADRESS)
		img_train, img_test, lbl_train, lbl_test = train_test_split(self.imgs, self.labels, test_size = 0.2)

		futures_img_train = dask_client.scatter(img_train)
		futures_img_test = dask_client.scatter(img_test)
		futures_lbl_train = dask_client.scatter(lbl_train)
		futures_lbl_test = dask_client.scatter(lbl_test)

		future_model_fit = dask_client.submit(model.fit, futures_img_train, futures_lbl_train)

		model = future_model_fit.result()

		future_score_train = dask_client.submit(model.score, futures_img_train, futures_lbl_train)
		future_score_test = dask_client.submit(model.score, futures_img_test, futures_lbl_test)

		score_test = future_score_test.result()
		score_train = future_score_train.result()		

		logging.info("Training complete, saving model %s to file" %(algorithm) )

		# saving the model to file
		with self.hdfs_client.write('/' + str(self.model_folder) + str(algorithm) + ".model") as writer:
			joblib.dump(model, writer)

		logging.info("Score on training set: %.4f, score on test set: %.4f" %(score_train, score_test) )

		return score_train, score_test
