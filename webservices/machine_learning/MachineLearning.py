from PIL import Image
import os
import numpy as np
import logging

# Dask distributed computing
import joblib
import dask.distributed

# Scikit-learn machine learning
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn import model_selection, neural_network
from sklearn.model_selection import GridSearchCV


class MachineLearning():

	# reads images and stores them
	def __init__(self, input_folder):
		self.input_folder = input_folder
		self.imgs, self.labels = read_images(self.input_folder)
		
	# reads images from a directory and resizes them
	# returns the list of images and list of labels
	def read_images(directory, img_size = 0):
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
				if directory == DIRECTORY_RAW:
					img = img.convert('L').convert('RGB')
				list_img.append(np.asarray(img).flatten())
				labels.append(1)
				
			for name in os.listdir(directory + 'no'):
				if name == "Thumbs.db":
					continue
				img = Image.open(directory + 'no/'+ name)
				if img_size != 0:
					img = img.resize((img_size, img_size))
				if directory == DIRECTORY_RAW:
					img = img.convert('L').convert('RGB')
				list_img.append(np.asarray(img).flatten())
				labels.append(0)
				
		except Exception as e:
			logging.error("Error in read_images")
			logging.error(err)
			list_img = []
			labels = []

		return list_img, labels
		
	# finds the best k-NN configuration for a given dataset
	# returns the best score and its associated arguments
	def best_knn(imgs, labels):
		logging.info("Finding best k-NN: This may take a while..")
		knn = KNeighborsClassifier(n_jobs=-1)
		grid = {
			'k': [i*2+1 for i in range(30)]
		}
		gs = GridSearchCV(knn, grid, verbose=2, cv=5, n_jobs=-1)

		# Dask distributed
		with joblib.parallel_backend("dask", scatter=[imgs, labels]):
			gs.fit(imgs, labels)

		return gs.best_score_, gs.best_params_

	# trains a k-NearestNeighbors algorithm and returns a prediction
	# if fast_train is disabled: best k-NN is used to find the best parameters
	# if fast_train is enabled: previously determined parameters are used
	def knn(img, imgs, labels, fast_train):
		if(!fast_train):
			_, params = best_knn(imgs, labels)
		else:
			params = {'k': '3'}
	
		logging.info("Using k-NN with the following parameters:")
		logging.info(params)
		model = KNeighborsClassifier(**params, n_jobs=-1)
		model.fit(imgs, labels)

		return model.predict(img)

	# finds the best SVM configuration for a given dataset
	# returns the best score and its associated arguments
	def best_SVM(imgs, labels):
		svm = SVC(gamma='auto', random_state=0, probability=True)
		grid = {
			'kernel': ['poly', 'linear', 'rbf', 'sigmoid'],
			'C': [10**-5, 10**-4, 10**-3, 10**-2],
			'shrinking': [True, False]
		}
		gs = GridSearchCV(svm, grid, verbose=2, cv=5, n_jobs=-1)

		# Dask distributed
		c = dask.distributed.Client()
		with joblib.parallel_backend("dask", scatter=[imgs, labels]):
			gs.fit(imgs, labels)

		return gs.best_score_, gs.best_params_

	# trains a Support Vector Machine algorithm and returns a prediction
	# if fast_train is disabled: best SVM is used to find the best parameters
	# if fast_train is enabled: previously determined parameters are used
	def svm(img, imgs, labels, fast_train):
		if(!fast_train):
			_, params = best_SVM(imgs, labels)
		else:
			params = {
				'kernel': 'poly',
				'C': 10**-4
			}
	
		logging.info("Using SVM with the following parameters:")
		logging.info(params)
		model = = SVC(**params, gamma='auto', random_state=0, probability=True)
		model.fit(imgs, labels)

		return model.predict(img)
		
	# finds the best GBC configuration for a given dataset
	# returns the best score and its associated arguments
	def best_GBC(imgs, labels):
		gbc = GradientBoostingClassifier()
		grid = {
			"loss":["deviance"],
			"learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
			"min_samples_split": np.linspace(0.1, 0.5, 12),
			"min_samples_leaf": np.linspace(0.1, 0.5, 12),
			"max_depth":[3,5,8],
			"max_features":["log2","sqrt"],
			"criterion": ["friedman_mse",  "mae"],
			"subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
			"n_estimators":[10]
		}
		gs = GridSearchCV(gbc, grid, verbose=2, cv=5, n_jobs=-1)

		# Dask distributed
		c = dask.distributed.Client()
		with joblib.parallel_backend("dask", scatter=[imgs, labels]):
			gs.fit(imgs, labels)

		return gs.best_score_, gs.best_params_

	# trains a Gradient Boosting Classifier algorithm and returns a prediction
	# if fast_train is disabled: best GBC is used to find the best parameters
	# if fast_train is enabled: previously determined parameters are used
	def gbc(img, imgs, labels, fast_train):
		if(!fast_train):
			_, params = best_GBC(imgs, labels)
		else:
			params = {
				'n_estimators': 10
			}
	
		logging.info("Using GBC with the following parameters:")
		logging.info(params)
		model = GradientBoostingClassifier(**params)
		model.fit(imgs, labels)

		return model.predict(img)

	# finds the best RFC configuration for a given dataset
	# returns the best score and its associated arguments
	def best_RFC(imgs, labels):
		rfc = RandomForestClassifier(n_estimators = 500)
		grid = {
			"max_depth": [i for i in range(4, 12)],
			'max_features': ['auto', 'sqrt', 'log2'],
			'criterion' :['gini', 'entropy']
		}
		gs = GridSearchCV(rfc, grid, verbose=2, cv=5, n_jobs=-1)

		# Dask distributed
		c = dask.distributed.Client()
		with joblib.parallel_backend("dask", scatter=[imgs, labels]):
			gs.fit(imgs, labels)

		return gs.best_score_, gs.best_params_

	# trains a Random Forest Classifier algorithm and returns a prediction
	# if fast_train is disabled: best RFC is used to find the best parameters
	# if fast_train is enabled: previously determined parameters are used
	def rfc(img, imgs, labels, fast_train):
		if(!fast_train):
			_, params = best_RFC(imgs, labels)
		else:
			params = {
				'max_depth': 8,
				'max_features': "auto",
				'criterion': "gini"
			}
	
		logging.info("Using RFC with the following parameters:")
		logging.info(params)
		model = RandomForestClassifier(**params, n_estimators = 500)
		model.fit(imgs, labels)

		return model.predict(img)
		
	# finds the best FC neural network configuration for a given dataset
	# returns the best score and its associated arguments
	def best_NN(imgs, labels):
		nb_nodes = [32, 64, 128, 256] # Number of nodes per hidden layer
		nb_layers = [2,5,8,12,20] # Number of hidden layers
		nn = neural_network.MLPClassifier()
		grid = {
			'hidden_layer_sizes': tuple([nb_node for i in range(nb_layer) for nb_layer in nb_layers for nb_node in nb_nodes])
		}
		gs = GridSearchCV(nn, grid, verbose=2, cv=5, n_jobs=-1)

		# Dask distributed
		c = dask.distributed.Client()
		with joblib.parallel_backend("dask", scatter=[imgs, labels]):
			gs.fit(imgs, labels)

		return gs.best_score_, gs.best_params_
		
	# trains a fully connected Neural Network algorithm and returns a prediction
	# if fast_train is disabled: best NN is used to find the best parameters
	# if fast_train is enabled: previously determined parameters are used
	def nn(img, imgs, labels, fast_train):
		if(!fast_train):
			_, params = best_NN(imgs, labels)
		else:
			params = {
				'criterion': tuple([64 for _ in range(10)])
			}
	
		logging.info("Using NN with the following parameters:")
		logging.info(params)
		model = neural_network.MLPClassifier(**params)
		model.fit(imgs, labels)

		return model.predict(img)