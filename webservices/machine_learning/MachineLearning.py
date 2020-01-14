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
	def __init__(self, input_folder, img_folder, model_folder):
		self.input_folder = input_folder
		self.model_folder = model_folder
		self.imgs, self.labels = self.read_images(input_folder, 240)
		self.img = self.read_image(img_folder, 240)
		
	# def read_image(self, path, img_size = 0):
		# logging.info('read_image')
		# img = 0
	
		# try:
			# img = Image.open(path)
			# if img_size != 0:
				# img = img.resize((img_size, img_size))
			# img = img.convert('L').convert('RGB')
			# img = np.asarray(img).flatten()
				
		# except IOError as err:
			# logging.error("Error reading image or path")
			# logging.error(err)

		# except Exception as err:
			# logging.error("Unkownown error in read_image")
			# logging.error(err)

		# return [img]
	
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

		return list_img, labels
		
	# finds the best k-NN configuration for a given dataset
	# returns the best score and its associated arguments
	def best_knn(self, imgs, labels):
		logging.info("Finding best k-NN: This may take a while..")
		knn = KNeighborsClassifier(n_jobs=-1)
		grid = {
			'k': [i*2+1 for i in range(30)]
		}
		gs = GridSearchCV(knn, grid, verbose=2, cv=5, n_jobs=-1)

		# Dask distributed
		with joblib.parallel_backend("dask", scatter=[imgs, labels]):
			gs.fit(imgs, labels)

		return gs.best_estimator_, gs.best_params_

	# trains a k-NearestNeighbors algorithm and returns a prediction
	# if fast_train is disabled: best k-NN is used to find the best parameters
	# if fast_train is enabled: previously determined parameters are used
	def knn(self, img, imgs, labels, fast_train):
		model, params = best_knn(imgs, labels)
		joblib.dump(model, self.model_folder + "knn.model")

		logging.info("Found best k-NN with the following parameters:")
		logging.info(params)

		return 0

	# finds the best SVM configuration for a given dataset
	# returns the best score and its associated arguments
	def best_SVM(self, imgs, labels):
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

		return gs.best_estimator_, gs.best_params_

	# trains a Support Vector Machine algorithm and returns a prediction
	# if fast_train is disabled: best SVM is used to find the best parameters
	# if fast_train is enabled: previously determined parameters are used
	def svm(self, img, imgs, labels, fast_train):
		model, params = best_svm(imgs, labels)
		joblib.dump(model, self.model_folder + "svm.model")
	
		logging.info("Found best SVM with the following parameters:")
		logging.info(params)

		return 0
		
	# finds the best GBC configuration for a given dataset
	# returns the best score and its associated arguments
	def best_GBC(self, imgs, labels):
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

		return gs.best_estimator_, gs.best_params_

	# trains a Gradient Boosting Classifier algorithm and returns a prediction
	# if fast_train is disabled: best GBC is used to find the best parameters
	# if fast_train is enabled: previously determined parameters are used
	def gbc(self, img, imgs, labels, fast_train):
		model, params = best_gbc(imgs, labels)
		joblib.dump(model, self.model_folder + "gbc.model")

		logging.info("Found best GBC with the following parameters:")
		logging.info(params)

		return 0

	# finds the best RFC configuration for a given dataset
	# returns the best score and its associated arguments
	def best_RFC(self, imgs, labels):
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

		return gs.best_estimator_, gs.best_params_

	# trains a Random Forest Classifier algorithm and returns a prediction
	# if fast_train is disabled: best RFC is used to find the best parameters
	# if fast_train is enabled: previously determined parameters are used
	def rfc(self, img, imgs, labels, fast_train):
		model, params = best_rfc(imgs, labels)
		joblib.dump(model, self.model_folder + "rfc.model")
	
		logging.info("Found best RFC with the following parameters:")
		logging.info(params)

		return 0
		
	# finds the best FC neural network configuration for a given dataset
	# returns the best score and its associated arguments
	def best_NN(self, imgs, labels):
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

		return gs.best_estimator_, gs.best_params_
		
	# trains a fully connected Neural Network algorithm and returns a prediction
	# if fast_train is disabled: best NN is used to find the best parameters
	# if fast_train is enabled: previously determined parameters are used
	def nn(self, img, imgs, labels, fast_train):
		model, params = best_nn(imgs, labels)
		joblib.dump(model, self.model_folder + "nn.model")

		logging.info("Found best NN with the following parameters:")
		logging.info(params)

		return 0