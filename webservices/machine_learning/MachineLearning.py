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
		
	# # finds the best k-NN configuration for a given dataset
	# # returns the best model and its associated arguments
	# def best_knn(self, imgs, labels):
		# logging.info("Finding best k-NN: This may take a while..")
		# knn = KNeighborsClassifier(n_jobs=-1)
		# grid = {
			# 'k': [i*2+1 for i in range(30)]
		# }
		# gs = GridSearchCV(knn, grid, verbose=2, cv=5, n_jobs=-1)

		# # Dask distributed
		# c = dask.distributed.Client(DASK_IP_ADRESS)
		# with joblib.parallel_backend("dask", scatter=[imgs, labels]):
			# gs.fit(imgs, labels)

		# return gs.best_estimator_, gs.best_params_

	# # trains a k-NearestNeighbors algorithm and saves the best model
	# def knn(self, imgs, labels):
		# model, params = self.best_knn(imgs, labels)
		# joblib.dump(model, self.model_folder + "knn.model")

		# logging.info("Found best k-NN with the following parameters:")
		# logging.info(params)

		# return 0

	# # finds the best SVM configuration for a given dataset
	# # returns the best model and its associated arguments
	# def best_svm(self, imgs, labels):
		# logging.info("Finding best SVM: This may take a while..")
		# svm = SVC(gamma='auto', random_state=0, probability=True)
		# grid = {
			# 'kernel': ['poly', 'linear', 'rbf', 'sigmoid'],
			# 'C': [10**-5, 10**-4, 10**-3, 10**-2],
			# 'shrinking': [True, False]
		# }
		# gs = GridSearchCV(svm, grid, verbose=2, cv=5, n_jobs=-1)

		# # Dask distributed
		# c = dask.distributed.Client(DASK_IP_ADRESS)
		# with joblib.parallel_backend("dask", scatter=[imgs, labels]):
			# gs.fit(imgs, labels)

		# return gs.best_estimator_, gs.best_params_

	# # trains a Support Vector Machine algorithm and saves the best model
	# def svm(self, imgs, labels):
		# model, params = self.best_svm(imgs, labels)
		# joblib.dump(model, self.model_folder + "svm.model")
	
		# logging.info("Found best SVM with the following parameters:")
		# logging.info(params)

		# return 0
		
	# # finds the best GBC configuration for a given dataset
	# # returns the best model and its associated arguments
	# def best_gbc(self, imgs, labels):
		# logging.info("Finding best GBC: This may take a while..")
		# gbc = GradientBoostingClassifier()
		# grid = {
			# "loss":["deviance"],
			# "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
			# "min_samples_split": np.linspace(0.1, 0.5, 12),
			# "min_samples_leaf": np.linspace(0.1, 0.5, 12),
			# "max_depth":[3,5,8],
			# "max_features":["log2","sqrt"],
			# "criterion": ["friedman_mse",  "mae"],
			# "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
			# "n_estimators":[10]
		# }
		# gs = GridSearchCV(gbc, grid, verbose=2, cv=5, n_jobs=-1)

		# # Dask distributed
		# c = dask.distributed.Client(DASK_IP_ADRESS)
		# with joblib.parallel_backend("dask", scatter=[imgs, labels]):
			# gs.fit(imgs, labels)

		# return gs.best_estimator_, gs.best_params_

	# # trains a Gradient Boosting Classifier algorithm and saves the best model
	# def gbc(self, imgs, labels):
		# model, params = self.best_gbc(imgs, labels)
		# joblib.dump(model, self.model_folder + "gbc.model")

		# logging.info("Found best GBC with the following parameters:")
		# logging.info(params)

		# return 0

	# # finds the best RFC configuration for a given dataset
	# # returns the best model and its associated arguments
	# def best_RFC(self, imgs, labels):
		# logging.info("Finding best RFC: This may take a while..")
		# rfc = RandomForestClassifier(n_estimators = 500)
		# grid = {
			# "max_depth": [i for i in range(4, 12)],
			# 'max_features': ['auto', 'sqrt', 'log2'],
			# 'criterion' :['gini', 'entropy']
		# }
		# gs = GridSearchCV(rfc, grid, verbose=2, cv=5, n_jobs=-1)

		# # Dask distributed
		# c = dask.distributed.Client(DASK_IP_ADRESS)
		# with joblib.parallel_backend("dask", scatter=[imgs, labels]):
			# gs.fit(imgs, labels)

		# return gs.best_estimator_, gs.best_params_

	# # trains a Random Forest Classifier algorithm and saves the best model
	# def rfc(self, imgs, labels):
		# model, params = self.best_rfc(imgs, labels)
		# joblib.dump(model, self.model_folder + "rfc.model")
	
		# logging.info("Found best RFC with the following parameters:")
		# logging.info(params)

		# return 0
		
	# # finds the best FC neural network configuration for a given dataset
	# # returns the best model and its associated arguments
	# def best_nn(self, imgs, labels):
		# logging.info("Finding best NN: This may take a while..")
		# nb_nodes = [32, 64, 128, 256] # Number of nodes per hidden layer
		# nb_layers = [2, 5, 8, 12, 20] # Number of hidden layers
		# nn = neural_network.MLPClassifier()
		# grid = {
			# 'hidden_layer_sizes': [tuple([nb_node for i in range(nb_layer)]) for nb_layer in nb_layers for nb_node in nb_nodes]
		# }
		# gs = GridSearchCV(nn, grid, cv=5, n_jobs=-1)

		# # Dask distributed
		# c = dask.distributed.Client(DASK_IP_ADRESS)
		# gs.fit(imgs, labels)

		# # with joblib.parallel_backend("dask", scatter=[imgs, labels]):
			# # gs.fit(imgs, labels)

		# return gs.best_estimator_, gs.best_params_
		
	# # trains a fully connected Neural Network algorithm and saves the best model
	# def nn(self, imgs, labels):
		# model, params = self.best_nn(imgs, labels)
		# joblib.dump(model, self.model_folder + "nn.model")

		# logging.info("Found best NN with the following parameters:")
		# logging.info(params)

		# return 0
	
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

		return score_test