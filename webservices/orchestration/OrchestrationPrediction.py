import logging
import os
import requests

list_algo_deep = ["cnn", "resnet", "alexnet", "vgg"]
list_algo_ml = ["knn", "svm", "gbc", "rfc", "nn"]

class OrchestrationPrediction():
	
	def __init__(self, url_img, list_algo):
		logging.info('orchestration_prediction.init')

		self.url_img = url_img
		self.list_algo = list_algo


	def run(self):

		dir_algo = '../algo_trained/'

		#Liste de String de format JSON : "algo" : {"label":label, "proba":proba}
		#exemple "resnet":{"label":0 ,"proba":0.82}
		list_returns_predictions = []

		for algo in self.list_algo:
			logging.info('orchestration.predict_' + algo)
			URL = "http://127.0.0.1:5007/api/v1/prediction"
			PARAMS = {'directory_from': dir_algo, 'algo': algo, 'directory_img': self.url_img}
			r = requests.get(url = URL, params = PARAMS)
			list_returns_predictions.append(r.text)

		logging.info('orchestration.train_end')
		return list_returns_predictions