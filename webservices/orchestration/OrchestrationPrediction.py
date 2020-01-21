import logging
import os
import requests
from PIL import Image  
import PIL
import csv

list_algo_deep = ["cnn", "resnet", "alexnet", "vgg"]
list_algo_ml = ["knn", "svm", "gbc", "rfc", "nn"]

class OrchestrationPrediction():
	
	def __init__(self, url_img, list_algo):
		logging.info('orchestration_prediction.init')

		self.url_img = url_img
		self.list_algo = list_algo

	def run(self):

		dir_algo = '../algo_trained/'
		dir_img_test = '../img_test/'
		dir_img_test_crop = '../img_test_crop/'
		dir_img_test_ready = '../img_test_ready/'

		try:
            os.mkdir(dir_img_test)
        except:
            logging.info(dir_img_test + " already exists")

		im1 = Image.open(self.url_img)  
  		im1 = im1.save(dir_img_test + 'img_test.jpg')

		with open(dir_img_test + 'img_test.csv', 'w', newline='') as file:
		    writer = csv.writer(file)
		    writer.writerow(["Path"])
		    writer.writerow(['img_test.jpg'])

  		logging.info('orchestration_prediction.crop')
		URL = "http://127.0.0.1:5004/api/v1/crop"
		PARAMS = {'from':dir_img_test, 'to': dir_img_test_crop} 
		r = requests.get(url = URL, params = PARAMS)

		logging.info('orchestration_prediction.normalize')
		URL = "http://127.0.0.1:5003/api/v1/normalize"
		PARAMS = {'from':dir_img_test_crop, 'to': dir_img_test_ready} 
		r = requests.post(url = URL, params = PARAMS) 

		#Liste de String de format JSON : "algo" : {"label":label, "proba":proba}
		#exemple "resnet":{"label":0 ,"proba":0.82}
		list_returns_predictions = []

		for algo in self.list_algo:
			logging.info('orchestration.predict_' + algo)
			URL = "http://127.0.0.1:5007/api/v1/prediction"
			PARAMS = {'directory_from': dir_algo, 'algo': algo, 'directory_img': dir_img_test_ready}
			r = requests.get(url = URL, params = PARAMS)
			list_returns_predictions.append(r.text)

		logging.info('orchestration.train_end')
		return list_returns_predictions