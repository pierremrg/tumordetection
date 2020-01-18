import logging
import os
import requests

list_algo_deep = ["cnn", "resnet", "alexnet", "vgg"]
list_algo_ml = ["knn", "svm", "gbc", "rfc", "nn"]

class Orchestration():
	
	def __init__(self, url, list_algo):
		logging.info('orchestration_training.init')

		self.url = url
		self.list_algo = list_algo


	def run(self):

		dir_img = '../images/'
		dir_img_augm = '../images_augmented/'
		dir_img_augm_crop = '../images_augmented_crop/'
		dir_img_norm = '../new_data_norm/'
		dir_algo = '../algo_trained/'

		#Liste de String de format JSON : "algo" : {"train_acc":train_acc, "val_acc":val_acc}
		#exemple "resnet":{"train_acc":0.85 ,"val_acc":0.82}
		list_returns_trains = []

		logging.info('orchestration.getPictures')
		#URL service getPicture 
		URL = "http://127.0.0.1:5001/api/v1/getPictures"
		PARAMS = {'directory_to':dir_img, 'url':self.url}
		r = requests.post(url = URL, params = PARAMS)

		logging.info('orchestration.dataAugment')
		#URL service dataAugment
		URL = "http://127.0.0.1:5002/api/v1/data_augment"
		PARAMS = {'directory_from':dir_img, 'max_augmentation':'1000', 'coef_rotation':'0.7', 'directory_to': dir_img_augm} 
		r = requests.post(url = URL, params = PARAMS) 

		logging.info('orchestration.crop')
		#URL service crop 
		URL = "http://127.0.0.1:5004/api/v1/crop"
		PARAMS = {'from':dir_img_augm, 'to': dir_img_augm_crop} 
		r = requests.get(url = URL, params = PARAMS)

		logging.info('orchestration.normalize')
		#URL service normalization 
		URL = "http://127.0.0.1:5003/api/v1/normalize"
		PARAMS = {'from':dir_img_augm_crop, 'to': dir_img_norm} 
		r = requests.post(url = URL, params = PARAMS) 

		for algo in self.list_algo:
			if algo in list_algo_deep:
				logging.info('orchestration.train_' + algo)
				if algo == "cnn":
					URL = "http://127.0.0.1:5005/api/v1/medium_cnn"
					PARAMS = {'images_directory': dir_img_norm, 'save_directory': dir_algo}
					r = requests.post(url = URL, params = PARAMS)
					list_returns_trains.append(r.text)
				elif algo == "alexnet" or algo == "resnet" or algo == "vgg":
					URL = "http://127.0.0.1:5006/api/v1/transferlearning"
					PARAMS = {'images_directory': dir_img_norm, 'save_directory': dir_algo, 'network': algo}
					r = requests.get(url = URL, params = PARAMS)
					list_returns_trains.append(r.text)
			elif algo in list_algo_ml:
				logging.info('orchestration.train_' + algo)
				URL = "http://127.0.0.1:5007/ai/v1/ml"
				PARAMS = {'images_directory': dir_img_norm, 'save_directory': dir_algo, 'algorithm': algo}
				r = requests.get(url = URL, params = PARAMS)
				list_returns_trains.append(r.text)
			else:
				logging.info("incorrect algo provided")

		logging.info('orchestration.train_end')
		return list_returns_trains