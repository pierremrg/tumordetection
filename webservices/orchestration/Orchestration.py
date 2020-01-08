import logging
import os
import requests

logger = logging.getLogger()
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)

class Orchestration():
	
	def __init__(self, directory_to, max_augmentation, coef_rotation = 0.7, algo, batch_size, epochs):
		logging.info('orchestration.init')
        
        self.directory_to = directory_to
        self.MAX_AUGMENTATION = max_augmentation
        self.COEF_ROTATION = coef_rotation
        self.algo = algo
        self.batch_size = batch_size
        self.epochs = epochs


    def run():

    	logging.info('orchestration.getPictures')
		#URL service getPicture 
		URL = "http://127.0.0.1:5000/api/v1/getPictures"
		PARAMS = {'directory_to' = self.directory_to}
		r = requests.get(url = URL, params = PARAMS)

		logging.info('orchestration.dataAugment')
		#URL service dataAugment
		URL = "http://127.0.0.1:5001/api/v1/data_augment"
		dir_from = self.directory_to + "data/"
		dir_to = self.directory_to + "data_augmented/"
		PARAMS = {'directory_from':dir_from, 'max_augmentation':self.MAX_AUGMENTATION, 'coef_rotation': self.COEF_ROTATION, 'directory_to': dir_to} 
		r = requests.post(url = URL, params = PARAMS) 

		logging.info('orchestration.crop')
		#URL service crop 
		URL = "http://127.0.0.1:5002/api/v1/crop"
		dir_from = self.directory_to + "data_augmented/"
		dir_to = self.directory_to + "data_augmented_crop/"
		PARAMS = {'from':dir_from, 'to': dir_to} 
		r = requests.get(url = URL, params = PARAMS)

		logging.info('orchestration.normalize')
		#URL service normalization 
		URL = "http://127.0.0.1:5003/api/v1/normalize"
		dir_from = self.directory_to + "data_augmented_crop/"
		dir_to = self.directory_to + "data_final/"
		PARAMS = {'from':dir_from, 'to': dir_to} 
		r = requests.post(url = URL, params = PARAMS) 

		logging.info('orchestration.train_' + algo)
		dir_from = self.directory_to + "data_final/"
		dir_to = self.directory_to + "../model_trained/"
		if self.algo == "cnn":
			#URL service mediumCNN 
			URL = "http://127.0.0.1:5004/api/v1/medium_cnn"
			PARAMS = {'images_directory': dir_from, 'save_directory': dir_to, 'batch_size': self.batch_size, 'epochs': self.epochs}
			r = requests.post(url = URL, params = PARAMS) 
		elif self.algo == "alexnet" || self.algo == "resnet" || self.algo == "vgg":
			#URL service transferlearning 
			URL = "http://127.0.0.1:5005/api/v1/transferlearning"
			PARAMS = {'images_directory': dir_from, 'save_directory': dir_to, 'batch_size': self.batch_size, 'network': self.algo}
			r = requests.get(url = URL, params = PARAMS)

		logging.info('orchestration.train_end')