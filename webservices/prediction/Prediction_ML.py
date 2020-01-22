import logging
import os
import numpy as np 
from PIL import Image
import joblib

from hdfs import InsecureClient

class Prediction_ML():

	def __init__(self, dir_algo, algo, path_img):
		logging.info('prediction_ML.init')
		self.directory_algo = dir_algo
		self.path_img = path_img
		self.algo = algo

		self.hdfs_client = InsecureClient('http://192.168.1.4:9870', user='hadoop')
		self.image = self.read_image(self.path_img, 240)

	def read_image(self, path_img, img_size = 0):
		logging.info('prediction_ML.read_image')
		img = 0
	
		try:
			with self.hdfs_client.read(path_img) as reader:
				img = Image.open(reader)
			if img_size != 0:
				img = img.resize((img_size, img_size))
			img = img.convert('L').convert('RGB')
			img = np.asarray(img).flatten()
				
		except IOError as err:
			logging.error("Error reading image or path")
			logging.error(err)

		except Exception as err:
			logging.error("Unkownown error in read_image")
			logging.error(err)

		return img

	def run(self):
		try:
			with self.hdfs_client(self.directory_from + self.algo + ".model") as reader:
				model = joblib.load(reader)

			label = model.predict([self.image])
			try:
				array_proba = model.predict_proba([self.image])[0]
				proba = array_proba[label[0]]
			except:
				proba = -1
				
			return label[0], proba
			
		except IOError as err:
			logging.error('Error model '+str(self.algo)+' is not trained yet!')
			logging.error('Train this model first before using it for predictions')
			return -1, 1
