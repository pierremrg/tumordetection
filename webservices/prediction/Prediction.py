import logging
import os
import requests
import torch
import numpy as np 
from PIL import Image
import torchvision.models as models
import torch.nn as nn

from pytorch_utils import test_model
from MediumCNN import brainCNN


device = torch.device("cuda:0")

class Prediction():

	def __init__(self, dir_from, algo, dir_img):
		logging.info('prediction.init')
		self.directory_from = dir_from
		self.algo = algo
		self.dir_img = dir_img	
		self.image = self.read_image(self.dir_img, 240)
		
		if self.algo == 'cnn':
			self.file = 'Medium_CNN_trained.pt'
		else:
			self.file = self.algo + "_trained.pt"


	def read_image(self, path, img_size = 0):
		logging.info('prediction_ML.read_image')
		img = 0
	
		try:
			img = Image.open(path)
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
		if self.algo == 'cnn':
			model = brainCNN()
		elif self.algo == 'resnet':
			model = models.resnet18()
			num_ftrs = model.fc.in_features
			model.fc = nn.Linear(num_ftrs, 2)
		elif self.algo == 'alexnet':
			model = models.alexnet()
			model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)
		elif self.algo == 'vgg':
			model = models.vgg16()
			num_features = model.classifier[6].in_features
			features = list(model.classifier.children())[:-1]
			features.extend([nn.Linear(num_features, 2)])
			model.classifier = nn.Sequential(*features)
			
		try:
			model = model.to(device)
			logging.info('prediction.loadModel')
			model.load_state_dict(torch.load(self.directory_from+self.file))
			logging.info('prediction.testModel')
		except IOError as err:
			logging.error('Error model '+str(self.algo)+' is not trained yet!')
			logging.error('Train this model first before using it for predictions')
			return -1, 1

		label, proba = test_model(model, self.image)
		logging.info('Label found : ' + str(label) + ' - Probability : ' + str(proba))
		return label, proba