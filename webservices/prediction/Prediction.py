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

    def __init__(self, dir_from, algo, dir_img, nomImage):
        logging.info('prediction.init')
        self.directory_from = dir_from
        self.algo = algo
        if self.algo == 'cnn':
            self.file = 'Medium_CNN_trained.pt'
        else:
            self.file = self.algo + "_trained.pt"
        self.dir_img = dir_img
        self.nomImage = nomImage

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
        model = model.to(device)
        logging.info('prediction.loadModel')
        model.load_state_dict(torch.load(self.directory_from+self.file))
        img = Image.open(self.dir_img + self.nomImage)
        logging.info('prediction.testModel')
        label, proba = test_model(model, np.asarray(img))
        logging.info('Label found : ' + str(label) + ' - Probability : ' + str(proba))
        return label, proba