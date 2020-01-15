import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from torchvision import models

from pytorch_utils import create_dataloader, train_model

device = torch.device("cuda:0")

class ModelTransferLearning():
    def __init__(self, images_directory, save_directory, batch_size = 30, network = 'resnet'):
        logging.info('ModelTransferLearning.init')

        self.images_directory = images_directory
        self.save_directory = save_directory
        self.createOutputDirectory()

        self.batch_size = batch_size
        self.network = network

        if (self.network == 'alexnet'):
            self.model = models.alexnet(pretrained=True)
            self.model.classifier[6] = torch.nn.Linear(self.model.classifier[6].in_features, 2)
            self.model = self.model.to(device)
            self.optimizer = optim.SGD(self.model.parameters(), lr=1e-2, momentum = 0.1, weight_decay = 1e-3)
            self.epochs = 10
        elif (self.network == 'vgg'):
            self.model = models.vgg16(pretrained=True)
            num_features = self.model.classifier[6].in_features
            features = list(self.model.classifier.children())[:-1] # Remove last layer
            features.extend([nn.Linear(num_features, 2)]) # Add our layer with 4 outputs
            self.model.classifier = nn.Sequential(*features) # Replace the model classifier
            self.model = self.model.to(device)
            self.optimizer = optim.SGD(self.model.parameters(), lr=1e-2)
            self.epochs = 5
        else:
            self.model = models.resnet18(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, 2)
            self.model = self.model.to(device)
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=1e-3, weight_decay = 1e-3)
            self.epochs = 6

    def createOutputDirectory(self):
        try:
            os.mkdir(self.save_directory)
        except:
            logging.info(self.save_directory + ' already exists')

    def initDataLoader(self):
        logging.info('ModelTransferLearning.initDataLoader')

        transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])                           
                       ])
        self.train_loader, self.val_loader = create_dataloader(self.images_directory, 
                                                                img_size = 240, 
                                                                batch_size = self.batch_size, 
                                                                transforms = transform)

    def training(self):
        logging.info('ModelTransferLearning.training')

        loss = nn.CrossEntropyLoss().type(torch.cuda.FloatTensor)

        self.train_history, self.val_history = train_model(self.model, 
                                                            self.train_loader, 
                                                            self.val_loader, 
                                                            loss, 
                                                            self.optimizer, 
                                                            self.epochs)
    def show_accuracy(self):
        logging.info('ModelTransferLearning.show_accuracy')

        print('Train accuracy : %f, Val accuracy : %f' % (self.train_history[-1], self.val_history[-1]))
    
    def save_model(self):
        logging.info('ModelTransferLearning.save_model')

        torch.save(self.model.state_dict(), self.save_directory + str(self.network) + '_trained.pt')

    def run(self):
        logging.info('ModelTransferLearning.run')

        self.initDataLoader()
        self.training()
        self.show_accuracy()
        self.save_model()  
          