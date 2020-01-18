import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from pytorch_utils import create_dataloader, train_model
from MediumCNN import brainCNN

device = torch.device("cuda:0")

class ClassMediumCNN():
    def __init__(self, images_directory, save_directory, batch_size = 30, epochs = 30):
        logging.info('ClassMediumCNN.init')

        self.images_directory = images_directory
        self.save_directory = save_directory
        self.createOutputDirectory()

        self.batch_size = batch_size
        self.epochs = epochs

        self.model = brainCNN()
        self.model.type(torch.cuda.FloatTensor)
        self.model.to(device) 

    def createOutputDirectory(self):
        try:
            os.mkdir(self.save_directory)
        except:
            logging.info(self.save_directory + ' already exists')

    def initDataLoader(self):
        logging.info('ClassMediumCNN.initDataLoader')

        transform = transform = transforms.Compose([
                           transforms.ToTensor()                         
                       ])
        self.train_loader, self.val_loader = create_dataloader(self.images_directory, 
                                                                img_size = 240, 
                                                                batch_size = self.batch_size, 
                                                                transforms = transform)

    def training(self):
        logging.info('ClassMediumCNN.training')

        loss = nn.CrossEntropyLoss().type(torch.cuda.FloatTensor)
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.train_history, self.val_history = train_model(self.model, 
                                                            self.train_loader, 
                                                            self.val_loader, 
                                                            loss, 
                                                            optimizer, 
                                                            self.epochs)
    def show_accuracy(self):
    	logging.info('ClassMediumCNN.show_accuracy')

    	print('Train accuracy : %f, Val accuracy : %f' % (self.train_history[-1], self.val_history[-1]))
    	return(str(self.train_history[-1]), str(self.val_history[-1]))
    
    def save_model(self):
        logging.info('ClassMediumCNN.save_model')

        torch.save(self.model.state_dict(), self.save_directory + 'Medium_CNN_trained.pt')

    def run(self):
        logging.info('ClassMediumCNN.run')

        self.initDataLoader()
        self.training()
        (train_acc, val_acc) = self.show_accuracy()
        self.save_model()
        return (train_acc, val_acc)
          
