from PIL import Image
import numpy as np
import os
import logging

import torch
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset

formatter = logging.Formatter(fmt = '%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)

device = torch.device("cuda:0")

class brain_Dataset(Dataset):
    def __init__ (self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data[1])
        
    def __getitem__(self, index):
        img = self.data[0][index]
        label = self.data[1][index]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

def read_images(directory, img_size):
    logging.info('pytorch_utils.read_images')

    list_img = []
    labels = []

    for name in os.listdir(directory + 'yes'):
        img = Image.open(directory + 'yes/'+ name)
        img = img.resize((img_size, img_size))
        list_img.append(np.asarray(img))
        labels.append(1)
        
    for name in os.listdir(directory + 'no'):
        img = Image.open(directory + 'no/'+ name)
        img = img.resize((img_size, img_size))
        list_img.append(np.asarray(img))
        labels.append(0)

    return list_img, labels

def create_dataloader(directory, img_size, batch_size, transforms = None, validation_split = 0.2):
    logging.info('pytorch_utils.create_dataloader')

    list_img, labels = read_images(directory, img_size)
    dataset = brain_Dataset([list_img, labels], transforms)

    data_size = len(list_img)
    validation_split = validation_split
    split = int(np.floor(validation_split * data_size))
    indices = list(range(data_size))
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                            sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            sampler=val_sampler)

    return train_loader, val_loader

def compute_accuracy(model, val_loader):
    model.eval() # Evaluation mode
    
    correct_samples = 0
    total_samples = 0
    
    for i_step, (x, y) in enumerate(val_loader):
      x_gpu = x.to(device, dtype=torch.float)
      y_gpu = y.to(device, dtype=torch.long)

      predictions = model(x_gpu)
      _, indices = torch.max(predictions, 1)

      correct_samples += torch.sum(indices == y_gpu)
      total_samples += y.shape[0]
      
    accuracy = float(correct_samples)/total_samples       
    
    return accuracy

def train_model(model, train_loader, val_loader, loss, optimizer, num_epochs):
    logging.info('pytorch_utils.train_model')

    train_history = []
    val_history = []
    
    for epoch in range(num_epochs):
        model.train() # Enter train mode        

        correct_samples = 0
        total_samples = 0

        for i_step, (x, y) in enumerate(train_loader):
            x_gpu = x.to(device, dtype=torch.float)
            y_gpu = y.to(device, dtype=torch.long)

            prediction = model(x_gpu)   
            loss_value = loss(prediction, y_gpu)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            
            _, indices = torch.max(prediction, 1)

            correct_samples += torch.sum(indices == y_gpu)
            total_samples += y.shape[0]

        train_accuracy = float(correct_samples) / total_samples
        val_accuracy = compute_accuracy(model, val_loader)

        train_history.append(train_accuracy)
        val_history.append(val_accuracy)
        
        logging.info('Epochs: %d, Train accuracy: %f, Val accuracy: %f', epoch, train_accuracy, val_accuracy)
        
    return train_history, val_history