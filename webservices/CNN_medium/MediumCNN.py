import logging

import torch.nn as nn

class brainCNN(nn.Module):
    def __init__(self):
        super(brainCNN, self).__init__()
        
        logging.info('brainCNN.init')
        
        self.zp = nn.ConstantPad2d(2, 0)
        
        self.conv = nn.Conv2d(3, 32, kernel_size=(7,7), stride=(1, 1))
        nn.init.xavier_uniform_(self.conv.weight)

        self.seq = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU()
            )
        
        self.mp1 = nn.MaxPool2d((4,4))
        self.mp2 = nn.MaxPool2d((4,4))
        self.ft = nn.Flatten()
        
        self.lin = nn.Linear(6272, 2)
        nn.init.xavier_uniform_(self.lin.weight)
    
    def forward(self, x):
        x = self.zp(x)
        x = self.conv(x)
        x = self.seq(x)
        x = self.mp1(x)
        x = self.mp2(x)
        x = self.ft(x)
        x = self.lin(x)
        
        return x
        