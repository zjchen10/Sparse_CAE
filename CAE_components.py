import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F



class Encoder(nn.Module):
    def __init__(self, in_channel=3, out_dim=3500, activation=nn.GELU()):
        super().__init__()
        # The input size is 3*250*250
        self.in_channel = in_channel
        self.out_dim = out_dim
        self.encoder = nn.Sequential(nn.Conv2d(in_channels=in_channel,out_channels=16,kernel_size=3,stride=1,padding=0,bias=False),
                                nn.BatchNorm2d(16), activation,
                                nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1,bias=False),
                                nn.BatchNorm2d(32), activation,
                                nn.Conv2d(in_channels=32,out_channels=32,kernel_size=2,stride=2,padding=0,bias=False),
                                nn.BatchNorm2d(32), activation,
                                nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False),
                                nn.BatchNorm2d(64), activation,
                                nn.Conv2d(in_channels=64,out_channels=64,kernel_size=2,stride=2,padding=0,bias=False),
                                nn.BatchNorm2d(64), activation,
                                nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1,bias=False),
                                nn.BatchNorm2d(128), activation,
                                nn.Conv2d(in_channels=128,out_channels=128,kernel_size=2,stride=2,padding=0,bias=False),
                                nn.BatchNorm2d(128), activation,
                                nn.Conv2d(in_channels=128,out_channels=320,kernel_size=3,stride=1,padding=1,bias=False),
                                nn.BatchNorm2d(320), activation,
                                nn.Conv2d(in_channels=320,out_channels=240,kernel_size=3,stride=1,padding=1,bias=False),
                                nn.BatchNorm2d(240), activation,
                                nn.Conv2d(in_channels=240,out_channels=150,kernel_size=3,stride=1,padding=1,bias=False),
                                nn.BatchNorm2d(150), activation,
                                nn.Conv2d(in_channels=150,out_channels=80,kernel_size=3,stride=2,padding=1,bias=False),
                                nn.BatchNorm2d(80), activation,
                                nn.Conv2d(in_channels=80,out_channels=40,kernel_size=3,stride=1,padding=1,bias=False),
                                nn.BatchNorm2d(40), activation,
                                nn.Conv2d(in_channels=40,out_channels=15,kernel_size=3,stride=1,padding=1,bias=False),
                                nn.BatchNorm2d(15), activation,
                                nn.Flatten(),
                                nn.Linear(in_features=16*16*15, out_features=self.out_dim),
                                nn.Softmax(dim=1))
        
    def forward(self, x):
        x = x.view(-1, self.in_channel, 250, 250)
        prob = self.encoder(x)
        return prob


class Decoder(nn.Module):
    def __init__(self, in_channel=3, out_dim=1600, activation=nn.GELU()):
        super().__init__()
        self.in_channel = in_channel
        self.out_dim = out_dim

        self.mlp = nn.Sequential(nn.Linear(in_features=self.out_dim, out_features=16*16*15, bias=False),
                                 nn.BatchNorm1d(16*16*15),activation)
        self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=15,out_channels=40,kernel_size=3,stride=1,padding=1,output_padding=0,bias=False),
                                nn.BatchNorm2d(40), activation,
                                     nn.ConvTranspose2d(in_channels=40,out_channels=80,kernel_size=3,stride=1,padding=1,output_padding=0,bias=False),
                                nn.BatchNorm2d(80), activation,
                                nn.ConvTranspose2d(in_channels=80,out_channels=150,kernel_size=3,stride=2,padding=1,output_padding=0,bias=False),
                                nn.BatchNorm2d(150), activation,
                                nn.ConvTranspose2d(in_channels=150,out_channels=240,kernel_size=3,stride=1,padding=1,output_padding=0,bias=False),
                                nn.BatchNorm2d(240), activation,
                                nn.ConvTranspose2d(in_channels=240,out_channels=320,kernel_size=3,stride=1,padding=1,output_padding=0,bias=False),
                                nn.BatchNorm2d(320), activation,
                                nn.ConvTranspose2d(in_channels=320,out_channels=128,kernel_size=3,stride=1,padding=1,output_padding=0,bias=False),
                                nn.BatchNorm2d(128), activation,
                                nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=3,stride=2,padding=1,output_padding=1,bias=False),
                                nn.BatchNorm2d(128), activation,
                                nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1,output_padding=0,bias=False),
                                nn.BatchNorm2d(64), activation,
                                nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=3,stride=2,padding=1,output_padding=1,bias=False),
                                nn.BatchNorm2d(64), activation,
                                nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1,output_padding=0,bias=False),
                                nn.BatchNorm2d(32), activation,
                                nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=3,stride=2,padding=1,output_padding=1,bias=False),
                                nn.BatchNorm2d(32), activation,
                                nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=3,stride=1,padding=1,output_padding=0,bias=False),
                                nn.BatchNorm2d(16), activation,
                                nn.ConvTranspose2d(in_channels=16, out_channels=in_channel, kernel_size=(3,3), stride=1, padding=0, output_padding=0),
                                nn.Sigmoid())
    
    def forward(self, x):
        x = x.view(-1, self.out_dim)
        temp = self.mlp(x)
        temp = temp.view(-1,15,16,16)
        output = self.decoder(temp)
        return output
