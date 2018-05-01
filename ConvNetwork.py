import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from math import floor
import torchvision
import torchvision.transforms as transforms
import torch.utils
import torch.optim as optim
import matplotlib.pyplot as plt
import time

    
def fit_net(net, optimizer, criterion, trainloader, num_epoch):
    running_loss = 0
    last_loss = 0
    for epoch in range(num_epoch): 
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
        if epoch % 5 == 4:
            print('[epoch %d] loss: %.3f' % 
                  (epoch + 1, running_loss / (5 * len(trainloader))))
            last_loss = running_loss / (5 * len(trainloader))
            running_loss = 0.0
    return last_loss


def check_accuracy(net, testloader):
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy of the network: %.2f %%' % (100 * correct / total))
    return 100 * correct / total


def batch(dataset, batch_size, index):
    return torch.stack([dataset[j][0] for j in range(index, index + batch_size)])


def fit_autoencoder(net, optimizer, criterion, unlabeledset, num_epoch, batch_size):
    for epoch in range(num_epoch): 
        running_loss = 0.0
        last_loss = 0.0
        for i in range(0, len(unlabeledset), batch_size):
            inputs = batch(unlabeledset, batch_size, i)
            inputs = Variable(inputs)
            optimizer.zero_grad()
            
            outputs = net(inputs, autoencoder=True)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            if (i / batch_size) % 5000 == 4999:    
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i / batch_size + 1, running_loss / 5000))
                last_loss = running_loss / 5000
                running_loss = 0.0
    return last_loss


class ConvNetAutoEncoder(nn.Module):
    def __init__(self, input_size=(3, 32, 32), layers_num=2, conv_out_channels=2, conv_kernel_size=4, 
                 conv_stride=2, conv_padding=0, pool_kernel_size=2, pool_stride=1, pool_padding=0):
        super(ConvNetAutoEncoder, self).__init__()
        if layers_num > 0:
            if type(conv_out_channels) == int:
                conv_out_channels = (conv_out_channels,) * layers_num
            if type(conv_kernel_size) == int:
                conv_kernel_size = (conv_kernel_size,) * layers_num
            if type(conv_stride) == int:
                conv_stride = (conv_stride,) * layers_num
            if type(conv_padding) == int:
                conv_padding = (conv_padding,) * layers_num    
            if type(pool_kernel_size) == int:
                pool_kernel_size = (pool_kernel_size,) * layers_num
            if type(pool_stride) == int:
                pool_stride = (pool_stride,) * layers_num
            if type(pool_padding) == int:
                pool_padding = (pool_padding,) * layers_num
                
            self.conv = nn.ModuleList([nn.Conv2d(input_size[0], conv_out_channels[0], conv_kernel_size[0], 
                                                 stride=conv_stride[0], padding=conv_padding[0])])
            for i in range(layers_num - 1):
                self.conv.append(nn.Conv2d(conv_out_channels[i], conv_out_channels[i + 1], conv_kernel_size[i + 1],
                                           stride=conv_stride[i + 1], padding=conv_padding[i + 1]))
            self.pool = nn.ModuleList([nn.MaxPool2d(pool_kernel_size[i], stride=pool_stride[i], 
                                                    padding=pool_padding[i], return_indices=True) 
                                       for i in range(layers_num)])
            self.unpool = nn.ModuleList([nn.MaxUnpool2d(pool_kernel_size[i], stride=pool_stride[i], 
                                                        padding=pool_padding[i]) 
                                         for i in range(layers_num)[::-1]])
            self.conv_transposed = nn.ModuleList([nn.ConvTranspose2d(conv_out_channels[i], conv_out_channels[i - 1], 
                                                                     conv_kernel_size[i], stride=conv_stride[i], 
                                                                     padding=conv_padding[i]) 
                                                  for i in range(layers_num)[:0:-1]])
            self.conv_transposed.append(nn.ConvTranspose2d(conv_out_channels[0], input_size[0], conv_kernel_size[0], 
                                                           stride=conv_stride[0], padding=conv_padding[0]))
            im_size = input_size[1]
            for i in range(layers_num):
                im_size = floor((im_size - conv_kernel_size[i] + 2 * conv_padding[i]) / conv_stride[i]) + 1
                im_size = floor((im_size - pool_kernel_size[i] + 2 * pool_padding[i]) / pool_stride[i]) + 1
                if im_size < 1:
                    raise TypeError("Error: during the convolutions and poolings image size became < 1")
            self.num_features = conv_out_channels[-1] * im_size * im_size
        elif layers_num == 0:       # layers_num == 0 means that there is only fully-connected layer in the network
            self.conv = []
            self.pool = []
            self.num_features = input_size[0] * input_size[1] * input_size[2]
        self.fc = nn.Linear(self.num_features, 10)
        
    def forward(self, x, autoencoder=False):
        if autoencoder is True:
            indices = []
            sizes = []
            for i in range(len(self.conv)):
                x = F.tanh(self.conv[i](x))
                sizes.append(x.size())
                x, ind = self.pool[i](x)
                indices.append(ind)
            for i in range(len(self.conv_transposed)):
                x = F.tanh(self.conv_transposed[i](self.unpool[i](x, indices[len(self.unpool) - i - 1], 
                                                                  sizes[len(self.unpool) - i - 1])))
        else:
            for i in range(len(self.conv)):
                x, _ = self.pool[i](F.tanh(self.conv[i](x)))
            x = x.view(-1, self.num_features)
            x = self.fc(x)
        return x
        
    def encode(self, x):
        indices = []
        sizes = []
        for i in range(len(self.conv)):
            x = F.tanh(self.conv[i](x))
            sizes.append(x.size())
            x, ind = self.pool[i](x)
            indices.append(ind)
        return x
    
    def decode(self, x, indices, sizes):
        for i in range(len(self.conv_transposed)):
            x = F.tanh(self.conv_transposed[i](self.unpool[i](x, indices[len(self.unpool) - i - 1], 
                                                              sizes[len(self.unpool) - i - 1])))
        return x
