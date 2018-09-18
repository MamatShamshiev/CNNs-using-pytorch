import numpy as np
from math import floor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.utils
import torch.optim as optim
from torch.autograd import Variable


class ConvNetAutoEncoder(nn.Module):
    """
    The implementation of convolutional neural network and convolutional autoencoder.
    
    The instance of a class can either be a CNN or an autoencoder.
    
    The implementation is based on a Pytorch library.
    
    """
    def __init__(self, input_size=(3, 32, 32), conv_layers_num=2, conv_out_channels=2, conv_kernel_size=4, conv_stride=2, 
                 conv_padding=0, pool_kernel_size=2, pool_stride=1, pool_padding=0, linear_layers_num=1, linear_out=2):
        
        """
        Note: in every layer parameter, that require array-like of ints, one can pass a single int, which is equivalent to 
            passing the tuple of required shape containing passed int in every position
            
        Parameters:
        ---------------

        input_size: (int, int, int)
            (number of channels, height, width) in the input image
        
        conv_layers_num: int
            Number of convolutional blocks (convolution + activation + pooling)
            
        conv_out_channels: array-like of ints with shape (conv_layers_num, ) or int
            Number of channels produced by the convolutional layers
            If int, defines number of channels produced by every convolutional block
        
        conv_kernel_size: array-like of ints with shape (conv_layers_num, ) or int
            Sizes of the convolving kernels
            If int, kernel with size (conv_kernel_size, conv_kernel_size) is used in every convolution
            
        conv_stride: array-like of ints with shape (conv_layers_num, ) or int
            Stride of the convolutions
            
        conv_padding: array-like of ints with shape (conv_layers_num, ) or int
            Zero-padding added to both sides of the image
            
        pool_kernel_size: array-like of ints with shape (conv_layers_num, ) or int
            The size of the window to take a max over
        
        pool_stride: array-like of ints with shape (conv_layers_num, ) or int
            The stride of the window
        
        pool_padding: array-like of ints with shape (conv_layers_num, ) or int
            Zero-padding added to both sides of the image
        
        linear_layers_num: int
            Number of fully-connected linear layers
        
        linear_out: array-like of ints with shape (linear_layers_num, ) or int
            Lengths of output feature-vectors
        
        """
        super(ConvNetAutoEncoder, self).__init__()
        
        if linear_layers_num < 1:
            raise ValueError("The number of linear layers must be >=1")
        
        
        if conv_layers_num > 0: 
            
            # if given int, make the tuple of required shape with this int in every position
            if type(conv_out_channels) is int:
                conv_out_channels = (conv_out_channels,) * conv_layers_num
            if type(conv_kernel_size) is int:
                conv_kernel_size = (conv_kernel_size,) * conv_layers_num
            if type(conv_stride) is int:
                conv_stride = (conv_stride,) * conv_layers_num
            if type(conv_padding) is int:
                conv_padding = (conv_padding,) * conv_layers_num    
            if type(pool_kernel_size) is int:
                pool_kernel_size = (pool_kernel_size,) * conv_layers_num
            if type(pool_stride) is int:
                pool_stride = (pool_stride,) * conv_layers_num
            if type(pool_padding) is int:
                pool_padding = (pool_padding,) * conv_layers_num
            if type(linear_out) is int:
                linear_out = (linear_out,) * linear_layers_num
            
            # initialize convolutional layers
            self.conv = nn.ModuleList([nn.Conv2d(input_size[0], conv_out_channels[0], conv_kernel_size[0], 
                                                 stride=conv_stride[0], padding=conv_padding[0])])
            for i in range(conv_layers_num - 1):
                self.conv.append(nn.Conv2d(conv_out_channels[i], conv_out_channels[i+1], conv_kernel_size[i+1],
                                           stride=conv_stride[i+1], padding=conv_padding[i+1]))
            
            # initialize pooling and unpooling layers
            self.pool = nn.ModuleList([nn.MaxPool2d(pool_kernel_size[i], stride=pool_stride[i], 
                                                    padding=pool_padding[i], return_indices=True) 
                                       for i in range(conv_layers_num)])
            
            self.unpool = nn.ModuleList([nn.MaxUnpool2d(pool_kernel_size[i], stride=pool_stride[i], 
                                                        padding=pool_padding[i]) 
                                         for i in range(conv_layers_num)[::-1]])
            
            # initialize transposed convolutional layers
            self.conv_transposed = nn.ModuleList([nn.ConvTranspose2d(conv_out_channels[i], conv_out_channels[i-1], 
                                                                     conv_kernel_size[i], stride=conv_stride[i], 
                                                                     padding=conv_padding[i]) 
                                                  for i in range(conv_layers_num)[:0:-1]])
            self.conv_transposed.append(nn.ConvTranspose2d(conv_out_channels[0], input_size[0], conv_kernel_size[0], 
                                                           stride=conv_stride[0], padding=conv_padding[0]))
            
            # calculate image size after all convolutional blocks (needed to initialize first linear layer)
            im_size = input_size[1]
            for i in range(conv_layers_num):
                im_size = floor((im_size - conv_kernel_size[i] + 2 * conv_padding[i]) / conv_stride[i]) + 1
                im_size = floor((im_size - pool_kernel_size[i] + 2 * pool_padding[i]) / pool_stride[i]) + 1
                if im_size < 1:
                    raise TypeError("Error: during the convolutions and poolings image size became < 1")
            self.num_features = conv_out_channels[-1] * im_size * im_size
        elif conv_layers_num == 0:  # there are only fully-connected layers in the network
            self.conv = []
            self.pool = []
            self.num_features = input_size[0] * input_size[1] * input_size[2]
            
        # initialize linear layers
        self.fc = nn.ModuleList([nn.Linear(self.num_features, linear_out[0])])
        for i in range(linear_layers_num - 1):
            self.fc.append(nn.Linear(linear_out[i], linear_out[i+1]))
        
    def forward(self, x, autoencoder=False):
        """
        Parameters:
        ---------------
        x: torch variable, containing tensor of shape (N, C, H, W), where 
            N is the number of samples in the batch
            C is the number of channels in the image
            H is the height of the image
            W is the width of the image
        
        autoencoder: bool
            If True, performs forward propagation for autoencoder
            else performs forward propagation for CNN
        """
        if autoencoder is True:
            indices = []
            sizes = []
            for i in range(len(self.conv)):
                x = F.tanh(self.conv[i](x))
                sizes.append(x.size())
                x, ind = self.pool[i](x)
                indices.append(ind)
            for i in range(len(self.conv_transposed)):
                x = F.tanh(self.conv_transposed[i](self.unpool[i](x, indices[len(self.unpool)-i-1], 
                                                                  sizes[len(self.unpool)-i-1])))
        else:
            for i in range(len(self.conv)):
                x, _ = self.pool[i](F.tanh(self.conv[i](x)))
            x = x.view(-1, self.num_features)
            for i in range(len(self.fc)):
                x = self.fc[i](x)
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
            x = F.tanh(self.conv_transposed[i](self.unpool[i](x, indices[len(self.unpool)-i-1], 
                                                              sizes[len(self.unpool)-i-1])))
        return x

    
def fit_net(net, trainloader, num_epoch=10, optimizer=None, criterion=None, verbose=False):
    """ 
    Fit convolutional neural network
    
    Parameters:
    ---------------
    
    net: instance of ConvNetAutoEncoder class
        CNN to train
    
    optimizer: torch.optim optimizer
        default: optim.SGD(net.parameters(), lr=0.01, weight_decay=0.01)
    
    criterion: loss function from torch.nn module
        default: torch.nn.CrossEntropyLoss()
    
    trainloader: torch.utils.data.DataLoader for training dataset
    
    num_epoch: int
        The number of epoch to train the network
        default: 10
        
    verbose: bool
        If True, print info about optimization process
        
    """
    if optimizer is None:
        optimizer = optim.SGD(net.parameters(), lr=0.01, weight_decay=0.01)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

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
            if verbose is True:
                print('[epoch %d] loss: %.3f' % 
                      (epoch + 1, running_loss / (5 * len(trainloader))))
            last_loss = running_loss / (5 * len(trainloader))
            running_loss = 0.0
    return last_loss


def check_accuracy(net, testloader, verbose=False):
    """
    Calculate accuracy of the network on the test dataset
    
    Parameters:
    ---------------
    net: instance of ConvNetAutoEncoder class
    
    testloader: torch.utils.data.DataLoader for test dataset
    
    verbose: bool
    
    """
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    if verbose is True:
        print('Accuracy of the network: %.2f %%' % (100 * correct / total))
    return 100 * correct / total


def fit_autoencoder(net, optimizer, criterion, loader, num_epoch):  
    """ 
    Fit convolutional autoencoder
    
    Parameters:
    ---------------
    
    net: instance of ConvNetAutoEncoder class
        Autoencoder to train
    
    optimizer: torch.optim optimizer
    
    criterion: loss function from torch.nn module
    
    loader: torch.utils.data.DataLoader for dataset
    
    num_epoch: int
        The number of epoch to train the network
        
    """
    
    running_loss = 0.0
    last_loss = 0.0
    for epoch in range(num_epoch): 
        for i, data in enumerate(loader, 0):
            inputs, _ = data
            inputs = Variable(inputs)
            optimizer.zero_grad()
    
            outputs = net(inputs, autoencoder=True)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
        if epoch % 5 == 4:
            print('[epoch %d] loss: %.3f' % (epoch + 1, running_loss / (5 * len(loader))))
            last_loss = running_loss / (5 * len(loader))
            running_loss = 0.0
    return last_loss

def loader_from_numpy(X, y, batchsize=4, shuffle=True):
    """
    Create dataloader to be passed in fit_net(), fit_autoencoder() or check_accuracy() from numpy array
    Also normalize images from [0..255] to [-1, 1] for better convergence
    
    Parameters:
    ---------------
    
    X: numpy array of shape (N, C, H, W) with values in [0..255], where 
        N is the number of samples in the dataset
        C is the number of channels in the image
        H is the height of the image
        W is the width of the image
    
    y: numpy array of length N 
        Class labels
    
    batchsize: int
        The number of images in the batch
        
    shuffle: bool
        If True, shuffle the data.
   
    """
    X_tensor = torch.stack([torch.Tensor(x) for x in X])
    y_tensor = torch.LongTensor(y)
    transform = transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
    for image in X_tensor:
        transform(image)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=shuffle) 
    return loader

def get_predictions(net, testloader):
    """
    Get predictions by CNN
    Returns the scores of classes
    
    """
    
    predictions = np.empty((0,2))
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images))
        predictions = np.vstack((predictions, outputs.data.numpy()))
    return predictions
