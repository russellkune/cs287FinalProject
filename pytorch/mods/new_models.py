import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import json
import os
import numpy as np


#### New stuff here:

class customConvNet(nn.Module):
    
    def __init__(self,input_embeddings, embedding_dim = 300, hidden_size = 100, vocab_size = 16284):
        super(customConvNet,self).__init__()  
        self.embedding = nn.Embedding(input_embeddings.size(0),input_embeddings.size(1))
        self.embedding.weight = nn.Parameter(input_embeddings,requires_grad = True)
        self.conv3 = nn.Conv1d(embedding_dim,hidden_size,kernel_size = 3,stride = 1)
        self.conv4 = nn.Conv1d(embedding_dim,hidden_size,kernel_size = 4,stride = 1)
        self.conv5 = nn.Conv1d(embedding_dim,hidden_size,kernel_size = 5,stride = 1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.5)
        
        self.linear = nn.Linear(3*hidden_size,2)
        
        
    def forward(self, input_):
        #apply embedding layer
        embeds = self.embedding(input_).permute(1,2,0).contiguous()
        #apply convolution layers
        out1 = F.relu(self.conv3(embeds))
        out2 = F.relu(self.conv4(embeds))
        out3 = F.relu(self.conv5(embeds))
        
        #apply max pooling layers
        out1 = self.maxpool(out1).squeeze(2)
        out2 = self.maxpool(out2).squeeze(2)
        out3 = self.maxpool(out3).squeeze(2)
        #concatenate the outputs; ending up with a batch_size x 3*hidden_size vector
        out = torch.cat((out1,out2,out3),dim = 1)
        out = self.dropout(out)
        return self.linear(out)

'''

Update this code to invert continous code back to z ***

'''
class MLP_I(nn.Module):
    # separate Inverter to map continuous code back to z
    def __init__(self, ninput, noutput, layers,
                 activation=nn.ReLU(), gpu=False):
        super(MLP_I, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)
            self.add_module("layer" + str(i + 1), layer)

            bn = nn.BatchNorm1d(layer_sizes[i + 1], eps=1e-05, momentum=0.1)
            self.layers.append(bn)
            self.add_module("bn" + str(i + 1), bn)

            self.layers.append(activation)
            self.add_module("activation" + str(i + 1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer" + str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass





