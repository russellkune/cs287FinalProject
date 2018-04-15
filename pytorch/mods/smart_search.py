import argparse
import numpy as np
import random

import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#this code gets the namespace from the directory above 
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from models import load_models, generate
from utils import to_gpu, Corpus, batchify


parser = argparse.ArgumentParser(description='PyTorch ARAE for Text Eval')

###############################################################
# Load required models:
### ARAE
### Inverter
### "black box classifier"
##############################################################
parser.add_argument('--load_path', type=str, required = True,
                    help='load path for ARAE model')
parser.add_argument('--inverter_path', type = str, required = True, help = 'load path for inverter')
parser.add_argument('--classifier_path', type = str, required = True, help = 'load path for classifier')

parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                    help='batch size')
parser.add_argument('--epochs' , type= int, default= 15, help = 'Number of epochs' )
parser.add_argument('--cuda', type = bool, default = True, help = 'cuda')
args = parser.parse_args()
print(vars(args))


if args.cuda:
    gpu = True
else:
    gpu = False 



################################################################

################################################################

class Search(nn.Module):
    def __init__(ninput, noutput, layers,vocab_size, activation=nn.ReLU(), gpu=True):
        super(Search, self).__init__()
        

        #embedding table and LSTM for actual sentence
        self.embedding = nn.Embedding(vocab_size, ninput)
        self.lstm = nn.LSTM(...)
        
        #MLP for latent space code
        self.ninput = ninput
        self.noutput= noutput
        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []
  
        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            # No batch normalization after first layer
            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)
        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)
    def forward(self, z, sent, delta_z):
        for layer in self.layers:
            z = layer(z)
        #take the 2-norm of Z
        qn = torch.norm(z, p=2, dim=1).detach()
        #normalize Z
        z = z.div(qn.expand_as(z))
        return(delta_z*z)
             


def train_search(batch, mag):
    '''
    training function for search 
    '''
    #psuedocode bassically -- first zero gradients
    sentence, label = batch
    encoded = autoencoder.encode(sentence).detach()
    latent_code = inverter(encoded)
    delta = search(latent_code,sentence,mag)
    perturbation = latent_code + delta
    new_sent = autoencode.decode(perturbation)
      
    out = classifier(new_sent)

    #some kind of polcy grad here
    return


