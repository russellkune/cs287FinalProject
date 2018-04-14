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
parser.add_argument('--load_path', type=str, required=True,
                        help='directory to load models from')
parser.add_argument('--data_path',type = str,required =True, help = 'directiory to load data from')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--maxlen', type=int, default=30,
                    help='maximum sentence length')
parser.add_argument('--temp', type=float, default=1,
                        help='softmax temperature (lower --> more discrete)')
parser.add_argument('--ngenerations', type=int, default=10,
                        help='Number of sentences to generate')
parser.add_argument('--ninterpolations', type=int, default=5,
                        help='Number z-space sentence interpolation examples')
parser.add_argument('--steps', type=int, default=5,
                        help='Number of steps in each interpolation')
parser.add_argument('--outf', type=str, default='./generated.txt',
                        help='filename and path to write to')
parser.add_argument('--noprint', action='store_true',
                        help='prevents examples from printing')
parser.add_argument('--sample', action='store_true',
                        help='sample when decoding for generation')
parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
parser.add_argument('--vocab_size', type=int, default=11000,
                    help='cut vocabulary down to this size '
                         '(most frequently seen words in train)')
parser.add_argument('--lowercase', action='store_true',
                    help='lowercase all text')
parser.add_argument('--lr_inv', type=float, default=1e-05,
                    help='inverter learning rate')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', action = 'store_true',help = 'use cuda')
parser.add_argument('--epochs',type = int, default = 15, help = 'Number of Epochs')
args = parser.parse_args()
print(vars(args))

########################## don't change import statements above this line, the rest needs to be gutted though

###############################################################################
# Training code for Inverter ~~~~
###############################################################################
'''
Adapting the generate.py file 
- given an existing autoencoder, generator, and discriminator, --> train an inverter, perform search to create adversaries



Includes:
- JSDistance
- Inverter
- train_inverter - executes one training step of the inverter
- eval_inverter - also saves output to disk
''' 
####################################################################################
# Monte Carlo Jensen Shannon Divergence
####################################################################################

class JSDistance(nn.Module):
    def __init__(self, mean=0, std=1, epsilon=1e-5):
        super(JSDistance, self).__init__()
        self.epsilon = epsilon
        self.distrib_type_normal = True

    def get_kl_div(self, input, target):
        src_mu = torch.mean(input)
        src_std = torch.std(input)
        tgt_mu = torch.mean(target)
        tgt_std = torch.std(target)
        kl = torch.log(tgt_std/src_std) - 0.5 +\
                    (src_std ** 2 + (src_mu - tgt_mu) ** 2)/(2 * (tgt_std ** 2))
        return kl

    def forward(self, input, target):
        ##KL(p, q) = log(sig2/sig1) + ((sig1^2 + (mu1 - mu2)^2)/2*sig2^2) - 1/2
        if self.distrib_type_normal:
            d1=self.get_kl_div(input, target)
            d2=self.get_kl_div(target, input)
            return 0.5 * (d1+d2)
        else:
            input_num_zero = input.data[torch.eq(input.data, 0)]
            if input_num_zero.dim() > 0:
                input_num_zero = input_num_zero.size(0)
                input.data = input.data - (self.epsilon/input_num_zero)
                input.data[torch.lt(input.data, 0)] = self.epsilon/input_num_zero
            target_num_zero = target.data[torch.eq(target.data, 0)]
            if target_num_zero.dim() > 0:
                target_num_zero = target_num_zero.size(0)
                target.data = target.data - (self.epsilon/target_num_zero)
                target.data[torch.lt(target.data, 0)] = self.epsilon/target_num_zero
            d1 = torch.sum(input * torch.log(input/target))/input.size(0)
            d2 = torch.sum(target * torch.log(target/input))/input.size(0)
            return (d1+d2)/2

###################################################################################
# Inverter class 
###################################################################################

class Inverter(nn.Module):
    # separate Inverter to map continuous code back to z
    def __init__(self, ninput, noutput, layers,
                 activation=nn.ReLU(), gpu=True):
        super(Inverter, self).__init__()
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

############################################################################################
# Save Model
############################################################################################
def save_model():
    print("Saving models")
    with open('./new_output/inverter_model.pt', 'wb') as f:
        torch.save(inverter.state_dict(), f)
############################################################################################
# Eval Inverter
############################################################################################
def evaluate_inverter(data_source, epoch):
    # Turn on evaluation mode which disables dropout.
    autoencoder.eval()
    inverter.eval()
    gan_gen.eval()
    gan_disc.eval()

    for batch in data_source:
        source, target, lengths = batch
        source = to_gpu(args.cuda, Variable(source, volatile=True))
        target = to_gpu(args.cuda, Variable(target, volatile=True))

        # sentence -> encoder -> decoder
        hidden = autoencoder.encode(source, lengths, noise=True)
        ae_indices = autoencoder.generate(hidden, args.maxlen, args.sample)

        # sentence -> encoder -> inverter -> generator -> decoder
        inv_z = inverter(hidden)
        inv_hidden = gan_gen(inv_z)
        eigd_indices = autoencoder.generate(inv_hidden, args.maxlen, args.sample)

        with open("/arae/new_output/%s_inverter.txt" % (epoch), "a") as f:
            target = target.view(ae_indices.size(0), -1).data.cpu().numpy()
            ae_indices = ae_indices.data.cpu().numpy()
            eigd_indices = eigd_indices.data.cpu().numpy()
            for t, ae, eigd in zip(target, ae_indices, eigd_indices):
                # real sentence
                f.write("# # # original sentence # # #\n")
                chars = " ".join([corpus.dictionary.idx2word[x] for x in t])
                f.write(chars)
                # autoencoder output sentence
                f.write("\n# # # sentence -> encoder -> decoder # # #\n")
                chars = " ".join([corpus.dictionary.idx2word[x] for x in ae])
                f.write(chars)
                # corresponding GAN sentence
                f.write("\n# # # sentence -> encoder -> inverter -> generator "
                        "-> decoder # # #\n")
                chars = " ".join([corpus.dictionary.idx2word[x] for x in eigd])
                f.write(chars)
                f.write("\n\n")
############################################################################################
# Training step for inverter
############################################################################################
def train_inv(batch,i,epoch,total_err):
    '''
    one training step of the inverter
    '''
    inverter.train()
    autoencoder.train()
    gan_gen.train()
    gan_disc.train()
    inverter.zero_grad()
    autoencoder.zero_grad()
    gan_gen.zero_grad()
    gan_disc.zero_grad()
   
    #generate noise
    noise = torch.ones(args.ngenerations, model_args['z_size'])
    noise.normal_()
    noise = to_gpu(args.cuda, Variable(noise))

    #generate a fake code from noise
    source,target,lengths = batch
    source = to_gpu(args.cuda, Variable(source))
    target = to_gpu(args.cuda, Variable(target))
    real_code = autoencoder.encode(source,lengths,noise = True)
    real_code = real_code.detach()
    fake_code = gan_gen(noise)
    inv_noise = inverter(fake_code)
    real_to_noise = inverter(real_code)
    errI = criterion_js(inv_noise,noise)
    noise_to_code = gan_gen(real_to_noise)
    errII = criterion_mse(noise_to_code,real_code)
    err = errI+errII
    err.backward()
    optimizer_inv.step()
    total_err += err
    if i %500 == 0:
        print("MSE: ", float(errII.data.cpu().numpy()))
        print("Jensen Shannon: ", float(errI.data.cpu().numpy()))
        print("Error: ", float(err.data.cpu().numpy()))
        print("Total Error: ", float(total_err.data.cpu().numpy()))    
    if i % 1000 ==0:
        
        
        cur_loss = float((total_err/1000.0).data.cpu().numpy())
        print(cur_loss)
        print('| epoch {0} | {1}/{2} batches|loss {3:.3f}'.format(epoch, i, len(train_data), cur_loss))
        save_model()
    
    return err

############################################################################################
# Main method
############################################################################################

# Set the random seed manually for reproducibility.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
else:
    print("Note that our pre-trained models require CUDA to evaluate.")

    ###########################################################################
    # Load the models
    ###########################################################################

model_args, idx2word, autoencoder, gan_gen, gan_disc = load_models(args.load_path)
criterion_mse = nn.MSELoss()
criterion_js = JSDistance()
criterion_ce = nn.CrossEntropyLoss()
print(model_args)
inverter = Inverter(ninput =model_args['nhidden'],noutput = model_args['z_size'],layers = '300-300')
print(inverter)
optimizer_inv = optim.Adam(inverter.parameters(),lr=args.lr_inv,betas=(args.beta1, 0.999))
    
    
if args.cuda:
    autoencoder= autoencoder.cuda()
    gan_gen = gan_gen.cuda()
    gan_disc = gan_disc.cuda()
    inverter = inverter.cuda()
    criterion_mse = criterion_mse.cuda()
    criterion_js = criterion_js.cuda()
    criterion_ce = criterion_ce.cuda()
    
corpus = Corpus(args.data_path,maxlen=args.maxlen,vocab_size=args.vocab_size,lowercase=args.lowercase)
eval_batch_size = 10
test_data = batchify(corpus.test, eval_batch_size, shuffle=False)
train_data = batchify(corpus.train, args.batch_size, shuffle=True)
print("Loaded data!")
###########################################################################
# Training Code
###########################################################################
print('Training...')
for epoch in range(1,1+args.epochs):
    total_loss = Variable(torch.FloatTensor([0.0])).cuda()
    niter = 0
    while niter < len(train_data):
        if niter == len(train_data):
            break    
        batch = train_data[niter]
        niter += 1
        loss = train_inv(batch,niter,epoch, total_loss)
        total_loss += loss 
        if niter > 1000:
            total_loss = Variable(torch.FloatTensor([0.0])).cuda()
    
    #shuffle training data between epochs  
    train_data = batchify(corpus.train, args.batch_size, shuffle=True)
    


