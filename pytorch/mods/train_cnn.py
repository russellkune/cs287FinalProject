#ssing library and methods for pretrained word embeddings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np 
# Text text processing library and methods for pretrained word embeddings
import torchtext
from torchtext.vocab import Vectors, GloVe
from new_models import customConvNet
import argparse
from tqdm import tqdm 
###############args

parser = argparse.ArgumentParser(description='PyTorch baseline for Text')

parser.add_argument('--epochs', type=int, default=20,
                    help='maximum number of epochs')

parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                    help='batch size')

parser.add_argument('--packed_rep', type=bool, default=True,
                    help='pad all sentences to fixed maxlen')
parser.add_argument('--train_mode', type=bool, default=True,
                    help='set training mode')
parser.add_argument('--maxlen', type=int, default=10,
                    help='maximum sentence length')
parser.add_argument('--lr', type=float, default=1e-04,
                    help='learning rate')
parser.add_argument('--seed', type=int, default=1111,
                    help='seed')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', action='store_false', default=False,
                    help='use CUDA')
parser.add_argument('--save_path', type=str, required=True,
                    help='used for saving the models')
parser.add_argument('--vocab_size', type=int, default=11004,
                    help='vocabulary size')

args = parser.parse_args()

#####################

# Our input $x$
TEXT = torchtext.data.Field()
# Our labels $y$
LABEL = torchtext.data.Field(sequential=False,unk_token = None)
train, val, test = torchtext.datasets.SST.splits(
    TEXT, LABEL,
    filter_pred=lambda ex: ex.label != 'neutral')
print('building vocab....')
TEXT.build_vocab(train)
LABEL.build_vocab(train)
train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, val, test), batch_size=args.batch_size, device=-1, repeat = False)
print('loading vectors...')
url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))


baseline_model = customConvNet(TEXT.vocab.vectors)

def model_val(model,val_iter):
    criterion = nn.CrossEntropyLoss()
    total_loss = 0 
    count = 0
    correct = 0 
    num_examples = 0
    model.eval()
    
    for batch in val_iter:
        if batch.text.shape[0] >= 5:
            txt = batch.text
            lbl = batch.label
            y_pred = model(txt)
            loss = criterion(y_pred,lbl)
            total_loss += loss.data
            count += 1

            y_pred_max, y_pred_argmax = torch.max(y_pred, 1)
            correct += (y_pred_argmax.data == lbl.data).sum()
            num_examples += y_pred_argmax.size(0)
    model.train()
    return(correct/num_examples, total_loss/count)

print("begin training loop...")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(baseline_model.parameters(), lr=args.lr)


best_accuracy = 0
if args.cuda:
    baseline_model = baseline_model.cuda()

for epoch in tqdm(range(args.epochs)):
    total_loss = 0 
    count = 0
    baseline_model.train()
    
    for batch in tqdm(train_iter):
        if batch.text.shape[0] >= 5:
            optimizer.zero_grad()
            txt = batch.text
            lbl = batch.label
            loss = criterion(baseline_model(txt),lbl)
            total_loss += loss.data
            count += 1
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm(baseline_model.parameters(), 3)
        
        
    print("Average NLL: ", (total_loss/count)) 
    a,b = model_val(baseline_model,test_iter)
    if a > best_accuracy:
    	print("saving model ... ")
    	with open(args.save_path+"/"+'cnn'+'.pt', 'wb') as f:
                torch.save(baseline_model.state_dict(), f)

    	best_accuracy = a

    print("Accuracy:", a)
    print("Val NLL: ", b)





