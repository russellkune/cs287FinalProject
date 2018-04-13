import os
import json
import codecs
import argparse
import torchtext
from torchtext.vocab import Vectors, GloVe

"""
Performs preprocessing for the SST sentiment classification task

idk how to do this yet

OK LETS FUCKING GO BOI
"""
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



def transform_data(input):
    processed = []
    for example in input:
        txt = example.text
        joined = " ".join(txt)
        processed.append(joined)
    return processed


def write_sentences(write_path, processed):
    print("Writing to {}\n".format(write_path))
    with open(write_path, "w") as f:
        for p in processed:
            f.write(p)
            f.write("\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, default="new_output/sst_processed",
                        help='path to write SST language modeling data to')
    args = parser.parse_args()
    # make out-path directory if it doesn't exist
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
        print("Creating directory "+args.out_path)
    processed_train = transform_data(train)
    processed_test = transform_data(test)
    write_sentences(write_path=os.path.join(args.out_path, "train.txt"), processed = processed_train)
    write_sentences(write_path=os.path.join(args.out_path, "test.txt"), processed = processed_test)




