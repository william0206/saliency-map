import os
import sys

import numpy as np
import torch

from torchtext.vocab import Vectors, GloVe
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator

def load_dataset(data_pth, device, glove):

    """
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied
    Field : A class that stores information about the way of preprocessing
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fi_length which
                 will pad each sequence to have a fix length of 200.
                 
    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.
                  
    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.
    
    """
    tokenize = lambda x: x.split()
    
    TEXT = Field(
        
        sequential=True, 
        tokenize=tokenize, 
        lower=False, 
        include_lengths=True,
        batch_first=True,
        fix_length=256
    )
    
    LABEL = Field(
        
        sequential=False,
        use_vocab=False,
        batch_first=True,
        dtype=torch.float
    )
    
    fields = {"text": ("text", TEXT), "label": ("label", LABEL)}

    train_data, valid_data, test_data = TabularDataset.splits(
        
        path=data_pth, 
        
        train="train.csv", 
        validation="valid.csv", 
        test="dev.csv", 
        
        format="csv",
        fields=fields,
        skip_header=False
    )
    
    TEXT.build_vocab(train_data, vectors=GloVe(name=glove, dim=300))
    word_embeddings = TEXT.vocab.vectors
    
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    
    train_loader, valid_loader = BucketIterator.splits(
        # Datasets for iterator to draw data from
        (train_data, valid_data),

        # Tuple of train and validation batch sizes.
        batch_sizes=(16, 16),

        # Device to load batches on.
        device=device, 

        # Function to use for sorting examples.
        sort_key=lambda x: len(x.text),

        # Repeat the iterator for multiple epochs.
        repeat=False, 

        # Sort all examples in data using `sort_key`.
        sort=False, 

        # Shuffle data on each epoch run.
        shuffle=True,

        # Use `sort_key` to sort examples in each batch.
        sort_within_batch=True,
    )
    test_loader = Iterator(test_data, batch_size=16, device=device, train=False, shuffle=False, sort=False)

    vocab_size = len(TEXT.vocab)

    return TEXT, vocab_size, word_embeddings, train_loader, valid_loader, test_loader