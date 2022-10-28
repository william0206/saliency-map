import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class BiLSTMs_Classifier(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights, train_embedding=False):
        super(BiLSTMs_Classifier, self).__init__()
        
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        
        self.make_embedding = self.embedding(vocab_size, embedding_length, weights, train_embedding)
        self.classifier = self.lstm_classifier(batch_size, output_size, hidden_size, embedding_length)
    
    class embedding(nn.Module):
        def __init__(self, vocab_size, embedding_length, weights, train_embedding=False):
            super(BiLSTMs_Classifier.embedding, self).__init__()
            
            self.word_embeddings = nn.Embedding(vocab_size, embedding_length)# Initializing the look-up table.
            self.word_embeddings.weight = nn.Parameter(weights) # Assigning the look-up table to the pre-trained GloVe word embedding.
            self.word_embeddings.weight.requires_grad = True if train_embedding else False
            
        def forward(self, input_sentence):
            
            sen_embedded = self.word_embeddings(input_sentence) # embedded input of shape = (batch_size, num_sequences, embedding_length)
            sen_embedded = sen_embedded.permute(1, 0, 2)

            return sen_embedded
            
    class lstm_classifier(nn.Module):
        def __init__(self, batch_size, output_size, hidden_size, embedding_length):
            super(BiLSTMs_Classifier.lstm_classifier, self).__init__()
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(embedding_length, hidden_size, bidirectional=True)
            self.label = nn.Linear(2*hidden_size, output_size)
            
        def forward(self, sen_embedded):
            
            batch_size = sen_embedded.shape[0]
            h_0 = Variable(torch.zeros(1*2, batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(1*2, batch_size, self.hidden_size).cuda())

            output, (final_hidden_state, final_cell_state) = self.lstm(sen_embedded)
            forward_output = output[-1, :, :self.hidden_size]
            backward_output = output[0, :, self.hidden_size:]
            output = torch.cat((forward_output, backward_output), 1)
            final_output = self.label(output)

            return final_output
    
    def forward(self, input_sentence):
        
        sen_embedded = self.make_embedding(input_sentence)
        out = self.classifier(sen_embedded)
        
        return out
