import pdb
import numpy as np
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from transformers import BertModel

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

class Embed(nn.Module):
    def __init__(self,ntoken, dictionary, ninp, word_vector=None):
        super(Embed, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.dictionary = dictionary
        if word_vector is not None:
            self.encoder.weight.data[self.dictionary.word2idx['<pad>']] = 0
            self.encoder.weight.data[self.dictionary.word2idx['<pad>']] = 0
            if os.path.exists(word_vector):
                print('Loading word vectors from', word_vector)
                vectors = torch.load(word_vector)
                assert vectors[3] >= ninp
                vocab = vectors[1]
                vectors = vectors[2]
                loaded_cnt = 0
                unseen_cnt = 0
                for word in self.dictionary.word2idx:
                    if word not in vocab:
                        to_add = torch.zeros_like(vectors[0]).uniform_(-0.25,0.25)
                        #print("uncached word: " + word)
                        unseen_cnt += 1
                        #print(to_add)
                    else:
                        loaded_id = vocab[word]
                        to_add = vectors[loaded_id][:ninp]
                        loaded_cnt += 1
                    real_id = self.dictionary.word2idx[word]
                    self.encoder.weight.data[real_id] = to_add
                print('%d words from external word vectors loaded, %d unseen' % (loaded_cnt, unseen_cnt))  
      
    def forward(self,input):
        return self.encoder(input)

class RNN(nn.Module):
    def __init__(self, inp_size, nhid, nlayers):
        super(RNN, self).__init__()
        self.nlayers = nlayers
        self.nhid = nhid
        self.rnn = nn.GRU(inp_size, nhid, nlayers, bidirectional=True)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return torch.zeros(self.nlayers * 2, bsz, self.nhid, dtype=weight.dtype,
                            layout=weight.layout, device=weight.device)

    def forward(self, input, hidden):
        out_rnn = self.rnn(input, hidden)[0]
        return out_rnn

class Attention(nn.Module):
    def __init__(self,inp_size, attention_unit, attention_hops, dictionary, dropout):
        super(Attention,self).__init__()
        self.ws1 = nn.Linear(inp_size, attention_unit, bias=False)
        self.ws2 = nn.Linear(attention_unit, attention_hops, bias=False)
        self.dictionary = dictionary
        self.attention_hops = attention_hops
        self.drop = nn.Dropout(dropout)

    def get_mask(self,input_raw):
        transformed_inp = torch.transpose(input_raw, 0, 1).contiguous()  # [bsz, seq_len]
        transformed_inp = transformed_inp.view(input_raw.size()[1], 1, input_raw.size()[0])  # [bsz, 1, seq_len]
        concatenated_inp = [transformed_inp for i in range(self.attention_hops)]
        concatenated_inp = torch.cat(concatenated_inp, 1)  # [bsz, hop, seq_len]
        mask = (concatenated_inp == self.dictionary.word2idx['<pad>']).float()
        mask = mask[:,:,:input_raw.size(0)]
        return mask

    def forward(self, input, input_raw): # input --> (seq_len, bsize, inp_size) input_raw --> (seq_len, bsize)
        inp = torch.transpose(input, 0, 1).contiguous()
        size = inp.size()  # [bsz, seq_len, inp_size]
        compressed_embeddings = inp.view(-1, size[2])  # [bsz*seq_len, inp_size]
        mask = self.get_mask(input_raw) # need this to mask out the <pad>s
        hbar = torch.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*seq_len, attention-unit]
        alphas = self.ws2(self.drop(hbar)).view(size[0], size[1], -1)  # [bsz, seq_len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, seq_len]
        penalized_alphas = alphas + -10000*mask
        alphas = F.softmax(penalized_alphas.view(-1, size[1]),1)  # [bsz*hop, seq_len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, seq_len]
        out_agg, attention = torch.bmm(alphas, inp), alphas # [bsz, hop, inp_size], [bsz, hop, seq_len] 
        return out_agg, attention

class Classifier(nn.Module):
    def __init__(self,inp_size, nfc, nclasses, dropout):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(inp_size, nfc)
        self.pred1 = nn.Linear(nfc, nfc)
        self.pred2 = nn.Linear(nfc, nclasses)
        self.drop = nn.Dropout(dropout)

    def forward(self, input):
        fc = torch.tanh(self.fc(self.drop(input)))
        pred1 = torch.tanh(self.pred1(self.drop(fc)))
        pred = self.pred2(self.drop(pred1))
        return pred

class RnnAtt(nn.Module):
    def __init__(self,config):
        super(RnnAtt, self).__init__()
        self.emb = Embed(config['ntoken'], config['dictionary'], config['ninp'], config['word-vector'])
        self.rnn = RNN(config['ninp'], config['nhid'], config['nlayers'])
        self.attention = Attention(2*config['nhid'], config['attention-unit'], config['attention-hops'], config['dictionary'], config['dropout'])
        self.downsize = nn.Linear(2*config['nhid'] * config['attention-hops'], 2*config['nhid'])
        self.classifier = Classifier(2*config['nhid'], config['nfc'], config['nclasses'], config['dropout'])
        self.drop = nn.Dropout(config['dropout'])

    def init_hidden(self,bsz):
        return self.rnn.init_hidden(bsz)

    def encode(self,input,hidden):
        emb_out = self.emb(input)
        rnn_out = self.rnn(self.drop(emb_out),hidden)
        out_agg, attn = self.attention(rnn_out,input)
        out_agg = out_agg.view(out_agg.size(0), -1)
        rep = self.downsize(self.drop(out_agg))
        return rep

    def classify(self,r):
        return self.classifier(r)

    def forward(self,input,hidden):
        rep = self.encode(input,hidden)
        return self.classify(rep)

    def flatten_parameters(self):
        self.rnn.rnn.flatten_parameters()

    def freeze_encoder(self):
        self.emb.eval()
        freeze(self.emb)
        self.rnn.eval()
        freeze(self.rnn)
        self.attention.eval()
        freeze(self.attention)
        self.downsize.eval()
        freeze(self.downsize)

    def freeze_cls(self):
        self.classifier.eval()
        freeze(self.classifier)

class BertClass(nn.Module):
    def __init__(self, config):
        super(BertClass, self).__init__()
        self.bert_model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states = True)
        #self.classifier = Classifier2(768, config['nclasses'], config['dropout']) #Classifier(768, config['nfc'], config['nclasses'], config['dropout'])
        self.classifier = Classifier(768, config['nfc'], config['nclasses'], config['dropout'])
        self.drop = nn.Dropout(config['dropout'])

    def pool(self, r, mask):
        lens = mask.sum(dim = 1, keepdim = True)
        r_ = (r * mask.unsqueeze(2)).sum(dim = 1) / lens
        return r_
        
    def encode(self, input, hidden=None):
        emb = self.bert_model.get_input_embeddings()
        eout = emb(input["input_ids"])
        mask = input["attention_mask"]
        out = self.bert_model(inputs_embeds = eout)
        last = out[0]
        r = self.pool(last, mask)
        return r

    def classify(self,r):
        return self.classifier(r)

    def forward(self,input,hidden):
        rep = self.encode(input,hidden)
        return self.classify(rep)

    def init_hidden(self,bsz):
        return None

    def freeze_encoder(self):
        self.bert_model.eval()
        freeze(self.bert_model)

    def freeze_cls(self):
        self.classifier.eval()
        freeze(self.classifier)

