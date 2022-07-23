# Most basic stuff for EDA.

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 150)
import matplotlib.pyplot as plt
import seaborn as sns

# !pip install transformers
# !pip install datasets
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer 

import warnings
warnings.filterwarnings("ignore")

# Some basic helper functions to clean text by removing urls, emojis, html tags and punctuations.
import re
import string

from datasets import load_metric
from torch.utils.data import DataLoader
import warnings,transformers,logging,torch
from transformers import TrainingArguments,Trainer
from transformers import AutoModelForSequenceClassification,AutoTokenizer

from datasets import load_dataset, Dataset, DatasetDict
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.optim as optim                                                                                                                                                                                       

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List
from collections import Counter
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

accuracy = load_metric("accuracy")
f1 = load_metric('f1')
lr,bs = 8e-5,32
wd,epochs = 0.01,10
model_nm = 'microsoft/deberta-v3-small'

tokz = AutoTokenizer.from_pretrained(model_nm)

torch.backends.cudnn.enabled=False

def tok_func(x):
    return tokz(x['train'])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy.compute(predictions = predictions, references=labels)['accuracy'],
        'f1': f1.compute(predictions = predictions, references=labels)['f1']
    }



def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)


def remove_emoji(text):
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_html(text):
    html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(html, '', text)


def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)

def get_data():
    train_tweets = pd.read_csv('../data/train.csv')
    test_tweets = pd.read_csv('../data/test.csv')
    
    
    # Applying helper functions on Train Dataset

    train_tweets['text_clean'] = train_tweets['text'].apply(lambda x: remove_URL(x))
    train_tweets['text_clean'] = train_tweets['text_clean'].apply(lambda x: remove_emoji(x))
    train_tweets['text_clean'] = train_tweets['text_clean'].apply(lambda x: remove_html(x))
    train_tweets['text_clean'] = train_tweets['text_clean'].apply(lambda x: remove_punct(x))

    # Applying helper functions on Test Dataset

    test_tweets['text_clean'] = test_tweets['text'].apply(lambda x: remove_URL(x))
    test_tweets['text_clean'] = test_tweets['text_clean'].apply(lambda x: remove_emoji(x))
    test_tweets['text_clean'] = test_tweets['text_clean'].apply(lambda x: remove_html(x))
    test_tweets['text_clean'] = test_tweets['text_clean'].apply(lambda x: remove_punct(x))

    train_tweets.fillna("",inplace=True)
    test_tweets.fillna("",inplace=True)
    
    model_nm = 'microsoft/deberta-v3-small'
    
    tokz = AutoTokenizer.from_pretrained(model_nm)
    sep = tokz.sep_token
    
    train_tweets['train'] = train_tweets['text_clean'] + sep + train_tweets['location'] + sep + train_tweets['keyword']
    test_tweets['train'] = test_tweets['text_clean'] + sep + test_tweets['location'] + sep + test_tweets['keyword']
    
    ds = Dataset.from_pandas(train_tweets)
    eval_ds = Dataset.from_pandas(test_tweets)
    
    
    return ds, eval_ds



class LSTMModel(nn.Module):
    def __init__(self, checkpoint, num_labels):
        super(LSTMModel, self).__init__()
        self.num_labels = num_labels
        
        # Load a model with given checkpoint and extract its body
        self.model  = AutoModel.from_pretrained(checkpoint,config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))
        self.lstm_hidden_size = 768
        self.lstm = nn.LSTM(self.lstm_hidden_size, self.lstm_hidden_size, bidirectional=True,batch_first=True)
        
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(2*self.lstm_hidden_size, num_labels)
        
    def forward(self, input_ids = None, attention_mask = None, labels = None):
        # Extract outputs from the body
        # get sentence length with pad_id = 0
        sent_lengths = attention_mask.sum(dim=-1).cpu()
        #sent_lengths = get_sent_lengths(input_ids)
        outputs = self.model(input_ids = input_ids, attention_mask=attention_mask)
        
        
        # pool the output using LSTM layer
        
        # sequence output is batch_size, seq_length, hidden_dim
        enc_hiddens, (last_hidden, last_cell) = self.lstm(pack_padded_sequence(outputs[0], sent_lengths, enforce_sorted=False, batch_first=True))
        
        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)  # (batch_size, 2*hidden_size)
        output_hidden = self.dropout(output_hidden)
        
        logits = self.classifier(output_hidden)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            

        return SequenceClassifierOutput(loss = loss, logits = logits, hidden_states = outputs.hidden_states, attentions=outputs.attentions)




def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
            
        return txt_input
    
    return func

def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]).cuda(), 
                      torch.tensor(token_ids).cuda(), 
                      torch.tensor([EOS_IDX]).cuda()))
    
from torch import nn

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()
        self.num_labels = num_class

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self,input_ids, attention_mask = None, labels = None):
        input_ids = input_ids.cuda()
        labels = labels.cuda()
        embedded = self.embedding(input_ids)
        
        logits = self.fc(embedded)
    
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            

        return SequenceClassifierOutput(loss = loss, logits = logits)


class TextClassificationModelWithLSTM(nn.Module):
    
    def __init__(self, vocab_size, embed_dim, num_class, hidden_dim, n_layers=2,dropout=0.2):
        super(TextClassificationModelWithLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=False)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,bidirectional=True,dropout= dropout, batch_first=True)
        self.fc = nn.Linear(2*hidden_dim, num_class)
        self.init_weights()
        self.num_labels = num_class

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self,input_ids, attention_mask = None, labels = None):
        input_ids = input_ids.cuda()
        labels = labels.cuda()
        embedded = self.embedding(input_ids)
        sent_lengths = attention_mask.sum(dim=-1).cpu()
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sent_lengths,batch_first=True,enforce_sorted=False)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        logits=self.fc(hidden)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            

        return SequenceClassifierOutput(loss = loss, logits = logits)


def get_lstm_trainer():
    ds, eval_ds = get_data()
    tokenizer = get_tokenizer('spacy', language='de_core_news_sm')

    def yield_tokens(ds):
        for row in ds:
            yield tokenizer(row['train'])
    
    vocab_transform = build_vocab_from_iterator(yield_tokens(ds), specials=special_symbols,min_freq=1,special_first=True)
    print('length of vocabulary ', len(vocab_transform))
    vocab_transform.set_default_index(UNK_IDX)

    text_transform = sequential_transforms(
        tokenizer,
        vocab_transform
    )
    
    def tok_func(x):
        return {'input_ids': text_transform(x['train'])}
    
    tok_ds = ( ds.map(tok_func, batched=False, remove_columns=['id', 'keyword', 'location', 'text', 'text_clean'])
               .rename_column('target','label'))
    dds = tok_ds.train_test_split(test_size=0.2)

    args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
    evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
    num_train_epochs=epochs, weight_decay=wd, report_to='none')
    
    vocab_size = len(vocab_transform)
    emsize = 64
    num_class = 2
    model = TextClassificationModel(vocab_size, emsize, num_class)
    #model = TextClassificationModelWithLSTM(vocab_size, emsize, num_class,hidden_dim=emsize)

    trainer = Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
               tokenizer=tokz, compute_metrics=compute_metrics)
    
    
    return trainer

def get_trainer():
    ds, eval_ds = get_data()
    tok_ds = ( ds.map(tok_func, batched=True, remove_columns=['id', 'keyword', 'location', 'text', 'text_clean'])
               .rename_column('target','label'))
    
    dds = tok_ds.train_test_split(test_size=0.2)
    
    args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
    evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
    num_train_epochs=epochs, weight_decay=wd, report_to='none')
    
    custom_model = LSTMModel(checkpoint = model_nm,num_labels=2)

    #model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=2)
    trainer = Trainer(custom_model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
               tokenizer=tokz, compute_metrics=compute_metrics)
    
    
    return trainer
    
if __name__ == '__main__':
    #trainer = get_lstm_trainer()
    trainer = get_trainer()
    trainer.train(ignore_keys_for_eval=['hidden_states','attentions'])