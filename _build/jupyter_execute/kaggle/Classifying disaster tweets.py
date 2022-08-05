#!/usr/bin/env python
# coding: utf-8

# # Classifying disaster tweets
# 

# In[1]:


# Most basic stuff for EDA.

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 150)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# !pip install transformers
# !pip install datasets
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer 

import warnings
warnings.filterwarnings("ignore")


# # Read the data
# 

# In[2]:


train_tweets = pd.read_csv('../data/train.csv')
test_tweets = pd.read_csv('../data/test.csv')


# In[3]:


train_tweets.head()


# # EDA

# In[4]:


sns.set_style('whitegrid')
sns.countplot(y=train_tweets['target'])


# In[5]:


train_tweets['location'].value_counts().head(n=20)


# Let's clean the text

# In[6]:


# Some basic helper functions to clean text by removing urls, emojis, html tags and punctuations.
import re
import string

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


# In[7]:


train_tweets.fillna("",inplace=True)
test_tweets.fillna("",inplace=True)


# In[8]:


model_nm = 'microsoft/deberta-v3-small'


# In[9]:


tokz = AutoTokenizer.from_pretrained(model_nm)


# In[10]:


sep = tokz.sep_token


# In[11]:


train_tweets['train'] = train_tweets['text_clean'] + sep + train_tweets['location'] + sep + train_tweets['keyword']


# In[12]:


train_tweets.head()


# # Training

# Time to import some stuff we'll need for training

# In[13]:


from torch.utils.data import DataLoader
import warnings,transformers,logging,torch
from transformers import TrainingArguments,Trainer
from transformers import AutoModelForSequenceClassification,AutoTokenizer

from datasets import load_dataset, Dataset, DatasetDict


# In[14]:


def tok_func(x):
    return tokz(x['train'])


# In[15]:


ds = Dataset.from_pandas(train_tweets)


# In[16]:


train_tweets.columns


# In[17]:


tok_ds = ds.map(tok_func, batched=True, remove_columns=['id', 'keyword', 'location', 'text', 'target', 'text_clean'])


# In[18]:


tok_ds[0]


# In[19]:


dds = tok_ds.train_test_split(test_size=0.2)


# # Initial model

# In[20]:


lr,bs = 8e-5,128
wd,epochs = 0.01,4


# In[21]:


args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
    evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
    num_train_epochs=epochs, weight_decay=wd, report_to='none')


# We can now create our model, and `Trainer` which is a class which combines the data and mdoel together (just like Learner in fastai)

# In[51]:


model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)
trainer = Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
               tokenizer=tokz, compute_metrics=corr)


# In[ ]:




