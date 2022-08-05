# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import cv2
import sys
import re
import math
import os, ast
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib
matplotlib.use('TkAgg')

from matplotlib import pyplot as plt # plotting
import matplotlib.patches as patches

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.preprocessing import OneHotEncoder

import seaborn as sns
from tqdm import tqdm
import matplotlib.cm as cm
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from pathlib import Path


## pytorch imports ##
from torch.utils.data import DataLoader
import warnings,transformers,logging,torch
from transformers import TrainingArguments,Trainer
from transformers import AutoModelForSequenceClassification,AutoTokenizer

from datasets import load_dataset, Dataset, DatasetDict, Image,IterableDataset
from datasets.iterable_dataset import iterable_dataset
from datasets import Features, Sequence, Value, ClassLabel

# cv2
import cv2
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2


from models.rcnn import LitWheat
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from functools import partial
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from typing import Iterable

dataset_path = Path('/extras/Data')
dataset_img_train = dataset_path/'train'
dataset_img_test = dataset_path/'test'
WANDB_PROJECT = "wheat-detection"

import wandb
wandb.login()

# Using a small image size so it trains faster but do try bigger images 
# for better performance.
IMG_SIZE = 256

def get_train_transforms():
    return A.Compose(
        [
            A.RandomSizedCrop(min_max_height=(800, 800), height=IMG_SIZE, width=IMG_SIZE, p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                           contrast_limit=0.2, p=0.9),
            ],p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=256, width=256, p=1),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )

def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )



def get_test_transforms():
    return A.Compose([
            A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(p=1.0),
        ], p=1.0)

def annotate_examples(examples, idx):
    image_id = examples['image_id']
    examples['boxes'] = np.array(examples['boxes'])
    examples['images']= str(dataset_img_train/(image_id+".jpg"))
    examples['labels'] = np.ones(examples['boxes'].shape[0])
    examples['area'] = examples['box_area']
    examples['iscrowd'] = np.zeros(examples['boxes'].shape[0])
    examples['image_id'] = idx

        
    return examples

def image_test_transform(example, transform=None):
    image = np.array(example['images'][0])
    if transform is not None:
        transformed = transform(image=image)
        image = transformed['image']
    example['pixel_values'] = [image]
    
    return example
 
def image_transform(example, transform=None):
    image = np.array(example['images'][0])
    boxes = np.array(example['boxes'][0])
    if transform is not None:
        transformed = transform(image=image, bboxes = boxes,labels=['wheat_head']*len(boxes))
        image = transformed['image']
        boxes = transformed['bboxes']
        
    if len(boxes) > 0:
        boxes = torch.stack([torch.tensor(item) for item in boxes])
    else:
        boxes = torch.zeros((0, 4))
    
    example['pixel_values'] = [image]
    example['boxes'] = [boxes]
    return example

def predict_box(model_path:str, file_paths:Iterable[str],draw=True,threshold=0.6):
    test_ds = (Dataset.from_dict({'images':file_paths})
           .cast_column('images',Image())
           .with_transform(partial(image_test_transform, transform=get_test_transforms()))
          )
    trainer = pl.Trainer(accelerator="gpu", devices=[0])

    test_dl = DataLoader(test_ds,batch_size=1, shuffle=True, collate_fn=partial(collate_fn,test=True), num_workers=1)
    model = LitWheat.load_from_checkpoint(model_path)
    predictions = trainer.predict(model, test_dl)
    print("length of predictions ", len(predictions))
    
    if draw:
        images = []
        
        # remember we need to add '0' to the predictions array
        # because our batch_size is 1
        
        # if our batch size were 2 then then each predictions will be 
        # 2 dimensional array
        for image, boxes in zip(test_ds, predictions):
            img_rgb = image['images'].resize((256, 256))
            img_rgb_np = np.array(img_rgb)
            boxes = boxes[0]
               
            for box, score in zip(boxes['boxes'].cpu().numpy(), boxes['scores'].cpu().numpy()):  
                if score > threshold:
                    cv2.rectangle(img_rgb_np,
                        (box[0], box[1]),
                        (box[2], box[3]),
                        (220, 0, 0), 2)
            
            images.append(img_rgb_np)

        return images, predictions
    return predictions

def prepare_data():
    train_df = pd.read_csv(dataset_path/'train.csv')
    sample_sub_df = pd.read_csv(dataset_path/'sample_submission.csv')    
    
    train_df['x_min']=train_df['bbox'].apply(lambda x:float(re.findall(r'[0-9.]+',x.split(',')[0])[0]))
    train_df['y_min']=train_df['bbox'].apply(lambda x:float(re.findall(r'[0-9.]+',x.split(',')[1])[0]))
    train_df['box_width']=train_df['bbox'].apply(lambda x:float(re.findall(r'[0-9.]+',x.split(',')[2])[0]))
    train_df['box_height']=train_df['bbox'].apply(lambda x:float(re.findall(r'[0-9.]+',x.split(',')[3])[0]))
    
    train_df.drop('bbox',axis=1,inplace=True)

    train_df['box_area']=train_df['box_width']*train_df['box_height']
    train_df['x_max']=train_df['x_min']+train_df['box_width']
    train_df['y_max']=train_df['y_min']+train_df['box_height']
    
    train_df['count'] = train_df.apply(lambda row: 1 if np.isfinite(row.width) else 0, axis=1)
    train_df['boxes'] = list(train_df[['x_min','y_min','x_max','y_max']].values.tolist())
    combined_train = train_df.groupby(['image_id']).agg({'count': 'sum', 'boxes': list,'box_area': list})
    
    
    ds = Dataset.from_pandas(combined_train)
    processed_ds = (ds.map(annotate_examples,batched=False, with_indices=True)
                .cast_column('images', Image(decode=True))
               )
    
    dds = processed_ds.train_test_split(test_size=0.2)
    
    
    dds['train'].set_transform(partial(image_transform, transform = get_train_transforms()))
    dds['test'].set_transform(partial(image_transform, transform = get_valid_transforms()))
    
    return dds
    

def collate_fn(batch,test=False):
    
    images = []
    labels = []

    for example in batch:
        # apply train transforms
        image = example['pixel_values']
        images.append(image)
        if not test:
            label = {
                'boxes' : example['boxes'].float(),
                'labels': torch.tensor(example['labels'],dtype=torch.int64),
                'area': torch.tensor(example['area']),
                'iscrowd': torch.tensor(example['iscrowd']),
                'image_id': torch.tensor(example['image_id'])
            }
            labels.append(label)
        
    pixel_values = torch.stack(images)
    if not test:
        return {'pixel_values': pixel_values, 'labels': labels}
    else:
        return {'pixel_values': pixel_values}
    
def get_dataloaders(dds):
    return (DataLoader(
        dds['train'],
        batch_size=8, shuffle=True, collate_fn=collate_fn,
        num_workers=4),
        DataLoader(dds['test'],batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=4))

def train():
    dds = prepare_data()
    
    early_stop_callback = EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='max'
    )
    
    logger = WandbLogger(project=WANDB_PROJECT, log_model=True)
    checkpoint_callback =  ModelCheckpoint(monitor='val_accuracy',mode='max',
                                             filename='{epoch}-{step}--{val_loss:.1f}',
                                             save_top_k=1,dirpath="./models/wheat-detection/")
    
    trainer = pl.Trainer(max_epochs=4, 
                         logger = logger,
                         callbacks=[
                            early_stop_callback, 
                            TQDMProgressBar(refresh_rate=2),
                            checkpoint_callback
                        ],
                        
                        accelerator='gpu',
                        fast_dev_run=False)

    detector = LitWheat(2)
    train_dl, test_dl = get_dataloaders(dds)
    trainer.fit(detector, train_dl, test_dl)
    
    # save the model
    torch.save(detector,"models/model.pth")
    
def predict():
    from PIL import Image
    import glob
    model_path = sys.argv[1]

    images = glob.glob('/extras/Data/test/*.jpg')
    print(images)
    
    imgs, predictinos = predict_box(model_path, images)
    for id, img in enumerate(imgs):
        im = Image.fromarray(img)
        output_filename = f'output/output_{id}.jpg'
        im.save(output_filename)
    # fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    # ax.set_axis_off()
    
    # ax.imshow(imgs[0])
    # plt.show()
    # plt.pause(3)
    
if __name__ == '__main__':
    train()
    #predict()
 
    