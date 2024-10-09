import sys
import torch
import cv2

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from plotter import show_image
from augmentations import get_train_augs, get_valid_augs
import trainer


####################### CONFIGURATIONS #######################

CSV_FILE = './data/train.csv'
DATA_DIR = './data/'

DEVICE = 'cuda'

EPOCHS = 5
learning_rate = 0.001
BATCH_SIZE = 16
IMG_DIM = 512

ENCODER = 'timm-efficientnet-b0'
WEIGHTS = 'imagenet'


####################### PREPARING DATA #######################

df = pd.read_csv(CSV_FILE)

train_df, valid_df = train_test_split(df, test_size=0.20, random_state=42)

train_set = trainer.SegmentationDataset(train_df, get_train_augs(IMG_DIM), DATA_DIR)
valid_set = trainer.SegmentationDataset(valid_df, get_valid_augs(IMG_DIM), DATA_DIR)

print(f'Size of train set: {len(train_set)}')
print(f'Size of valid set: {len(valid_set)}')

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE)

print(f'Total no. of batches in train_loader {len(train_loader)}')
print(f'Total no. of batches in valid_loader {len(valid_loader)}')

for images, masks in train_loader:
  print(f'One batch image shape: {images.shape}')
  print(f'One batch mask shape: {masks.shape}')
  break

model = trainer.SegmentataionModel(ENCODER, WEIGHTS)
model.to(DEVICE)

####################### TRAINING & EVALUATION PHASE #######################

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

best_loss = np.Inf

for i in range(EPOCHS):
  print("Calculating Train Loss...")
  train_loss = trainer.train_fn(train_loader, model, optimizer)
  print("Calculating Validation Loss...")
  valid_loss = trainer.eval_fn(valid_loader, model)

  if valid_loss < best_loss:
    torch.save(model.state_dict(), 'Best-Model.pt')
    print("Saved Model")
    best_loss = valid_loss

  print(f'Epoch: {i+1} Train Loss: {train_loss} Valid Loss: {valid_loss}')



####################### PREDICTION PHASE #######################

idx = 30

model.load_state_dict(torch.load('/content/Best-Model.pt'))
image, mask = valid_set[idx]

logits_mask = model(image.to(DEVICE).unsqueeze(0)) 
pred_mask = torch.sigmoid(logits_mask)
pred_mask = (pred_mask > 0.5)*1.0

show_image(image, mask, pred_mask.detach().cpu().squeeze(0))