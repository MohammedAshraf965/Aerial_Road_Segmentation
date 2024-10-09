import cv2
import numpy as np
import tqdm
import torch
from torch import nn

from torch.utils.data import Dataset
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

##############################

class SegmentationDataset(Dataset):

  def __init__(self, df, augmentations, DATA_DIR):
    self.df = df
    self.augmentations = augmentations
    self.DATA_DIR = DATA_DIR

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):

    row = self.df.iloc[idx]

    image_path = self.DATA_DIR + row.images
    mask_path = self.DATA_DIR + row.masks

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # H x W
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # H x W x C
    mask = np.expand_dims(mask, axis=-1)

    # If the augmentations are true, apply them
    if self.augmentations:
      data = self.augmentations(image=image, mask=mask)        # Return is in dictionary format
      image = data['image']                                    # (H x W x C)
      mask = data['mask']

    image = np.transpose(image, (2, 0, 1)).astype(np.float32)  # (C x H x W)
    mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)

    image = torch.tensor(image) / 255.0
    mask = torch.round(torch.tensor(mask) / 255.0)             # Round to 0 or 1

    return image, mask
  
  #################################################
  
  class SegmentationModel(nn.Module):
    def __init__(self, ENCODER, WEIGHTS):
        super(SegmentationModel, self).__init__()
        self.backbone = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=WEIGHTS,
            in_channels=3,            # RGB
            classes=1,                 # Binary Segmentation problem (pixel is 0 or 1)
            activation=None           # Output will be logits (raw outputs with no sigmoid and softmax activation functions)
        )

    def forward(self, images, mask=None):

        logits = self.backbone(images)

        # Two loss functions:
        #   DiceLoss
        #   Binary Cross Entropy (BCE)
        # Calculating the difference between logits and mask
        if mask != None:
            return logits, DiceLoss(mode='binary')(logits, mask) + nn.BCEWithLogitsLoss()(logits, mask)

        # During testing time, the mask is none as we do not have the mask. Return only prediction (logits)
        return logits

###################################

def train_fn(data_loader, model, optimizer, DEVICE):

  model.train()   # Train on dropout, batchnorm, etc...

  total_loss = 0.0

  for images, masks in tqdm(data_loader):     #tqdm is useful to track the loop dataloader

    images = images.to(DEVICE)
    masks = masks.to(DEVICE)

    optimizer.zero_grad()                     # Initialize gradients to zero
    logits, loss = model(images, masks)       # Pass the images and masks to the model
    loss.backward()                           # Find the gradients
    optimizer.step()                          # Update the weights and the biases that are parameters of the model

    total_loss += loss.item()                 # Add per batch loss to total loss

  return total_loss / len(data_loader)        # Average the total loss by number of batches

##########################################

def eval_fn(data_loader, model, DEVICE):

  model.eval()   # Train off dropout, batchnorm, etc...

  total_loss = 0.0

  with torch.no_grad():

    for images, masks in tqdm(data_loader):     #tqdm is useful to track the loop dataloader

      images = images.to(DEVICE)
      masks = masks.to(DEVICE)
      logits, loss = model(images, masks)       # Pass the images and masks to the model
      total_loss += loss.item()                 # Add per batch loss to total loss

    return total_loss / len(data_loader)        # Average the total loss by number of batches