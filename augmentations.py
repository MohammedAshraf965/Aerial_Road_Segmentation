import albumentations as A

def get_train_augs(IMG_DIM):
  # Defines a series of the augmentation needed
  return A.Compose([
      A.Resize(IMG_DIM, IMG_DIM),
      A.HorizontalFlip(p = 0.5),   # 50% rotation chance
      A.VerticalFlip(p = 0.5)
  ])

def get_valid_augs(IMG_DIM):
  # If validation or testing is done, no augmentation is applied except resize
  return A.Compose([
      A.Resize(IMG_DIM, IMG_DIM)
  ])