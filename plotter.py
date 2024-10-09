import matplotlib.pyplot as plt
import matplotlib.image as img

def show_image(image, mask, mask_pred = None):
    
    if mask_pred == None:

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        ax1.set_title('IMAGE')
        ax1.imshow(image.permute(1, 2, 0).squeeze(), cmap='gray')

        ax2.set_title('GROUND TRUTH')
        ax2.imshow(mask.permute(1, 2, 0).squeeze(), cmap='gray')

    elif mask_pred != None:

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
        
        ax1.set_title('IMAGE')
        ax1.imshow(image.permute(1, 2, 0).squeeze(), cmap='gray')

        ax2.set_title('GROUND TRUTH')
        ax2.imshow(mask.permute(1, 2, 0).squeeze(), cmap='gray')

        ax3.set_title('MODEL PREDICTION')
        ax3.imshow(mask_pred.permute(1, 2, 0).squeeze(), cmap='gray')
