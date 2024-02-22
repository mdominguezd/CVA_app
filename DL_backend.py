import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage import io
import torchvision.transforms as T
from torchvision import tv_tensors
from torch.utils.data import Dataset
# from torchmetrics import F1Score
import sys
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

# These values were calculated using the histograms on notebook 01_02_DataDistributionShift.ipynb
oneperc_CIV = [217.0,	528.0,	389.0,	2162.0]
ninenine_CIV = [542.0,	896.0,	984.0,	3877.0]

oneperc_TNZ = [209.0, 483.35, 335.0, 2560.0]
ninenine_TNZ = [416.0, 723.65, 751.0, 3818.0]

class Img_Dataset(Dataset):
    """Specially adapted for dashboard"""
    def __init__(self, img_folder, transform = None, norm = 'Linear_1_99', VI = True, domain = 'target'):
        self.img_folder = img_folder
        self.transform = transform
        self.domain = domain
        self.norm = norm
        self.VI = VI

    def __len__(self):
        return sum(['crop' in i for i in os.listdir(self.img_folder)])

    def plot_imgs(self, idx):
        fig, ax = plt.subplots(1,1,figsize = (6,6))

        im = self.__getitem__(idx)

        ax.imshow(im[[2,1,0],:,:].permute(1,2,0))
        ax.set_title('Planet image\nRGB')

        return ax

    def __getitem__(self, idx):
        #__getitem__ asks for the sample number idx.

        conversion = T.ToTensor()
        
        if 'test' in self.img_folder:
            img = io.imread(fname = self.img_folder + '/crop_{:03d}'.format(idx) + '.tif').astype(np.float32)

        if self.VI:
            ndvi = (img[:,:,3] - img[:,:,2])/(img[:,:,3] + img[:,:,2])
            ndwi = (img[:,:,1] - img[:,:,3])/(img[:,:,3] + img[:,:,1])

        if self.norm == 'Linear_1_99':
            if self.domain == 'target':
                for i in range(img.shape[-1]):
                    img[:,:,i] = (img[:,:,i] - oneperc_TNZ[i])/(ninenine_TNZ[i] - oneperc_TNZ[i])
            else:
                for i in range(img.shape[-1]):
                    img[:,:,i] = (img[:,:,i] - oneperc_CIV[i])/(ninenine_CIV[i] - oneperc_CIV[i])

        if self.VI:
            ndvi = np.expand_dims(ndvi, axis = 2)
            ndwi = np.expand_dims(ndwi, axis = 2)
            img = np.concatenate((img, ndvi, ndwi), axis = 2)

        img = conversion(img).float()

        img = tv_tensors.Image(img)

        if self.transform != None:
            img = self.transform(img)

        return img

def predict_cashew(DS, model = 'Target-only'):
    
    if model == 'Target-only':
        model = torch.load('models_trained/Tanzania.pt', map_location=torch.device('cpu'))
        model.eval()
    elif model == 'Source-only':
        model = torch.load('models_trained/IvoryCoast.pt', map_location=torch.device('cpu'))
        model.eval()
    elif model == 'DANN':
        model = torch.load('models_trained/DANN.pt', map_location=torch.device('cpu'))
        model.eval()

    colors = [(0,0,0,0.01), (0,0.9,0.9,0.4)]
    
    cmap = LinearSegmentedColormap.from_list('name', colors, N=2)

    fig, ax = plt.subplots(3,3,figsize = (20, 20))
    
    for i in range(9):
        
        img = DS.__getitem__(i)[None,:,:,:].to('cpu')

        # im = io.imread(fname = 'test/crop_{:03d}'.format(i) + '.tif').astype(np.float32)
        im = torch.permute(img, (0,2,3,1))[0].cpu().numpy()
        print(im.shape)
        
        # if model == 'Target-only':
        #     for i in range(im.shape[-1]):
        #         im[:,:,i] = (im[:,:,i] - oneperc_TNZ[i])/(ninenine_TNZ[i] - oneperc_TNZ[i])
        # else:
        #     for i in range(im.shape[-1]):
        #         im[:,:,i] = (im[:,:,i] - oneperc_CIV[i])/(ninenine_CIV[i] - oneperc_CIV[i])

        # im[im>1] = 0.9999
        # im[im<0] = 0.0001

        # print(im)

        preds = model(img)[0].max(0)[1].to('cpu')
            
        ax[round(9*(i/3 - i//3))//3, i//3].imshow(im[:,:,[2,1,0]])
        ax[round(9*(i/3 - i//3))//3, i//3].imshow(preds, cmap = cmap)

        ax[round(9*(i/3 - i//3))//3, i//3].tick_params(axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=False) # labels along the bottom edge are off

        ax[round(9*(i/3 - i//3))//3, i//3].tick_params(axis='y',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        left=False,      # ticks along the bottom edge are off
                        labelleft=False) # labels along the bottom edge are off
        
        # ax[round(9*(i/3 - i//3))//3, i//3].set_title('Predictions for image '+str(i))
        
    fig = plt.gcf()

    plt.suptitle('Predictions for Planet image', y = 1.05, fontsize = 45)

    plt.tight_layout()
    
    st.pyplot(fig)



