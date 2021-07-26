import os

import numpy as np
import torch
import torch.nn as nn

import natsort

from skimage.color import rgb2gray

"""
import resize to resize image
"""
from skimage.transform import resize

"""
import tools for data augmentation
"""
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.filters import gaussian

import imageio

# Data Loader
class CustomDataset(torch.utils.data.Dataset):
  def __init__(self, data_dir, transform=None, train=False):#fdir, pdir, sdir, transform=None):
    self.r_dir = os.path.join(data_dir,'rock/')
    self.p_dir = os.path.join(data_dir,'paper/')
    self.s_dir = os.path.join(data_dir,'scissors/')

    self.transform = transform
    self.train = train

    lst_r = os.listdir(self.r_dir)
    lst_p = os.listdir(self.p_dir)
    lst_s = os.listdir(self.s_dir)

    lst_r = [f for f in lst_r if f.endswith(".png")]
    lst_p = [f for f in lst_p if f.endswith(".png")]
    lst_s = [f for f in lst_s if f.endswith(".png")]

    self.lst_dir = [self.r_dir] * len(lst_r) + [self.p_dir] * len(lst_p) + [self.s_dir] * len(lst_s)
    self.lst_prs = natsort.natsorted(lst_r) + natsort.natsorted(lst_p) + natsort.natsorted(lst_s)
 
  def __len__(self):
    return len(self.lst_prs)

  def __getitem__(self, index): 
    self.img_dir = self.lst_dir[index]
    self.img_name = self.lst_prs[index]

    return [self.img_dir, self.img_name] 
    
  def custom_collate_fn(self, data):

    inputImages = []
    outputVectors = []
    trans = AffineTransform(translation=(9, 10))

    for sample in data:
      prs_img = imageio.imread(os.path.join(sample[0] + sample[1]))
      gray_img = rgb2gray(prs_img)
      # resize image
      gray_img = resize(gray_img, (89, 100))
      if self.train:

        # rotate image
        rotated = rotate(gray_img, angle=45, mode='wrap')
        # translate image
        wrapShift = warp(gray_img, trans, mode='wrap')
        # flip image left-to-right
        flipLR = np.fliplr(gray_img)
        # flip image up-to-down
        # flipUD = np.flipud(gray_img)
        # add noise
        # sd = 0.155
        # noisyRandom = random_noise(gray_img, var=sd**2)
        # blurring
        # blurred = gaussian(gray_img, sigma=1, multichannel=True)

      if gray_img.ndim == 2:
        gray_img = gray_img[:, :, np.newaxis]

      inputImages.append(gray_img.reshape(89, 100, 1))

      dir_split = sample[0].split('/')
      if dir_split[-2] == 'rock':
        outputVectors.append(np.array(0))
      elif dir_split[-2] == 'paper':
        outputVectors.append(np.array(1))
      elif dir_split[-2] == 'scissors':
        outputVectors.append(np.array(2))
     
      if self.train:
          # add rotated
          if rotated.ndim == 2:
            rotated = rotated[:, :, np.newaxis]

          inputImages.append(rotated.reshape(89, 100, 1))

          dir_split = sample[0].split('/')
          if dir_split[-2] == 'rock':
            outputVectors.append(np.array(0))
          elif dir_split[-2] == 'paper':
            outputVectors.append(np.array(1))
          elif dir_split[-2] == 'scissors':
            outputVectors.append(np.array(2))

          # add wrapShift
          if wrapShift.ndim == 2:
            wrapShift = wrapShift[:, :, np.newaxis]

          inputImages.append(wrapShift.reshape(89, 100, 1))

          dir_split = sample[0].split('/')
          if dir_split[-2] == 'rock':
            outputVectors.append(np.array(0))
          elif dir_split[-2] == 'paper':
            outputVectors.append(np.array(1))
          elif dir_split[-2] == 'scissors':
            outputVectors.append(np.array(2))
          # add flipLR
          if flipLR.ndim == 2:
            flipLR = flipLR[:, :, np.newaxis]

          inputImages.append(flipLR.reshape(89, 100, 1))

          dir_split = sample[0].split('/')
          if dir_split[-2] == 'rock':
            outputVectors.append(np.array(0))
          elif dir_split[-2] == 'paper':
            outputVectors.append(np.array(1))
          elif dir_split[-2] == 'scissors':
            outputVectors.append(np.array(2))


          
    # if not self.train:
    #     print(len(inputImages))
    # print(f"train: {self.train}, length: {len(inputImages)}")
    data = {'input': inputImages, 'label': outputVectors}

    if self.transform:
      data = self.transform(data)

    return data


class ToTensor(object):
  def __call__(self, data):
    label, input = data['label'], data['input']

    input_tensor = torch.empty(len(input),89,100)
    label_tensor = torch.empty(len(input))
    for i in range(len(input)):
      input[i] = input[i].transpose((2, 0, 1)).astype(np.float32)
      input_tensor[i] = torch.from_numpy(input[i])
      label_tensor[i] = torch.from_numpy(label[i])
    input_tensor = torch.unsqueeze(input_tensor, 1)
    
    data = {'label': label_tensor.long(), 'input': input_tensor}

    return data

