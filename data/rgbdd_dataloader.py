import numpy as np
import os

from torch.utils.data import Dataset
from PIL import Image
from scipy.ndimage import gaussian_filter


class RGBDD_Dataset(Dataset):
    """RGB-D-D Dataset."""

    def __init__(self, root_dir="./dataset/RGB-D-D/", scale=4, downsample='real', train=True,
                 transform=None, isNoisy=False, blur_sigma=1.2):

        self.root_dir = root_dir
        self.transform = transform
        self.scale = scale
        self.downsample = downsample
        self.train = train
        self.isNoisy = isNoisy
        self.blur_sigma = blur_sigma

        types = ['models', 'plants', 'portraits']

        if train:
            if self.downsample == 'real':
                self.GTs = []
                self.LRs = []
                self.RGBs = []
                for type in types:
                    list_dir = os.listdir('%s/%s/%s_train'% (root_dir, type, type))
                    for n in list_dir:
                        self.RGBs.append('%s/%s/%s_train/%s/%s_RGB.jpg' % (root_dir, type, type, n, n))
                        self.GTs.append('%s/%s/%s_train/%s/%s_HR_gt.png' % (root_dir, type, type, n, n))
                        self.LRs.append('%s/%s/%s_train/%s/%s_LR_fill_depth.png' % (root_dir, type, type, n, n))
            else:
                self.GTs = []
                self.RGBs = []
                for type in types:
                    list_dir = os.listdir('%s/%s/%s_train'% (root_dir, type, type))
                    for n in list_dir:
                        self.RGBs.append('%s/%s/%s_train/%s/%s_RGB.jpg' % (root_dir, type, type, n, n))
                        self.GTs.append('%s/%s/%s_train/%s/%s_HR_gt.png' % (root_dir, type, type, n, n))

        else:
            if self.downsample == 'real':
                self.GTs = []
                self.LRs = []
                self.RGBs = []
                for type in types:
                    list_dir = os.listdir('%s/%s/%s_test'% (root_dir, type, type))
                    for n in list_dir:
                        self.RGBs.append('%s/%s/%s_test/%s/%s_RGB.jpg' % (root_dir, type, type, n, n))
                        self.GTs.append('%s/%s/%s_test/%s/%s_HR_gt.png' % (root_dir, type, type, n, n))
                        self.LRs.append('%s/%s/%s_test/%s/%s_LR_fill_depth.png' % (root_dir, type, type, n, n))
            else:
                self.GTs = []
                self.RGBs = []
                for type in types:
                    list_dir = os.listdir('%s/%s/%s_test'% (root_dir, type, type))
                    for n in list_dir:
                        self.RGBs.append('%s/%s/%s_test/%s/%s_RGB.jpg' % (root_dir, type, type, n, n))
                        self.GTs.append('%s/%s/%s_test/%s/%s_HR_gt.png' % (root_dir, type, type, n, n))

    def __len__(self):
        return len(self.GTs)

    def __getitem__(self, idx):
        if self.downsample == 'real':
            image = np.array(Image.open(self.RGBs[idx]).convert("RGB")).astype(np.float32)
            name = self.RGBs[idx][-22:-8]
            gt = np.array(Image.open(self.GTs[idx])).astype(np.float32)
            h, w = gt.shape
            lr = np.array(Image.open(self.LRs[idx]).resize((w, h), Image.BICUBIC)).astype(np.float32)
        else:
            image = Image.open(self.RGBs[idx]).convert("RGB")
            name = self.RGBs[idx][-22:-8]
            image = np.array(image).astype(np.float32)
            gt = Image.open(self.GTs[idx])
            w, h = gt.size
            s = self.scale
            lr = np.array(gt.resize((w // s, h // s), Image.BICUBIC).resize((w, h), Image.BICUBIC)).astype(np.float32)
            gt = np.array(gt).astype(np.float32)

        # normalization
        if self.train:
            max_out = 5000.0
            min_out = 0.0
            lr = (lr - min_out) / (max_out - min_out)
            gt = (gt-min_out)/(max_out-min_out)
        else:
            max_out = 5000.0
            min_out = 0.0
            lr = (lr - min_out) / (max_out - min_out)
            
        maxx = np.max(image)
        minn = np.min(image)
        image = (image - minn) / (maxx - minn)

        lr_minn = np.min(lr)
        lr_maxx = np.max(lr)

        if not self.train:
            np.random.seed(42)

        if self.isNoisy:
            lr = gaussian_filter(lr, sigma=self.blur_sigma)

            gaussian_noise = np.random.normal(0, 0.07, lr.shape)
            lr = lr + gaussian_noise
            lr = np.clip(lr, lr_minn, lr_maxx)

        image = self.transform(image).float()
        gt = self.transform(np.expand_dims(gt, 2)).float()
        lr = self.transform(np.expand_dims(lr, 2)).float()
        sample = {'guidance': image, 'lr': lr, 'gt': gt, 'max': max_out, 'min': min_out, 'name':name}

        return sample
