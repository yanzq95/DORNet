import argparse
from net.dornet import Net
from net.CR import *

from data.rgbdd_dataloader import *
from data.nyu_dataloader import *
from data.tofdc_dataloader import *
from utils import calc_rmse, rgbdd_calc_rmse, tofdsr_calc_rmse
 
from torch.utils.data import Dataset
from torchvision import transforms, utils
import torch
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm
import logging
from datetime import datetime
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=int, default=4, help='scale factor')
parser.add_argument('--lr', default='0.0001', type=float, help='learning rate')
parser.add_argument('--result', default='experiment', help='learning rate')
parser.add_argument('--tiny_model', action='store_true', help='tiny model')
parser.add_argument('--epoch', default=300, type=int, help='max epoch')
parser.add_argument("--decay_iterations", type=list, default=[1.2e5, 2e5, 3.6e5],
                    help="steps to start lr decay")
parser.add_argument("--gamma", type=float, default=0.2, help="decay rate of learning rate")
parser.add_argument("--root_dir", type=str, default='./dataset/RGB-D-D', help="root dir of dataset")
parser.add_argument("--batch_size", type=int, default=3, help="batch_size of training dataloader")
parser.add_argument("--blur_sigma", type=int, default=3.6, help="blur_sigma")

opt = parser.parse_args()
print(opt)

s = datetime.now().strftime('%Y%m%d%H%M%S')
dataset_name = opt.root_dir.split('/')[-1]
result_root = '%s/%s-lr_%s-s_%s-%s-b_%s' % (opt.result, s, opt.lr, opt.scale, dataset_name, opt.batch_size)
if not os.path.exists(result_root):
    os.mkdir(result_root)

logging.basicConfig(filename='%s/train.log' % result_root, format='%(asctime)s %(message)s', level=logging.INFO)
logging.info(opt)

net = Net(tiny_model=opt.tiny_model).cuda()

print("*********************************************")
print(sum(p.numel() for p in net.parameters() if p.requires_grad))
print("*********************************************")
net.train()

optimizer = optim.Adam(net.parameters(), lr=opt.lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.decay_iterations, gamma=opt.gamma)

CL = ContrastLoss(ablation=False)
l1 = nn.L1Loss().cuda()

data_transform = transforms.Compose([transforms.ToTensor()])


if dataset_name == 'RGB-D-D':
    train_dataset = RGBDD_Dataset(root_dir=opt.root_dir, scale=opt.scale, downsample='real', train=True,
                                  transform=data_transform, isNoisy=False, blur_sigma=opt.blur_sigma)
    test_dataset = RGBDD_Dataset(root_dir=opt.root_dir, scale=opt.scale, downsample='real', train=False,
                                 transform=data_transform, isNoisy=False, blur_sigma=opt.blur_sigma)
elif dataset_name == 'TOFDSR':
    train_dataset = TOFDSR_Dataset(root_dir=opt.root_dir, scale=opt.scale, downsample='real', train=True,
                                   txt_file="./data/TOFDSR_Train.txt", transform=data_transform,
                                   isNoisy=False, blur_sigma=3.6)
    test_dataset = TOFDSR_Dataset(root_dir=opt.root_dir, scale=opt.scale, downsample='real', train=False,
                                  txt_file="./data/TOFDSR_Test.txt", transform=data_transform,
                                  isNoisy=False, blur_sigma=3.6)
elif dataset_name == 'NYU-v2':
    test_minmax = np.load('%s/test_minmax.npy' % opt.root_dir)
    train_dataset = NYU_v2_datset(root_dir=opt.root_dir, scale=opt.scale, transform=data_transform, train=True)
    test_dataset = NYU_v2_datset(root_dir=opt.root_dir, scale=opt.scale, transform=data_transform, train=False)


train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

max_epoch = opt.epoch
num_train = len(train_dataloader)
best_rmse = 100.0
best_epoch = 0
for epoch in range(max_epoch):
    # ---------
    # Training
    # ---------
    net.train()
    running_loss = 0.0

    t = tqdm(iter(train_dataloader), leave=True, total=len(train_dataloader))

    for idx, data in enumerate(t):
        if dataset_name == 'TOFDSR':
            batches_done = num_train * epoch + idx
            optimizer.zero_grad()
            guidance, lr, gt = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda()

            restored, d_lr_, aux_loss = net(x_query=lr, rgb=guidance)

            cl_loss = CL(d_lr_, lr, restored)

            mask = (gt >= 0.02) & (gt <= 1)
            gt = gt[mask]
            restored = restored[mask]
            lr = lr[mask]
            d_lr_ = d_lr_[mask]

            rec_loss = l1(restored, gt)
            da_loss = l1(d_lr_, lr)
            loss = rec_loss + 0.1 * da_loss + 0.1 * cl_loss + aux_loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.data.item()

            t.set_description(
                '[train epoch:%d] loss: Rec_loss:%.8f DA_loss:%.8f CL_loss:%.8f' % (epoch + 1, rec_loss.item(), da_loss.item(), cl_loss.item()))
            t.refresh()
        else:
            batches_done = num_train * epoch + idx
            optimizer.zero_grad()
            guidance, lr, gt = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda()

            restored, d_lr_, aux_loss = net(x_query=lr, rgb=guidance)

            rec_loss = l1(restored, gt)
            da_loss = l1(d_lr_, lr)
            cl_loss = CL(d_lr_, lr, restored)
            loss = rec_loss + 0.1 * da_loss + 0.1 * cl_loss + aux_loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.data.item()

            t.set_description(
                '[train epoch:%d] loss: Rec_loss:%.8f DA_loss:%.8f CL_loss:%.8f' % (
                epoch + 1, rec_loss.item(), da_loss.item(), cl_loss.item()))
            t.refresh()

    logging.info('epoch:%d iteration:%d running_loss:%.10f' % (epoch + 1, batches_done + 1, running_loss / num_train))


    # -----------
    # Validating
    # -----------
    with torch.no_grad():

        net.eval()
        if dataset_name == 'RGB-D-D':
            rmse = np.zeros(405)
        elif dataset_name == 'TOFDSR':
            rmse = np.zeros(560)
        elif dataset_name == 'NYU-v2':
            rmse = np.zeros(449)
        t = tqdm(iter(test_dataloader), leave=True, total=len(test_dataloader))

        for idx, data in enumerate(t):
            if dataset_name == 'RGB-D-D':
                guidance, lr, gt, max, min = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda(), data[
                    'max'].cuda(), data['min'].cuda()
                out = net(x_query=lr, rgb=guidance)
                minmax = [max, min]
                rmse[idx] = rgbdd_calc_rmse(gt[0, 0], out[0, 0], minmax)
                t.set_description('[validate] rmse: %f' % rmse[:idx + 1].mean())
                t.refresh()
            elif dataset_name == 'TOFDSR':
                guidance, lr, gt, max, min = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda(), data[
                    'max'].cuda(), data['min'].cuda()
                out = net(x_query=lr, rgb=guidance)
                minmax = [max, min]
                rmse[idx] = tofdsr_calc_rmse(gt[0, 0], out[0, 0], minmax)
                t.set_description('[validate] rmse: %f' % rmse[:idx + 1].mean())
                t.refresh()
            elif dataset_name == 'NYU-v2':
                guidance, lr, gt = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda()
                out = net(x_query=lr, rgb=guidance)
                minmax = test_minmax[:, idx]
                minmax = torch.from_numpy(minmax).cuda()
                rmse[idx] = calc_rmse(gt[0, 0], out[0, 0], minmax)
                t.set_description('[validate] rmse: %f' % rmse[:idx + 1].mean())
                t.refresh()
        r_mean = rmse.mean()
        if r_mean < best_rmse:
            best_rmse = r_mean
            best_epoch = epoch
            torch.save(net.state_dict(),
                       os.path.join(result_root, "modelbestRMSE%f_8%d.pth" % (r_mean, epoch + 1)))
        logging.info(
            '---------------------------------------------------------------------------------------------------------------------------')
        logging.info('epoch:%d lr:%f-------mean_rmse:%f (BEST: %f @epoch%d)' % (
            epoch + 1, scheduler.get_last_lr()[0], r_mean, best_rmse, best_epoch + 1))
        logging.info(
            '---------------------------------------------------------------------------------------------------------------------------')

