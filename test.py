import argparse

from utils import *
import torchvision.transforms as transforms

from net.dornet import Net

from data.nyu_dataloader import *
from data.rgbdd_dataloader import *
from data.tofdc_dataloader import *

import os

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=int, default=4, help='scale factor')
parser.add_argument("--root_dir", type=str, default='./dataset/NYU-v2', help="root dir of dataset")
parser.add_argument("--model_dir", type=str, default="./checkpoints/RGBDD.pth", help="path of net")
parser.add_argument("--results_dir", type=str, default='./results/', help="root dir of results")
parser.add_argument('--tiny_model', action='store_true', help='tiny model')

opt = parser.parse_args()

net = Net(tiny_model=opt.tiny_model).cuda()

print("*********************************************")
print(sum(p.numel() for p in net.parameters() if p.requires_grad))
print("*********************************************")
net.load_state_dict(torch.load(opt.model_dir, map_location='cuda:0'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

data_transform = transforms.Compose([transforms.ToTensor()])

dataset_name = opt.root_dir.split('/')[-1]

if dataset_name == 'RGB-D-D':
    dataset = RGBDD_Dataset(root_dir=opt.root_dir, scale=opt.scale, downsample='real', train=False,
                            transform=data_transform, isNoisy=False, blur_sigma=3.6)
    rmse = np.zeros(405)
elif dataset_name == 'TOFDSR':
    dataset = TOFDSR_Dataset(root_dir=opt.root_dir, scale=opt.scale, downsample='real', train=False,
                             txt_file="./data/TOFDSR_Test.txt", transform=data_transform, isNoisy=False, blur_sigma=3.6)
    rmse = np.zeros(560)
elif dataset_name == 'NYU-v2':
    dataset = NYU_v2_datset(root_dir=opt.root_dir, scale=opt.scale, transform=data_transform, train=False)
    test_minmax = np.load('%s/test_minmax.npy' % opt.root_dir)
    rmse = np.zeros(449)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
data_num = len(dataloader)

with torch.no_grad():
    net.eval()
    if dataset_name == 'RGB-D-D':
        for idx, data in enumerate(dataloader):
            guidance, lr, gt, maxx, minn, name = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda(), data[
                'max'].cuda(), data['min'].cuda(), data['name']
            out = net(x_query=lr, rgb=guidance)
            rmse[idx] = rgbdd_calc_rmse(gt[0, 0], out[0, 0], [maxx, minn])

            # Save results  (Save the output depth map)
            # path_output = '{}/output'.format(opt.results_dir)
            # os.makedirs(path_output, exist_ok=True)
            # path_save_pred = '{}/{}.png'.format(path_output, name[0])

            # pred = out[0, 0] * (maxx - minn) + minn
            # pred = pred.cpu().detach().numpy()
            # pred = pred.astype(np.uint16)
            # pred = Image.fromarray(pred)
            # pred.save(path_save_pred)

            print('idx:%d RMSE:%f' % (idx + 1, rmse[idx]))
        print("==========RGB-D-D=========")
        print(rmse.mean())
        print("==========RGB-D-D=========")
    elif dataset_name == 'TOFDSR':
        for idx, data in enumerate(dataloader):
            guidance, lr, gt, maxx, minn, name = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda(), data[
                'max'].cuda(), data['min'].cuda(), data['name']
            out = net(x_query=lr, rgb=guidance)
            rmse[idx] = tofdsr_calc_rmse(gt[0, 0], out[0, 0], [maxx, minn])

            # Save results  (Save the output depth map)
            # path_output = '{}/output'.format(opt.results_dir)
            # os.makedirs(path_output, exist_ok=True)
            # path_save_pred = '{}/{}.png'.format(path_output, name[0])

            # pred = out[0, 0] * (maxx - minn) + minn
            # pred = pred.cpu().detach().numpy()
            # pred = pred.astype(np.uint16)
            # pred = Image.fromarray(pred)
            # pred.save(path_save_pred)

            print('idx:%d RMSE:%f' % (idx + 1, rmse[idx]))
        print("==========TOFDSR=========")
        print(rmse.mean())
        print("==========TOFDSR=========")
    elif dataset_name == 'NYU-v2':
        t = np.zeros(449)
        for idx, data in enumerate(dataloader):
            guidance, lr, gt = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda()
            out = net(x_query=lr, rgb=guidance)

            minmax = test_minmax[:, idx]
            minmax = torch.from_numpy(minmax).cuda()
            rmse[idx] = calc_rmse(gt[0, 0], out[0, 0], minmax)

            # Save results  (Save the output depth map)
            # path_output = '{}/output'.format(opt.results_dir)
            # os.makedirs(path_output, exist_ok=True)
            # path_save_pred = '{}/{:010d}.png'.format(path_output, idx)

            # pred = out[0,0] * (minmax[0] - minmax[1]) + minmax[1]
            # pred = pred * 1000.0
            # pred = pred.cpu().detach().numpy()
            # pred = pred.astype(np.uint16)
            # pred = Image.fromarray(pred)
            # pred.save(path_save_pred)

            print('idx:%d RMSE:%f' % (idx + 1, rmse[idx]))
        print("=========NYU-v2==========")
        print(rmse.mean())
        print("=========NYU-v2==========")


