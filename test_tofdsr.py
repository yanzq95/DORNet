import argparse

from utils import *
import torchvision.transforms as transforms

from net.dornet_ddp import Net

from data.tofdc_dataloader import *

import os

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=int, default=4, help='scale factor')
parser.add_argument("--root_dir", type=str, default='/opt/data/private/dataset', help="root dir of dataset")
parser.add_argument("--model_dir", type=str, default="./checkpoints/TOFDSR.pth", help="path of net")
parser.add_argument("--results_dir", type=str, default='./results/', help="root dir of results")
parser.add_argument('--tiny_model', action='store_true', help='tiny model')
parser.add_argument("--blur_sigma", type=int, default=3.6, help="blur_sigma")
parser.add_argument('--isNoisy', action='store_true', help='Noisy')

opt = parser.parse_args()

net = Net(tiny_model=opt.tiny_model).srn.cuda()

net.load_state_dict(torch.load(opt.model_dir, map_location='cuda:0'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

data_transform = transforms.Compose([transforms.ToTensor()])

dataset_name = opt.root_dir.split('/')[-1]

dataset = TOFDSR_Dataset(root_dir=opt.root_dir, train=False, txt_file="./data/TOFDSR_Test.txt", transform=data_transform, isNoisy=opt.isNoisy, blur_sigma=opt.blur_sigma)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

data_num = len(dataloader)
rmse = np.zeros(data_num)

with torch.no_grad():
    net.eval()

    for idx, data in enumerate(dataloader):
        guidance, lr, gt, maxx, minn, name = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda(), data[
            'max'].cuda(), data['min'].cuda(), data['name']
        out, _ = net(x_query=lr, rgb=guidance)
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
    print("=========TOFDSR==========")
    print(rmse.mean())
    print("=========TOFDSR==========")


