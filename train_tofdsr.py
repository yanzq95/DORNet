import argparse

from net.dornet_ddp import Net

from data.tofsr_dataloader import *
from utils import tofdsr_calc_rmse

from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, utils
import torch.optim as optim

import random

from net.CR import *
from tqdm import tqdm
import logging
from datetime import datetime
import os

parser = argparse.ArgumentParser()

parser.add_argument("--local-rank", default=-1, type=int)

parser.add_argument('--scale', type=int, default=4, help='scale factor')
parser.add_argument('--lr', default='0.0002', type=float, help='learning rate')  # 0.0001
parser.add_argument('--tiny_model', action='store_true', help='tiny model')
parser.add_argument('--epoch', default=300, type=int, help='max epoch')
parser.add_argument('--device', default="0,1", type=str, help='which gpu use')
parser.add_argument("--decay_iterations", type=list, default=[1.2e5, 2e5, 3.6e5],
                    help="steps to start lr decay")
parser.add_argument("--gamma", type=float, default=0.2, help="decay rate of learning rate")
parser.add_argument("--root_dir", type=str, default='./dataset/TOFDSR', help="root dir of dataset")
parser.add_argument("--batchsize", type=int, default=3, help="batchsize of training dataloader")
parser.add_argument("--num_gpus", type=int, default=2, help="num_gpus")
parser.add_argument('--seed', type=int, default=7240, help='random seed point')
parser.add_argument("--result_root", type=str, default='experiment/TOFDSR', help="root dir of dataset")
parser.add_argument("--blur_sigma", type=int, default=3.6, help="blur_sigma")
parser.add_argument('--isNoisy', action='store_true', help='Noisy')

opt = parser.parse_args()
print(opt)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')
device = torch.device("cuda", local_rank)

s = datetime.now().strftime('%Y%m%d%H%M%S')
dataset_name = opt.root_dir.split('/')[-1]

rank = dist.get_rank()

logging.basicConfig(filename='%s/train.log' % opt.result_root, format='%(asctime)s %(message)s', level=logging.INFO)
logging.info(opt)

net = Net(tiny_model=opt.tiny_model).cuda()

data_transform = transforms.Compose([transforms.ToTensor()])

train_dataset = TOFDSR_Dataset(root_dir=opt.root_dir, train=True, txt_file="./data/TOFDSR_Train.txt", transform=data_transform,
                               isNoisy=opt.isNoisy, blur_sigma=opt.blur_sigma)
test_dataset = TOFDSR_Dataset(root_dir=opt.root_dir, train=False, txt_file="./data/TOFDSR_Test.txt", transform=data_transform,
                              isNoisy=opt.isNoisy, blur_sigma=opt.blur_sigma)

if torch.cuda.device_count() > 1:
    train_sampler = DistributedSampler(dataset=train_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=False, pin_memory=True, num_workers=8,
                              drop_last=True, sampler=train_sampler)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=8)

net = DistributedDataParallel(net, device_ids=[local_rank], output_device=int(local_rank), find_unused_parameters=True)

l1 = nn.L1Loss().to(device)

optimizer = optim.Adam(net.module.parameters(), lr=opt.lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.decay_iterations, gamma=opt.gamma)
net.train()

max_epoch = opt.epoch
num_train = len(train_dataloader)
best_rmse = 100.0
best_epoch = 0
for epoch in range(max_epoch):
    # ---------
    # Training
    # ---------
    train_sampler.set_epoch(epoch)
    net.train()
    running_loss = 0.0

    t = tqdm(iter(train_dataloader), leave=True, total=len(train_dataloader))

    for idx, data in enumerate(t):
        batches_done = num_train * epoch + idx
        optimizer.zero_grad()
        guidance, lr, gt = data['guidance'].to(device), data['lr'].to(device), data['gt'].to(device)

        restored, d_lr_, aux_loss, cl_loss = net(x_query=lr, rgb=guidance)

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
        running_loss_50 = running_loss

        if idx % 50 == 0:
            running_loss_50 /= 50
            t.set_description(
                '[train epoch:%d] loss: Rec_loss:%.8f DA_loss:%.8f CL_loss:%.8f' % (
                    epoch + 1, rec_loss.item(), da_loss.item(), cl_loss.item()))
            t.refresh()

    logging.info('epoch:%d iteration:%d running_loss:%.10f' % (epoch + 1, batches_done + 1, running_loss / num_train))

    # -----------
    # Validating
    # -----------
    if rank == 0:
        with torch.no_grad():

            net.eval()
            rmse = np.zeros(560)
            t = tqdm(iter(test_dataloader), leave=True, total=len(test_dataloader))

            for idx, data in enumerate(t):
                guidance, lr, gt, maxx, minn = data['guidance'].to(device), data['lr'].to(device), data['gt'].to(
                    device), data[
                    'max'].to(device), data['min'].to(device)
                out, _ = net.module.srn(x_query=lr, rgb=guidance)
                minmax = [maxx, minn]
                rmse[idx] = tofdsr_calc_rmse(gt[0, 0], out[0, 0], minmax)
                t.set_description('[validate] rmse: %f' % rmse[:idx + 1].mean())
                t.refresh()

            r_mean = rmse.mean()
            if r_mean < best_rmse:
                best_rmse = r_mean
                best_epoch = epoch
                torch.save(net.module.srn.state_dict(),
                           os.path.join(opt.result_root, "RMSE%f_8%d.pth" % (best_rmse, best_epoch + 1)))
            logging.info(
                '---------------------------------------------------------------------------------------------------------------------------')
            logging.info('epoch:%d lr:%f-------mean_rmse:%f (BEST: %f @epoch%d)' % (
                epoch + 1, scheduler.get_last_lr()[0], r_mean, best_rmse, best_epoch + 1))
            logging.info(
                '---------------------------------------------------------------------------------------------------------------------------')

