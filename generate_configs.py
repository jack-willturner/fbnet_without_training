import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#import torchvision.models as models
from utils import *
from tqdm import tqdm
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
args = parser.parse_args()

# Data loading code
traindir = os.path.join(args.data, 'train')
valdir = os.path.join(args.data, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.9, scale=(0.02, 1/3))
    ]))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    num_workers=args.workers, pin_memory=True, sampler=train_sampler)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=False,
    sampler= RepeatSampler(torch.utils.data.sampler.SubsetRandomSampler(range(len(train_data))), args.repeat))

configs = []
scores = []
for i in tqdm(range(100)):
    config = gen_random_net_config()
    network = FBNet(config)

    ### do scorey thing
    data_iterator = iter(score_train_loader)
    x, target = next(data_iterator)
    #x, target = x.to(device), target.to(device)
    network = network
    jacobs, labels = get_batch_jacobian(network, x, target, device, args)
    score = corrdistintegral_eval_score(jacobs)

    del network
    del x
    del target
    configs.append(config)
    scores.append(score)

print(config)
config = configs[np.argmax(scores)]
model = FBNet(config)

torch.save(config, f'checkpoints/seed_{args.seed}.t7')
