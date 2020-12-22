from __future__ import print_function

import numpy as np
from models import *

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

np.random.seed(0)

def fisher_score(net, train_loader):
    input, target = next(iter(train_loader))
    out, logits, ints = net(input)

    for int in ints:
        int.retain_grad()

    loss = criterion(logits, target)
    loss.backward()

    fishers = []

    for int in ints:
        fish = (int.data.detach() * int.grad.detach()).sum(-1).sum(-1).pow(2).mean(0).sum()
        fishers.append(fish)

    return fishers

def get_batch_jacobian(net, x, target, device, args=None):
    net.zero_grad()
    x.requires_grad_(True)
    y = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, target.detach()

def corrdistintegral_eval_score(jacob, upp=0.25):
    xx = jacob.reshape(jacob.size(0), -1).detach().cpu().numpy()
    corrs = np.corrcoef(xx)
    return np.logical_and(corrs < upp, corrs > 0).sum()

def gen_random_net_config():
    configs = []

    configs.append([np.random.choice([K3E1,K3E1G2,K3E3,K3E6,K5E1,K5E1G2,K5E3,K5E6])])

    for i in range(5):
        # first block can't be Skip
        c1 = np.random.choice([K3E1,K3E1G2,K3E3,K3E6,K5E1,K5E1G2,K5E3,K5E6])
        c2 = np.random.choice([K3E1,K3E1G2,K3E3,K3E6,K5E1,K5E1G2,K5E3,K5E6,Skip])
        c3 = np.random.choice([K3E1,K3E1G2,K3E3,K3E6,K5E1,K5E1G2,K5E3,K5E6,Skip])
        c4 = np.random.choice([K3E1,K3E1G2,K3E3,K5E1,K5E1G2])
        # last block can't have Skip or expansion factor > 1

        configs.append([c1,c2,c3,c4])

    # final block choice
    configs.append([np.random.choice([K3E1,K3E1G2,K5E1,K5E1G2])])

    return configs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
global error_history
error_history = []

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_cifar_loaders(data_loc='./data', batch_size=128, cutout=False, n_holes=1, length=16):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if cutout:
        transform_train.transforms.append(Cutout(n_holes=n_holes, length=length))

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=data_loc, train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=8)

    testset = torchvision.datasets.CIFAR10(root=data_loc, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    return trainloader, testloader

def load_model(model, sd):
    sd = torch.load('checkpoints/%s.t7' % sd, map_location='cpu')
    new_sd = model.state_dict()
    old_sd = sd['net']
    new_names = [v for v in new_sd]
    old_names = [v for v in old_sd]
    for i, j in enumerate(new_names):
        if not 'mask' in j:
            new_sd[j] = old_sd[old_names[i]]

    model.load_state_dict(new_sd)
    return model

def get_error(output, target, topk=(1,)):
    """Computes the error@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(correct[:k].numel()).float().sum(0, keepdim=True)
        res.append(100.0 - correct_k.mul_(100.0 / batch_size))
    return res

class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

## parameter counting
import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce
import operator

count_ops = 0
count_params = 0


def get_num_gen(gen):
    return sum(1 for x in gen)

def is_leaf(model):
    return get_num_gen(model.children()) == 0

def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name

def get_layer_param(model):
    return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])

### The input batch size should be 1 to call this function
def measure_layer(layer, x):
    global count_ops, count_params
    delta_ops = 0
    delta_params = 0
    multi_add = 1
    type_name = get_layer_info(layer)

    ### ops_conv
    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        delta_params = get_layer_param(layer)

    ### ops_learned_conv
    elif type_name in ['LearnedGroupConv']:
        measure_layer(layer.relu, x)
        measure_layer(layer.norm, x)
        conv = layer.conv
        out_h = int((x.size()[2] + 2 * conv.padding[0] - conv.kernel_size[0]) /
                    conv.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * conv.padding[1] - conv.kernel_size[1]) /
                    conv.stride[1] + 1)
        delta_ops = conv.in_channels * conv.out_channels * conv.kernel_size[0] * \
                conv.kernel_size[1] * out_h * out_w / layer.condense_factor * multi_add
        delta_params = get_layer_param(conv) / layer.condense_factor

    ### ops_nonlinearity
    elif type_name in ['ReLU','ReLU6','SwishMe']:
        delta_ops = x.numel()
        delta_params = get_layer_param(layer)

    ### ops_pooling
    elif type_name in ['AvgPool2d']:
        in_w = x.size()[2]
        kernel_ops = layer.kernel_size * layer.kernel_size
        out_w = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        out_h = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        delta_ops = x.size()[0] * x.size()[1] * out_w * out_h * kernel_ops
        delta_params = get_layer_param(layer)

    elif type_name in ['AdaptiveAvgPool2d','MaxPool2d']:
        delta_ops = x.size()[0] * x.size()[1] * x.size()[2] * x.size()[3]
        delta_params = get_layer_param(layer)

    ### ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = layer.bias.numel()
        delta_ops = x.size()[0] * (weight_ops + bias_ops)
        delta_params = get_layer_param(layer)

    ### ops_nothing
    elif type_name in ['BatchNorm2d', 'Dropout2d', 'DropChannel', 'Dropout']:
        delta_params = get_layer_param(layer)

    elif type_name in ['Identity']:
        pass

    elif type_name in ['Sequential']:
        for sub_layer in layer:
            measure_layer(sub_layer, x)
    ### unknown layer type
    else:
        raise TypeError('unknown layer type: %s' % type_name)

    count_ops += delta_ops
    count_params += delta_params
    return


def measure_model(model, H, W):
    global count_ops, count_params
    count_ops = 0
    count_params = 0
    data = Variable(torch.zeros(1, 3, H, W))

    def should_measure(x):
        return is_leaf(x)

    def modify_forward(model):
        for child in model.children():
            if should_measure(child):
                def new_forward(m):
                    def lambda_forward(x):
                        measure_layer(m, x)
                        return m.old_forward(x)
                    return lambda_forward
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            # leaf node
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    model.forward(data)
    restore_forward(model)

    return count_ops, count_params
