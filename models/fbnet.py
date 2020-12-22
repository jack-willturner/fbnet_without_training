import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        assert C % g == 0, "Incompatible group size {} for input channel {}".format(
            g, C
        )
        return (
            x.view(N, g, int(C / g), H, W)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(N, C, H, W)
        )

class K3E1(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(K3E1, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.skip_con = (in_planes == self.expansion*planes) and (stride==1)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.skip_con:
            out += x
        return out

class K3E1G2(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(K3E1G2, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False, groups=2)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False, groups=2)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.shuffle = ChannelShuffle(groups=2)

        self.skip_con = (in_planes == self.expansion*planes) and (stride==1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.shuffle(out)

        if self.skip_con:
            out += x
        return out

class K3E3(nn.Module):
    expansion = 3
    def __init__(self, in_planes, planes, stride=1):
        super(K3E3, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.skip_con = (in_planes == self.expansion*planes) and (stride==1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.skip_con:
            out += x
        return out

class K3E6(nn.Module):
    expansion = 6

    def __init__(self, in_planes, planes, stride=1):
        super(K3E6, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.skip_con = (in_planes == self.expansion*planes) and (stride==1)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.skip_con:
            out += x
        return out

class K5E1(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(K5E1, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=5, stride=stride, padding=2, bias=False, groups=planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.skip_con = (in_planes == self.expansion*planes) and (stride==1)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.skip_con:
            out += x
        return out

class K5E1G2(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(K5E1G2, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False, groups=2)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=5, stride=stride, padding=2, bias=False, groups=planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False, groups=2)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.shuffle = ChannelShuffle(groups=2)

        self.skip_con = (in_planes == self.expansion*planes) and (stride==1)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.shuffle(out)
        if self.skip_con:
            out += x
        return out

class K5E3(nn.Module):
    expansion = 3
    def __init__(self, in_planes, planes, stride=1):
        super(K5E3, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=5, stride=stride, padding=2, bias=False, groups=planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.skip_con = (in_planes == self.expansion*planes) and (stride==1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.skip_con:
            out += x
        return out

class K5E6(nn.Module):
    expansion = 6
    def __init__(self, in_planes, planes, stride=1):
        super(K5E6, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=5, stride=stride, padding=2, bias=False, groups=planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.skip_con = (in_planes == self.expansion*planes) and (stride==1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.skip_con:
            out += x
        return out

class Skip(nn.Module):
    def __init__(self,in_planes=None, planes=None, stride=None):
        super(Skip, self).__init__()
    def forward(self, x):
        return x

class FBNet(nn.Module):
    def __init__(self, configs=None, num_classes=1000):
        super(FBNet, self).__init__()
        self.in_planes = 16
        self.configs = configs

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16,  1, configs[0], stride=1)
        self.layer2 = self._make_layer(24,  4, configs[1], stride=2)
        self.layer3 = self._make_layer(32,  4, configs[2], stride=2)
        self.layer4 = self._make_layer(64,  4, configs[3], stride=2)
        self.layer5 = self._make_layer(112, 4, configs[4], stride=1)
        self.layer6 = self._make_layer(184, 4, configs[5], stride=2)
        self.layer7 = self._make_layer(352, 1, configs[6], stride=1)

        self.conv2 = nn.Conv2d(352, 1504, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(1504)

        self.dropout = nn.Dropout(0.2)

        self.linear = nn.Linear(1504, num_classes)

    def _make_layer(self, planes, num_blocks, blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        most_recent_non_skip = 0

        for i, stride in enumerate(strides):
            layers.append(blocks[i](self.in_planes, planes, stride))

            if blocks[i] == Skip:
                self.in_planes = planes * blocks[most_recent_non_skip].expansion
            else:
                most_recent_non_skip = i
                self.in_planes = planes * blocks[i].expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        ints = []
        out = F.relu(self.bn1(self.conv1(x)))
        #ints.append(out)
        i = 0

        for layer in self.layer1:
            out = layer(out)
            #print(i)
            i += 1
            #ints.append(out)

        for layer in self.layer2:
            out = layer(out)
            #print(i)
            i += 1
            #ints.append(out)

        for layer in self.layer3:
            out = layer(out)
            #print(i)
            i += 1
            #ints.append(out)

        for layer in self.layer4:
            out = layer(out)
            #print(i)
            i += 1
            #ints.append(out)

        for layer in self.layer5:
            out = layer(out)
            #print(i)
            i += 1
            #ints.append(out)

        for layer in self.layer6:
            out = layer(out)
            #print(i)
            i += 1
            #ints.append(out)

        for layer in self.layer7:
            out = layer(out)
            #print(i)
            i += 1
            #ints.append(out)

        out = F.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        out = F.avg_pool2d(out, 7)
        #logits = out

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out #, logits, ints


def test():
    import numpy as np
    from tqdm import tqdm
    tot = 100
    invalid = 0
    for j in tqdm(range(tot)):
        try:
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

            net = FBNet(configs)
            out, logits, ints = net(torch.randn(1,3,224,224))
            #print(out.size())

        except:
            invalid += 1

    print(f"total invalid configs: {invalid} / {tot}")
#test()
