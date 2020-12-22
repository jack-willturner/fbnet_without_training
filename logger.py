import os
import torch
import pandas as pd
from models import *
from utils import *
from tqdm import tqdm

def test_config_rate():
    invalid = 0
    confs = 100

    x = torch.randn((1,3,32,32))
    for i in range(confs):
        try:
            net =  ResNet34(gen_random_net_config())
            _ = net(x)
        except Exception as e:
            print(e)
            invalid += 1

    print(f"{invalid}/{confs} were invalid")

def log_checkpoint_data():
    ckpts = os.listdir('checkpoints/')

    df =[]
    for ckpt in tqdm(ckpts):
        try:
            sd = torch.load(f"checkpoints/{ckpt}", map_location='cpu')
            err_hist = sd['error_history']
            configs = sd['configs']
            df.append([ckpt, configs, err_hist])
        except:
            print(f"Error processing {ckpt}")
    df = pd.DataFrame(df, columns=['name','configs','error_history'])

    df.to_pickle('checkpoints/results.ckpt')

def add_param_counts():
    df = pd.read_pickle('checkpoints/results.ckpt')
    configs = df['configs']

    parameters = []
    flops = []
    for config in tqdm(configs):
        try:
            net = ResNet34(config)
            x = torch.randn((1,3,32,32))
            ops, params = measure_model(net, 32, 32)
            parameters.append(params)
            flops.append(ops)
        except:
            parameters.append(np.nan)
            flops.append(np.nan)
            print(f"Error processing {config}")

    df['parameters'] = parameters
    df['flops'] = flops
    df.to_pickle('checkpoints/results.ckpt')

log_checkpoint_data()
add_param_counts()
