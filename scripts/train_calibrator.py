import os
from os.path import join

import argparse

from calibrator.tools.train_task_net import train as train_source
from calibrator.tools.test_task_net import load_and_test_net
from calibrator.tools.train_calibrator_net import train_calibrator
import torch
import numpy as np
from torch.utils.data import DataLoader
import yaml
import ast 

parser = argparse.ArgumentParser(description='')

parser.add_argument('--yaml_path', type = str, default = '')


args = parser.parse_args()
gpu_num = torch.cuda.device_count()
#np.random.seed(4325)



###################################
# Set to your preferred data path #
###################################

###################################

# Choose GPU ID


try:
    with open (args.yaml_path,'r') as file:
        cfg = yaml.safe_load(file)
except Exception as e:
    print ('Error reading the config file')


    
adatadir = cfg['TRAIN']['DATA_DIR']

# Problem Params

src = cfg['DATASET']['SRC']
tgt = cfg['DATASET']['TGT']

base_src = src.split('2')[0]

src_net_arch = cfg['SRC_NET']['ARCH']
num_cls = cfg['SRC_NET']['NUM_CLS']

# Output directory


outdir = '{}/{}_to_{}'.format(cfg['TRAIN']['RESULT_DIR'],
                                           src,tgt)                          

datadir = cfg['TRAIN']['DATA_DIR']


src_net_file = join(outdir, '{}_net_{}.pth'.format(src_net_arch, src))


calibrator_net_file = join(outdir, 'calibrator_{:s}_net_{:s}_{:s}.pth'
                           .format(src_net_arch, src, tgt))


cfg['PATH'] = {}

cfg['PATH']['OUTDIR'] = outdir

cfg['PATH']['SRC_NET_PATH'] = src_net_file

cfg['PATH']['CALIBRATOR_NET_PATH'] = calibrator_net_file

src_datadir = join(datadir, src)
tgt_datadir = join(datadir, tgt)


# Optimization Params

betas = cfg['OPTIMIZER']['ADAM']['BETAS']#(0.9, 0.999)  # Adam default

betas = ast.literal_eval(betas)

cfg['OPTIMIZER']['ADAM']['BETAS'] = betas


#######################
# 1. Train Source Net #
#######################


if os.path.exists(src_net_file):
    print('Skipping source net training, exists:', src_net_file)
else:
    train_source(src, src_datadir, cfg)


#####################
# 2. Train Calibrator Net #
#####################

if os.path.exists(calibrator_net_file):
    print('Skipping calibrator training, exists:', calibrator_net_file)
else:
    train_calibrator(cfg,src_net_file)
                     

##############################
# 3. Evaluate source and calibrator #
##############################


if src == base_src:
    print('----------------')
    print('Test set:', src)
    print('----------------')
    print('Evaluating {} source model: {}'.format(src, src_net_file))
    load_and_test_net(src, src_datadir, src_net_file, src_net_arch,cfg,
                      dset='test')

print('----------------')
print('Test set:', tgt)
print('----------------')
print('Evaluating {} source model: {}'.format(src, src_net_file))
cm = load_and_test_net(tgt, tgt_datadir, src_net_file, src_net_arch,cfg,
                       dset='test')

print('Evaluating {}->{} calibrator model: {}'.format(src, tgt, calibrator_net_file))
cm = load_and_test_net(tgt, tgt_datadir, calibrator_net_file, 'CalibratorNet', cfg,
                       dset='test')


print('Evaluating {}->{} calibrator model: {}'.format(src, src, calibrator_net_file))
cm = load_and_test_net(src, src_datadir, calibrator_net_file, 'CalibratorNet', cfg,
                       dset='test')
print(cm)

