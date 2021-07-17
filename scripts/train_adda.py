import os
from os.path import join

import argparse
from cycada.tools.train_task_net import train as train_source
from cycada.tools.test_task_net import load_and_test_net
from cycada.tools.train_adda_net import train_adda
import torch
import numpy as np

# set random seed to 4325 
# to reproduce the exact numbers
np.random.seed(4325)

###################################
# Set to your preferred data path #
###################################
datadir = './'
###################################

# Choose GPU ID
parser = argparse.ArgumentParser(description='')

parser.add_argument('--src', type=str, default = 'usps')
parser.add_argument('--tgt', type=str, default = 'mnist')
parser.add_argument('--third', type=str, default = '')
parser.add_argument('--lr', type=float, default = 1e-5)
parser.add_argument('--name', type = str, default = '', help = 'for multiple card training')
parser.add_argument('--arch', type = str, default = '', help = 'use of arch')
parser.add_argument('--epoch', type = int, default = 10)

args = parser.parse_args()

# Problem Params
src = args.src
tgt = args.tgt

third = args.third
iteration = 1 #'no_cycle' 


base_src = src.split('2')[0]

model = args.arch
num_cls = 10

# Output directory
outdir = 'results/{}_to_{}/iter_{}'.format(src, tgt, iteration)
# outdir = 'results/{}_to_{}'.format(src, tgt)

# Optimization Params
betas = (0.9, 0.999)  # Adam default
weight_decay = 0  # Adam default
batch = 128

src_lr = 1e-4
src_num_epoch = 20
src_datadir = join(datadir, src)
src_net_file = join(outdir, '{}_net_{}.pth'.format(model, src))
adda_num_epoch = args.epoch
adda_lr = args.lr
adda_net_file = join(outdir, 'adda_{:s}_net_{:s}_{:s}_{:s}.pth'
                     .format(model, src, tgt, args.name))

#######################
# 1. Train Source Net #
#######################


if os.path.exists(src_net_file):
    print('Skipping source net training, exists:', src_net_file)
else:

    train_source(src, src_datadir, model, num_cls, 
            outdir=outdir, num_epoch=src_num_epoch, batch=batch, 
            lr=src_lr, betas=betas, weight_decay=weight_decay)



#####################
# 2. Train Adda Net #
#####################

if os.path.exists(adda_net_file):
    print('Skipping adda training, exists:', adda_net_file)
else:

    train_adda(src, tgt,third, model, num_cls, num_epoch=adda_num_epoch, 
            batch=batch, datadir=datadir,
            outdir=outdir, src_weights=src_net_file, 
               lr=adda_lr, betas=betas, weight_decay=weight_decay, name=args.name)

##############################
# 3. Evalute source and adda #
##############################
tgt_datadir = join(datadir, tgt)
print()
if src == base_src:
    print('----------------')
    print('Test set:', src)
    print('----------------')
    print('Evaluating {} source model: {}'.format(src, src_net_file))
    load_and_test_net(src, src_datadir, src_net_file, model, num_cls,
                      dset='test', base_model=None)

print('----------------')
print('Test set:', tgt)
print('----------------')
print('Evaluating {} source model: {}'.format(src, src_net_file))
cm = load_and_test_net(tgt, tgt_datadir, src_net_file, model, num_cls,
                       dset='test', base_model=None
                       )

print(cm)

print('Evaluating {}->{} adda model: {}'.format(src, tgt, adda_net_file))
cm = load_and_test_net(tgt, tgt_datadir, adda_net_file, 'AddaNet', num_cls,
                       dset='test', base_model=model)
print(cm)
