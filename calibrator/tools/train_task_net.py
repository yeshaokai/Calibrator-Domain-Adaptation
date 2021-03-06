from __future__ import print_function

import os
from os.path import join
import numpy as np
import argparse

# Import from torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Import from Cycada Package 
from ..models.models import get_model
from ..data.data_loader import load_data
from ..data.cyclegan import Svhn2MNIST
from .test_task_net import test
from .util import make_variable


def train_epoch(loader, net, opt_net, epoch):
    log_interval = 100  # specifies how often to display
    net.train()
    for batch_idx, (data, target) in enumerate(loader):

        # make data variables
        data = make_variable(data, requires_grad=False)
        target = make_variable(target, requires_grad=False)

        # zero out gradients
        opt_net.zero_grad()

        # forward pass
        score = net(data)
        loss = net.criterion(score, target)

        # backward pass
        loss.backward()

        # optimize classifier and representation
        opt_net.step()

        # Logging
        if batch_idx % log_interval == 0:
            print('[Train] Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset),
                       100. * batch_idx / len(loader), loss.item()), end="")
            pred = score.data.max(1)[1]
            correct = pred.eq(target.data).cpu().sum()
            acc = correct.item() / len(pred) * 100.0
            print('  Acc: {:.2f}'.format(acc))


def train(data, datadir, cfg):
    """Train a classification net and evaluate on test set."""

    
    # Setup GPU Usage
    if torch.cuda.is_available():
        kwargs = {'num_workers': cfg['TRAIN']['NUM_WORKERS'], 'pin_memory': True}
    else:
        kwargs = {}

    ############
    # Load Net #
    ############

    model = cfg['SRC_NET']['ARCH']

    net = get_model(model, cfg)
    print('-------Training net--------')
    print(net)

    ############################
    # Load train and test data # 
    ############################
    #train_data = load_data(data, 'train', batch=batch,
    #                       rootdir=datadir, num_channels=net.num_channels,
    #                       image_size=net.image_size, download=True, kwargs=kwargs)

    #svhn2mnist_dataset = Svhn2MNIST(datadir,train=True)


    #train_data = torch.utils.data.DataLoader(svhn2mnist_dataset,batch_size = batch, shuffle = True)
    train_data = load_data(data, 'train', batch=cfg['SRC_NET']['BATCH_SIZE'],
                          rootdir=datadir, num_channels=net.num_channels,
                           image_size=net.image_size, download=True,kwargs=kwargs)
    
    test_data = load_data(data, 'test', batch=cfg['SRC_NET']['BATCH_SIZE'],
                          rootdir=datadir, num_channels=net.num_channels,
                          image_size=net.image_size, download=False,kwargs=kwargs)

    ###################
    # Setup Optimizer #
    ###################
    opt_net = optim.Adam(net.parameters(), lr=cfg['SRC_NET']['SRC_LR'], betas=cfg['OPTIMIZER']['ADAM']['BETAS'],
                         weight_decay=cfg['OPTIMIZER']['ADAM']['WEIGHT_DECAY'])

    

    #########
    # Train #
    #########
    print('Training {} model for {}'.format(model, data))
    for epoch in range(cfg['SRC_NET']['SRC_EPOCH']):
        train_epoch(train_data, net, opt_net, epoch)

    ########
    # Test #
    ########
    if test_data is not None:
        print('Evaluating {}-{} model on {} test set'.format(model, data, data))
        test(test_data, net, cfg)

    ############
    # Save net #
    ############
    os.makedirs(cfg['PATH']['OUTDIR'], exist_ok=True)

    outfile = cfg['PATH']['SRC_NET_PATH']
    
    print('Saving to', outfile)
    net.save(outfile)

    return net
