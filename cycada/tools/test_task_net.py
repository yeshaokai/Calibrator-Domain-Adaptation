from __future__ import print_function
from os.path import join
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import os
import argparse
from sklearn.metrics import confusion_matrix

from ..data.data_loader import load_data
from ..models.models import get_model
from .util import make_variable


def test(loader, net):
    net.eval()
    test_loss = 0
    correct = 0
    cm = 0
    max_scores = []
    for idx, (data, target) in enumerate(loader):
        data = make_variable(data, requires_grad=False)
        target = make_variable(target, requires_grad=False)
        if net.name == 'CalibratorNet':
            pert = net.calibrator_T(data)
            score = net.src_net(torch.clamp(data + pert, -1, 1))
        else:
            score = net(data)

        max_scores.append(torch.max(score).item())
        test_loss += nn.CrossEntropyLoss()(score, target).item()

        pred = score.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
    n = len(loader.dataset)
    test_loss /= len(loader)  # loss function already averages over batch size
    mean_max_score = np.mean(max_scores)
    var_max_score = np.var(max_scores)
    info_str = '[Evaluate]'
    info_str += ' Average loss: {:.4f},'.format(test_loss)
    info_str += ' Accuracy: {}/{} ({:.2f}%),'.format(correct, n, 100. * correct/n)
    info_str += ' mean_max_score: {:.4f},'.format(mean_max_score)
    info_str += ' var_max_score: {:.4f},'.format(var_max_score)
    print(info_str)
    return cm


def load_and_test_net(data, datadir, weights, model, num_cls,
                      dset='test', cali_width_factor =4,base_model=None):
    # Setup GPU Usage
    if torch.cuda.is_available():
        kwargs = {'num_workers': 1, 'pin_memory': True}
    else:
        kwargs = {}

    # Eval tgt from AddaNet or TaskNet model #
    if model == 'AddaNet':
        net = get_model(model, num_cls=num_cls, weights_init=weights,
                        model=base_model)
        net = net.tgt_net
    elif model == 'CalibratorNet':
        net = get_model(model, num_cls=num_cls, weights_init=weights,
                        model=base_model,cali_width_factor = cali_width_factor)
    else:
        net = get_model(model, num_cls=num_cls, weights_init=weights)

    # Load data
    
    test_data = load_data(data, dset, batch=100,
                          rootdir=datadir, num_channels=net.num_channels,
                          image_size=net.image_size, download=True, kwargs=kwargs)
    if test_data is None:
        print('skipping test')
    else:
        return test(test_data, net)
