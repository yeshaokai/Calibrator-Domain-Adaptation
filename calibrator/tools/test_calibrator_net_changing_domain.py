from __future__ import print_function

from queue import Queue
from typing import List, Any

import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from cycada.data.data_loader import load_data
from cycada.models import get_model
from .util import make_variable


class SuffixAverageCalculator(object):
    def __init__(self, average_length):
        self.__queue = Queue()
        self.__sum = 0.0
        self.__length_max = average_length

    def append(self, value):
        self.__queue.put(value)
        self.__sum += value
        if self.__queue.qsize() > self.__length_max:
            self.__sum -= self.__queue.get()

    def average(self):
        return self.__sum / self.__queue.qsize()

    def __len__(self):
        return self.__queue.qsize()


def test(net: nn.Module, data_block_size: int = 100, use_tensorboard: bool = False,
         fixed_change_rate: bool = True, change_rate: int = 20,
         random_change_possibility: float = 0.001, buffer_length=10, save_path="log",
         loaders_list: List[DataLoader] = []):
    """
    tests model in a changing domain ways
    :param loaders_list: data_loader for each domain
    :param save_path: the saving path for tensorboardX
    :param buffer_length: acc and loss is calculated on recent buffer_length*batch_size data
    :param change_rate: if fixed_change_rate is True, after every change_rate change a domain
    :param fixed_change_rate: whether use a fixed change rate.(1000 src + 1000 tgt + 1000src ...)
    :param random_change_possibility: random change possibility after each batch
    :param use_tensorboard: whether use a tensorboard to show the result or just print on screen
    :param data_block_size: every data_block_size data will sample from same origin to ensure a nice visualize
    :param net: networks
    :return:
    """
    print_test_loss_list = []
    print_acc_list = []
    print_data_source_list = []
    test_loss = SuffixAverageCalculator(buffer_length)
    acc = SuffixAverageCalculator(buffer_length)

    data_iterators = [iter(d) for d in loaders_list]
    current_dataset = np.random.randint(0, len(loaders_list))
    cnt = 0

    # define other metrics here
    max_scores = SuffixAverageCalculator(buffer_length)
    print_max_score_average = []

    net.eval()
    while True:
        try:
            (data, target) = next(data_iterators[current_dataset])
            data = make_variable(data, requires_grad=False)
            target = make_variable(target, requires_grad=False)
            if net.name == 'CalibratorNet':
                pert = net.calibrator_T(data)
                score = net.src_net(torch.clamp(data + pert, -1, 1))
            else:
                score = net(data)
            pred = score.data.max(1)[1]  # get the index of the max log-probability

            test_loss.append(nn.CrossEntropyLoss()(score, target).item())
            acc.append(pred.eq(target.data).cpu().sum())

            print_data_source_list.append(current_dataset)
            print_test_loss_list.append(test_loss.average())
            print_acc_list.append(acc.average())

            # add other metrics here
            max_scores.append(torch.max(score).item())
            print_max_score_average.append(max_scores.average())

            cnt += 1
            if fixed_change_rate:
                if cnt % change_rate == 0:
                    current_dataset = np.random.randint(low=0, high=len(loaders_list))
                    print(str(cnt) + ": change data set to "+str(current_dataset))
            else:
                if np.random.binomial(1, random_change_possibility) == 1:
                    current_dataset = np.random.randint(low=0, high=len(loaders_list))
                    print(str(cnt) + "change data set to " + str(current_dataset))

        except StopIteration:
            break

    log_test_info(test_loss_list=print_test_loss_list, acc_list=print_acc_list, data_source_list=print_data_source_list,
                  use_tensorboard=use_tensorboard, save_path=save_path, max_scores=print_max_score_average)

    return 0


def log_test_info(test_loss_list: List = [], acc_list: List = [], data_source_list: List = [],
                  use_tensorboard: bool = False, save_path="log", **other_list: List):
    """
    print test infos
    :param save_path: the saving path for tensorboardX
    :param use_tensorboard: whether use a tensorboard to show the result or just print on screen
    :param data_block_size: every data_block_size data will sample from same origin to ensure a nice visualize
    :param test_loss_list: list of losses
    :param acc_list: list of acc
    :param data_source_list: list of labels of domain for each data
    :param other_list: other things you want to print
    :return:
    """
    metrics = other_list
    metrics["Loss"] = test_loss_list
    metrics["Accuracy"] = acc_list
    metrics["Data Source"] = data_source_list
    if use_tensorboard:
        print("writing data...")
        writer = SummaryWriter(save_path)
        for key, data_list in metrics.items():
            for i, v in enumerate(data_list):
                writer.add_scalar("Test/"+key, v, i)
        print("Result saved to " + save_path)
    else:
        info_str = '[Evaluate]'
        for (key, data_list) in metrics.items():
            info_str += key + ": {:.4f}, ".format(np.mean(data_list))
        print(info_str)


def load_and_test_net(data_list, datadir, weights, model, num_cls,
                      dset='test', base_model=None, base_path=None, **test_args):
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
                        model=base_model)

    else:
        net = get_model(model, num_cls=num_cls, weights_init=weights)

    # Load data
    loaders = []
    for (i, data) in enumerate(data_list):
        test_data = load_data(data, dset, batch=100,
                              rootdir=datadir[i], num_channels=net.num_channels,
                              image_size=net.image_size, download=True, kwargs=kwargs)
        if test_data is None:
            print('skipping test')
            return
        loaders.append(test_data)

    return test(net, loaders_list=loaders, save_path=base_path + "_with_" + "_".join(data_list) + "log", **test_args)
