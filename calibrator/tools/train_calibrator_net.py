from __future__ import print_function

import os
from os.path import join
import numpy as np

# Import from torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import itertools
# Import from within Package 
from ..models.models import get_model
from ..models.backbones import GANLoss
from cycada.data.data_loader import load_data # use cycada data loader
from ..tools.test_task_net import test
from ..tools.util import make_variable
import time
import torch.optim as optim

torch.backends.cudnn.benchmark = True

from tensorboardX import SummaryWriter


class Eventer:
    def __init__(self,cfg,variable):
        self.cfg = cfg        
        self.variable  = variable    
        self.setup()
    def setup(self):
        gpu_id = os.environ['CUDA_VISIBLE_DEVICES']
        # varying a variable, kept the rest change
        value = self.cfg['CALIBRATOR'][self.variable]
        log_filename = 'runs/gpu{}_{}_{}'.format(gpu_id,self.variable,value)
        self.writer = SummaryWriter(log_filename)
    def add_scalar(self,tag,value,iteration):
        self.writer.add_scalar(tag,value,iteration)
        
    def write_gradients(self,net,epoch):
        # perform gradient analysis in softmax layer first
        for name,w  in net.named_parameters():            
            gradient = w.grad
            self.writer.add_histogram('{}_{}'.format(net.name,name),gradient,epoch)
        

def forward_calibrator(data_s, data_t, calibrator):
    # task option impacts the L_finty_norm setting    
    pert_s = calibrator(data_s)
    pert_t = calibrator(data_t)
    return pert_s, pert_t


def forward_clean_data(data_s, data_t, src_net):
    score_s = src_net(data_s)
    score_t = src_net(data_t)

    return score_s, score_t


def forward_pert_data(fake_s,fake_t,src_net):
    score_s = src_net(fake_s)
    score_t = src_net(fake_t)

    return score_s, score_t

def reset_parameters(pix_discriminator):
    for m in pix_discriminator.modules():
        print ('reset {}'.format(m))
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)



def train(loader_src, loader_tgt, net, opt_dis, opt_p, opt_cali, lr_schedulers,epoch,cfg):
    
    net.pixel_discriminator.train()
    net.discriminator.train()
    net.calibrator_T.train()
    net.src_net.train()
    
    log_interval = 100  # specifies how often to display

    start = time.time()

    # n = min(len(loader_src.dataset), len(loader_tgt.dataset))
    n = len(loader_tgt.dataset)
    joint_loader = zip(loader_src, loader_tgt)

        
    good_epoch = -1
    last_update = -1

    # reset pixel discriminator every epoch


    for batch_idx, ((data_s, _), (data_t, _)) in enumerate(joint_loader):
        with torch.autograd.set_detect_anomaly(True):
        
            if data_s.shape[0] != data_t.shape[0]:
                # skip in the tail of the epoch
                continue


            # log basic adda train info
            info_str = "[Train CalibratorNet] Epoch: {} [{}/{} ({:.2f}%)]".format(
                epoch, batch_idx * len(data_t), n, 100 * batch_idx / n)

            ########################
            # Setup data variables #
            ########################
            data_s = make_variable(data_s, requires_grad=False)
            data_t = make_variable(data_t, requires_grad=False)

            ##########################
            # Optimize discriminator #
            ##########################

            # zero gradients for optimizer

            opt_cali.zero_grad()            
            opt_dis.zero_grad()
            opt_p.zero_grad()

            # first, forward inputs from both domains via calibrator
            # this gives calibration for both source imgs and target imgs
            pert_T_s, pert_T_t = forward_calibrator(data_s,data_t,net.calibrator_T)

            # after adding the calibration to imgs, clip calibrated imgs back to the range

            box_min = cfg['CALIBRATOR']['BOX_MIN']
            box_max = cfg['CALIBRATOR']['BOX_MAX']        


            
            fake_T_s = torch.clamp(data_s+pert_T_s,box_min,box_max)
            fake_T_t = torch.clamp(data_t+pert_T_t,box_min,box_max)

            # get the logits of original inputs
            score_s, score_t = forward_clean_data(data_s,data_t,net.src_net)
            # get the logits of calibrated inputs
            score_T_s,score_T_t = forward_pert_data(fake_T_s,fake_T_t,net.src_net)

            # prediction of pixel discriminator for 4 groups

            pred_p_t = net.pixel_discriminator(data_t)
            pred_p_s = net.pixel_discriminator(data_s)
            pred_p_T_s = net.pixel_discriminator(fake_T_s)
            pred_p_T_t = net.pixel_discriminator(fake_T_t)

            gan_criterion = GANLoss().cuda()


            # prediction for feature discriminator for 4 groups
            pred_s = net.discriminator(score_s)
            pred_t = net.discriminator(score_t)
            pred_T_t = net.discriminator(score_T_t)
            pred_T_s = net.discriminator(score_T_s)


            # feature domain discriminator, assigning prediction to their corresponding group ids        
            D_loss_s = gan_criterion(pred_s, 0)
            D_loss_t = gan_criterion(pred_t, 1)
            D_loss_T_t = gan_criterion(pred_T_t, 2)
            D_loss_T_s = gan_criterion(pred_T_s, 3)

            # pixel domain discriminator, assigning prediction to their corresponding group ids                    
            P_loss_s = gan_criterion(pred_p_s,0)                
            P_loss_t = gan_criterion(pred_p_t,1)
            P_loss_T_t =  gan_criterion(pred_p_T_t,2)
            P_loss_T_s = gan_criterion(pred_p_T_s,3)


            # overall domain discrminator loss
            if cfg['CALIBRATOR']['USE_PIXEL']:        
                D_loss = P_loss_T_t + P_loss_s + P_loss_t  + P_loss_T_s + \
                      D_loss_s + D_loss_t + D_loss_T_t + D_loss_T_s 
            else:
                D_loss = D_loss_s + D_loss_t + D_loss_T_t + D_loss_T_s 

            D_loss.backward(retain_graph=False)  

            # update pixel and feature domain discriminators

            opt_dis.step()
            opt_p.step()

            # print overall loss for discriminators
            info_str += " D_loss {:.3f}".format(D_loss.item())


            ###########################
            # Optimize target network #
            ###########################

            if True:
                # Different from ADDA, we do not choose to only update
                # adversary when loss of discriminators is low

                last_update = batch_idx

                # zero out optimizer gradients            
                opt_cali.zero_grad()
                opt_p.zero_grad()
                opt_dis.zero_grad()

                # trying to fool feature discriminator
                # to treat both calibrated src imgs and tgt imgs as source imgs

                
                pert_T_s, pert_T_t = forward_calibrator(data_s,data_t,net.calibrator_T)


                fake_T_s = torch.clamp(data_s+pert_T_s,box_min,box_max)
                fake_T_t = torch.clamp(data_t+pert_T_t,box_min,box_max)

                score_T_s,score_T_t = forward_pert_data(fake_T_s,fake_T_t,net.src_net)

                pred_T_t = net.discriminator(score_T_t)
                pred_T_s = net.discriminator(score_T_s)
                
                pred_p_T_s = net.pixel_discriminator(fake_T_s)
                pred_p_T_t = net.pixel_discriminator(fake_T_t)             
                
                
                G_loss_T_t = gan_criterion(pred_T_t,0)
                G_loss_T_s = gan_criterion(pred_T_s,0)

                P_loss_T_t = gan_criterion(pred_p_T_t,0)
                P_loss_T_s = gan_criterion(pred_p_T_s,0)


                # there is a trade off between not losing source domain performance and gaining good target domain performance. This coef is good for digits

                if cfg['CALIBRATOR']['USE_PIXEL']:
                    G_loss =  1.0*P_loss_T_t + 0.1*P_loss_T_s + 1.0*G_loss_T_t + 0.1*G_loss_T_s
                else:
                    G_loss =  1.0*G_loss_T_t + 1.0*G_loss_T_s
                G_loss.backward()

                # optimize calibrator network
                opt_cali.step()

                # log net update info

                info_str += " G_loss_T_s: {:.3f}".format(G_loss_T_s.item())
                info_str += " G_loss_T_t: {:.3f}".format(G_loss_T_t.item())

                info_str += " P_loss_T_s: {:.3f}".format(P_loss_T_s.item())
                info_str += " P_loss_T_t: {:.3f}".format(P_loss_T_t.item())

            ###########
            # Logging #
            ###########
            if batch_idx % log_interval == 0:
                print(info_str)

    print('elpased {} seconds one epoch'.format(time.time() - start))
    return last_update


def train_calibrator(cfg, src_net_file):
    """Main function for training ADDA."""

    ###########################
    # Setup cuda and networks #
    ###########################

    # setup cuda
    if torch.cuda.is_available():
        kwargs = {'num_workers': cfg['TRAIN']['NUM_WORKERS'], 'pin_memory': True}
    else:
        kwargs = {}

    # setup network 
    # calibrator net contains  source domain model, calibrator and domain discriminators
    net = get_model('CalibratorNet',cfg, src_weights_init = src_net_file)


    
    src = cfg['DATASET']['SRC']
    tgt = cfg['DATASET']['TGT']
    datadir = cfg['TRAIN']['DATA_DIR']
    src_net = cfg['SRC_NET']['ARCH']
    # print network and arguments
    print(net)
    print('Training calibrator with pretrained {} model for {}->{}'.format(src_net, src, tgt))

    #######################################
    # Setup data for training and testing #
    #######################################

    # define dataset properties
    num_channels = net.src_net.module.num_channels if hasattr(net.src_net,'module') else net.src_net.num_channels
    image_size = net.src_net.module.image_size if hasattr(net.src_net,'module') else net.src_net.image_size


    batch = cfg['CALIBRATOR']['BATCH_SIZE']
        
    train_src_data = load_data(src, 'train', batch=batch,
                               rootdir=join(datadir,src), num_channels=num_channels,
                               image_size=image_size, download=True, kwargs=kwargs)

    test_src_data = load_data(src, 'test', batch=batch,
                               rootdir=join(datadir,src), num_channels=num_channels,
                               image_size=image_size, download=True, kwargs=kwargs)
    

    train_tgt_data = load_data(tgt, 'train', batch=batch,
                               rootdir=join(datadir,tgt), num_channels=num_channels,
                               image_size=image_size, download=True, kwargs=kwargs)


    test_tgt_data = load_data(tgt, 'test', batch=batch,
                               rootdir=join(datadir,tgt), num_channels=num_channels,
                               image_size=image_size, download=True, kwargs=kwargs)
    

    
    ######################
    # Optimization setup #
    ######################

    # net_param = net.tgt_net.parameters()
    # opt_net = optim.Adam(net_param, lr=lr, weight_decay=weight_decay, betas=betas)


    lr = cfg['CALIBRATOR']['CALI_LR']

    weight_decay = cfg['OPTIMIZER']['ADAM']['WEIGHT_DECAY']
    betas = cfg['OPTIMIZER']['ADAM']['BETAS']


    
    
    
    opt_dis = optim.Adam(net.discriminator.parameters(), lr=lr,
                         weight_decay=weight_decay, betas=betas)

    opt_p = optim.Adam(net.pixel_discriminator.parameters(), lr=lr,
                         weight_decay=weight_decay, betas=betas)

    opt_cali = optim.Adam(net.calibrator_T.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)


    
    lr_decay_step = 10
    lr_decay_rate = 0.5
    
    lr_scheduler_cali = optim.lr_scheduler.StepLR(opt_cali, step_size=lr_decay_step, gamma=lr_decay_rate)
    lr_scheduler_p_dis = optim.lr_scheduler.StepLR(opt_p, step_size=lr_decay_step, gamma=lr_decay_rate)
    lr_scheduler_dis = optim.lr_scheduler.StepLR(opt_dis, step_size=lr_decay_step, gamma=lr_decay_rate)
    
    lr_schedulers = [lr_scheduler_cali, lr_scheduler_p_dis, lr_scheduler_dis]

    
    ##############
    # Train Adda #
    ##############


    eventer = Eventer(cfg, 'PATCH_SIZE')
    
    for epoch in range(cfg['CALIBRATOR']['CALI_EPOCH']):
        
        err = train(train_src_data, train_tgt_data, net, opt_dis, opt_p, opt_cali, lr_schedulers,epoch,cfg)
        
        #test(test_src_data, net) for source performance test
        print('{} test'.format(tgt))
        
        net.pixel_discriminator.eval()
        net.discriminator.eval()
        net.calibrator_T.eval()
        net.src_net.eval()        
        test_acc = test(test_tgt_data, net,cfg)

        eventer.add_scalar('test_acc', test_acc, epoch)
        eventer.write_gradients(net.calibrator_T,epoch)
        #eventer.write_gradients(net.pixel_discriminator,epoch)
        eventer.write_gradients(net.discriminator,epoch)        

    ##############
    # Save Model #
    ##############
    os.makedirs(cfg['PATH']['OUTDIR'], exist_ok=True)

    outfile = cfg['PATH']['CALIBRATOR_NET_PATH']
    
    print('Saving calibrator net to', outfile)
    net.save(outfile)



