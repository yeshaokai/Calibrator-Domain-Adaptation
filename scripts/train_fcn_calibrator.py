import logging
import os
import os.path
from collections import deque
import itertools
from datetime import datetime
import torch.optim as optim
import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter

from PIL import Image
from torch.autograd import Variable

from cycada.data.adda_datasets import AddaDataLoader
from cycada.models import get_model
from cycada.models.backbones import GANLoss
from cycada.models.models import models
from cycada.models import VGG16_FCN8s, Discriminator
from cycada.util import config_logging
from cycada.util import to_tensor_raw
from cycada.tools.util import make_variable


def check_label(label, num_cls):
    "Check that no labels are out of range"
    label_classes = np.unique(label.numpy().flatten())
    label_classes = label_classes[label_classes < 255]
    if len(label_classes) == 0:
        print('All ignore labels')
        return False
    class_too_large = label_classes.max() > num_cls
    if class_too_large or label_classes.min() < 0:
        print('Labels out of bound')
        print(label_classes)
        return False
    return True


def forward_pass(net, discriminator, im, requires_grad=False, discrim_feat=False):
    if discrim_feat:
        score, feat = net(im)
        dis_score = discriminator(feat)
    else:
        score = net(im)
        dis_score = discriminator(score)
    if not requires_grad:
        score = Variable(score.data, requires_grad=False)

    return score, dis_score


def supervised_loss(score, label, weights=None):
    loss_fn_ = torch.nn.NLLLoss(weight=weights, size_average=True,
                                ignore_index=255)
    loss = loss_fn_(F.log_softmax(score, dim=1), label)
    return loss


def discriminator_loss(score, target_val, lsgan=False):
    if lsgan:
        loss = 0.5 * torch.mean((score - target_val) ** 2)
    else:
        _, _, h, w = score.size()
        target_val_vec = Variable(target_val * torch.ones(1, h, w), requires_grad=False).long().cuda()
        loss = supervised_loss(score, target_val_vec)
    return loss


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def seg_accuracy(score, label, num_cls):
    _, preds = torch.max(score.data, 1)
    hist = fast_hist(label.cpu().numpy().flatten(),
                     preds.cpu().numpy().flatten(), num_cls)
    intersections = np.diag(hist)
    unions = (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-8) * 100
    acc = np.diag(hist).sum() / hist.sum()
    return intersections, unions, acc


def forward_calibrator(data_s, data_t, calibrator, mean=0.6):
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


@click.command()
@click.argument('output')
@click.option('--dataset', required=True, multiple=True)
@click.option('--datadir', default="", type=click.Path(exists=True))
@click.option('--lr', '-l', default=0.0001)
@click.option('--momentum', '-m', default=0.9)
@click.option('--batch', default=1)
@click.option('--snapshot', '-s', default=5000)
@click.option('--downscale', type=int)
@click.option('--crop_size', default=None, type=int)
@click.option('--half_crop', default=None)
@click.option('--cls_weights', type=click.Path(exists=True))
@click.option('--weights_discrim', type=click.Path(exists=True))
@click.option('--weights_init', type=click.Path(exists=True))
@click.option('--model', default='fcn8s', type=click.Choice(models.keys()))
@click.option('--lsgan/--no_lsgan', default=False)
@click.option('--num_cls', type=int, default=19)
@click.option('--gpu', default='0')
@click.option('--max_iter', default=10000)
@click.option('--lambda_d', default=1.0)
@click.option('--lambda_g', default=1.0)
@click.option('--train_discrim_only', default=False)
@click.option('--discrim_feat/--discrim_score', default=False)
@click.option('--weights_shared/--weights_unshared', default=False)
def main(output, dataset, datadir, lr, momentum, snapshot, downscale, cls_weights, gpu,
         weights_init, num_cls, lsgan, max_iter, lambda_d, lambda_g,
         train_discrim_only, weights_discrim, crop_size, weights_shared,
         discrim_feat, half_crop, batch, model):
    # So data is sampled in consistent way
    #np.random.seed(1337)
    #torch.manual_seed(1337)
    logdir = 'runs/{:s}/{:s}_to_{:s}/lr{:.1g}_ld{:.2g}_lg{:.2g}'.format(model, dataset[0],
                                                                        dataset[1], lr, lambda_d, lambda_g)
    if weights_shared:
        logdir += '_weightshared'
    else:
        logdir += '_weightsunshared'
    if discrim_feat:
        logdir += '_discrimfeat'
    else:
        logdir += '_discrimscore'
    logdir += '/' + datetime.now().strftime('%Y_%b_%d-%H:%M')
    writer = SummaryWriter(log_dir=logdir)

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    config_logging()
    print('Train Discrim Only', train_discrim_only)
    print (weights_init)
    
    net = get_model('CalibratorNet', model=model, num_cls=num_cls, cali_model='resnet_9blocks',src_weights_init = weights_init,task = 'segmentation')
    

    net.src_net.eval()

    
    #odim = 1 if lsgan else 2
    odim = 4
    idim = num_cls if not discrim_feat else 4096

    '''
    discriminator = Discriminator(input_dim=idim, output_dim=odim,
                                  pretrained=not (weights_discrim == None),
                                  weights_init=weights_discrim).cuda()
    
    net.discriminator = discriminator
    '''
    print (net)
    transform = net.src_net.module.transform if hasattr(net.src_net,'module') else net.src_net.transform
    
    loader = AddaDataLoader(transform, dataset, datadir, downscale,
                            crop_size=crop_size, half_crop=half_crop,
                            batch_size=batch, shuffle=True, num_workers=2)
    print('dataset', dataset)

    # Class weighted loss?
    if cls_weights is not None:
        weights = np.loadtxt(cls_weights)
    else:
        weights = None

    # setup optimizers

    weight_decay = 0.005
    betas = (0.9,0.999)
    lr = 2e-4

    '''
    opt_dis = optim.SGD(net.discriminator.parameters(), lr=lr,
                        momentum = momentum, weight_decay = 0.0005)

    opt_p = optim.SGD(net.pixel_discriminator.parameters(), lr=lr,
                        momentum = momentum, weight_decay = 0.0005)
    
    opt_cali = optim.SGD(net.calibrator_T.parameters(), lr=lr,
                         momentum = momentum, weight_decay = 0.0005)
    '''
    


    opt_dis = optim.Adam(net.discriminator.parameters(), lr=lr,
                         weight_decay=weight_decay, betas=betas)

    opt_p = optim.Adam(net.pixel_discriminator.parameters(), lr=lr,
                         weight_decay=weight_decay, betas=betas)

    
    opt_cali = optim.Adam(net.calibrator_T.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)

    iteration = 0
    num_update_g = 0
    last_update_g = -1
    losses_super_s = deque(maxlen=100)
    losses_super_t = deque(maxlen=100)
    losses_dis = deque(maxlen=100)
    losses_rep = deque(maxlen=100)
    accuracies_dom = deque(maxlen=100)
    intersections = np.zeros([100, num_cls])
    unions = np.zeros([100, num_cls])
    accuracy = deque(maxlen=100)
    print('max iter:', max_iter)

    net.src_net.train()
    net.discriminator.train()
    net.pixel_discriminator.train()
    net.calibrator_T.train()
    freq_D = 8
    freq_G = 8
    while iteration < max_iter:

        for im_s, im_t, label_s, label_t in loader:
            if iteration > max_iter:
                break
            if im_s.size(0) != im_t.size(0):
                continue
            
            info_str = 'Iteration {}: '.format(iteration)

            if not check_label(label_s, num_cls):
                continue

            ###########################
            # 1. Setup Data Variables #
            ###########################
            im_s = make_variable(im_s, requires_grad=False)
            label_s = make_variable(label_s, requires_grad=False)
            im_t = make_variable(im_t, requires_grad=False)
            label_t = make_variable(label_t, requires_grad=False)

            #############################
            # 2. Optimize Discriminator #
            #############################

            # zero gradients for optimizer
            if (iteration+1) % freq_D ==0:
                opt_dis.step()
                opt_p.step()                
                opt_dis.zero_grad()
                opt_p.zero_grad()

            pert_T_s,pert_T_t = forward_calibrator(im_s,im_t,net.calibrator_T)

            fake_T_s = torch.clamp(im_s+pert_T_s,-3,3)
            fake_T_t = torch.clamp(im_t+pert_T_t,-3,3)

            score_s,score_t = forward_clean_data(im_s,im_t,net.src_net)
            score_T_s,score_T_t = forward_pert_data(fake_T_s,fake_T_t,net.src_net)

            pred_p_t = net.pixel_discriminator(im_t)
            pred_p_s = net.pixel_discriminator(im_s)
            pred_p_T_s = net.pixel_discriminator(fake_T_s)
            pred_p_T_t = net.pixel_discriminator(fake_T_t)

            # prediction for feature discriminator            

            gan_criterion = GANLoss().cuda()
            idt_criterion = torch.nn.L1Loss()
            cycle_criterion = torch.nn.L1Loss()

            # 0,1,2,3 for 4 different domains


            pred_s = net.discriminator(score_s)
            pred_t = net.discriminator(score_t)
            pred_T_t = net.discriminator(score_T_t)
            pred_T_s = net.discriminator(score_T_s)
                       
            

                    
            #dis_pred_concat = torch.cat((dis_score_s, dis_score_t))

            # prepare real and fake labels
            batch_t, _, h, w = score_t.size()
            batch_s, _, _, _ = score_s.size()
            label_0 = make_variable(
                    0*torch.ones(batch_s, h, w).long(), #s
                 requires_grad=False)
            label_1 = make_variable(
                    1*torch.ones(batch_t, h, w).long(), #t
                 requires_grad=False)

            label_2 = make_variable(
                    2*torch.ones(batch_t, h, w).long(), #T_t
                 requires_grad=False)
            label_3 = make_variable(
                    3*torch.ones(batch_s, h, w).long(), #T_s
                 requires_grad=False)                                    

            P_loss_s = gan_criterion(pred_p_s,0)                
            P_loss_t = gan_criterion(pred_p_t,1)
            P_loss_T_t = gan_criterion(pred_p_T_t,2)
            P_loss_T_s = gan_criterion(pred_p_T_s,3)


            dis_pred_concat = torch.cat([pred_s,pred_t,pred_T_t,pred_T_s])
            dis_label_concat = torch.cat([label_0,label_1,label_2,label_3])
            
            # compute loss for discriminator
            loss_dis = supervised_loss(dis_pred_concat, dis_label_concat)

            D_loss = loss_dis + P_loss_T_t + P_loss_s + P_loss_t + P_loss_T_s
            
            D_loss.backward(retain_graph = True)
            losses_dis.append(D_loss.item())


            info_str += ' D:{:.3f}'.format(np.mean(losses_dis))
            # optimize discriminator

            

            ###########################
            # Optimize Target Network #
            ###########################



            if True:
                
                last_update_g = iteration
                num_update_g += 1
                if num_update_g % 1 == 0:
                    pass
                    #print('Updating G with adversarial loss ({:d} times)'.format(num_update_g))

                # zero out optimizer gradients
                if (iteration+1)% freq_G == 0:
                    opt_cali.step()                    
                    opt_dis.zero_grad()
                    opt_p.zero_grad()
                    opt_cali.zero_grad()
                # create fake label
                batch, _, h, w = score_t.size()

                label_T_t = make_variable(
                    0*torch.ones(batch, h, w).long(),
                 requires_grad=False)
                label_T_s = make_variable(
                    0*torch.ones(batch, h, w).long(),
                 requires_grad=False)
                

                P_loss_T_t = gan_criterion(pred_p_T_t,0)
                P_loss_T_s = gan_criterion(pred_p_T_s,0)

                
                G_loss_T_t = supervised_loss(pred_T_t,label_T_t)
                
                G_loss_T_s = supervised_loss(pred_T_s,label_T_s)
                

                G_loss = G_loss_T_t + 0.2*G_loss_T_s + P_loss_T_t + 0.2*P_loss_T_s  
                G_loss.backward()



                losses_rep.append(G_loss.item())
                #writer.add_scalar('loss/generator', np.mean(losses_rep), iteration)

                # optimize target net

                # log net update info
                info_str += ' G:{:.3f}'.format(np.mean(losses_rep))


            # compute supervised losses for target -- monitoring only!!!


            ###########################
            # Log and compute metrics #
            ###########################
            if iteration % 10 == 0 and iteration > 0:
                # compute metrics
                intersection, union, acc = seg_accuracy(score_T_t, label_t.data, num_cls)
                intersections = np.vstack([intersections[1:, :], intersection[np.newaxis, :]])
                unions = np.vstack([unions[1:, :], union[np.newaxis, :]])
                accuracy.append(acc.item() * 100)
                acc = np.mean(accuracy)
                mIoU = np.mean(np.maximum(intersections, 1) / np.maximum(unions, 1)) * 100

                info_str += ' acc:{:0.2f}  mIoU:{:0.2f}'.format(acc, mIoU)
                writer.add_scalar('metrics/acc', np.mean(accuracy), iteration)
                writer.add_scalar('metrics/mIoU', np.mean(mIoU), iteration)
                print (info_str)
                logging.info(info_str)

            iteration += 1

            ################
            # Save outputs #
            ################

            # every 500 iters save current model
            if iteration % 500 == 0:
                os.makedirs(output, exist_ok=True)
                if not train_discrim_only:
                    torch.save(net.state_dict(),
                               '{}/net-itercurr.pth'.format(output))
                torch.save(net.discriminator.state_dict(),
                           '{}/discriminator-itercurr.pth'.format(output))

            # save labeled snapshots
            if iteration % snapshot == 0:
                os.makedirs(output, exist_ok=True)
                if not train_discrim_only:
                    torch.save(net.state_dict(),
                               '{}/net-iter{}.pth'.format(output, iteration))
                torch.save(net.discriminator.state_dict(),
                           '{}/discriminator-iter{}.pth'.format(output, iteration))

            if iteration - last_update_g >= len(loader):
                print('No suitable discriminator found -- returning.')
                torch.save(net.state_dict(),
                           '{}/net-iter{}.pth'.format(output, iteration))
                iteration = max_iter  # make sure outside loop breaks
                break

    writer.close()


if __name__ == '__main__':
    main()
