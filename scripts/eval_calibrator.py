import os
from tqdm import *

import click
import numpy as np
import torch
import torchvision
from torch.autograd import Variable

from cycada.data.data_loader import dataset_obj
from cycada.data.data_loader import get_fcn_dataset
from cycada.models.models import get_model
from cycada.models.models import models
from cycada.util import to_tensor_raw
from torchvision.utils import save_image


def convert(tensor,color_map):
    tensor = tensor[0]

    tensor_transpose = tensor

    shape = tensor_transpose.shape
    h = shape[0]
    w = shape[1]    
    out = np.zeros((h,w,3))
    
    for h in range(shape[0]):
        for w in range(shape[1]):
            index = tensor_transpose[h][w]
            if index == 255:
                continue
            out[h][w] = color_map[index]
    out = np.transpose(out,(2,0,1))
    out = torch.from_numpy(out)
    return out 
    

def save_seg_results(source_out,id):    

    labels = ['road','sidewalk','parking','rail track','building','wall','fence','guard rail','bridge','tunnel','pole','polegroup','traffic light','traffic sign','vegetation','terrain','sky','person','rider','car','truck','bus','caravan','trailer','train','motorcycle','bicycle']
    
    color_map = np.array([     
                               [0.50196078, 0.25098039, 0.50196078],
                               [0.95686275, 0.1372549 , 0.90980392],
                               [0.98039216, 0.66666667, 0.62745098],
                               [0.90196078, 0.58823529, 0.54901961],
                               [0.2745098 , 0.2745098 , 0.2745098 ],
                               [0.4       , 0.4       , 0.61176471],
                               [0.74509804, 0.6       , 0.6       ],
                               [0.70588235, 0.64705882, 0.70588235],
                               [0.58823529, 0.39215686, 0.39215686],
                               [0.58823529, 0.47058824, 0.35294118],
                               [0.6       , 0.6       , 0.6       ],
                               [0.6       , 0.6       , 0.6       ],
                               [0.98039216, 0.66666667, 0.11764706],
                               [0.8627451 , 0.8627451 , 0.        ],
                               [0.41960784, 0.55686275, 0.1372549 ],
                               [0.59607843, 0.98431373, 0.59607843],
                               [0.2745098 , 0.50980392, 0.70588235],
                               [0.8627451 , 0.07843137, 0.23529412],
                               [1.        , 0.        , 0.        ],
                               [0.        , 0.        , 0.55686275],
                               [0.        , 0.        , 0.2745098 ],
                               [0.        , 0.23529412, 0.39215686],
                               [0.        , 0.        , 0.35294118],
                               [0.        , 0.        , 0.43137255],
                               [0.        , 0.31372549, 0.39215686],
                               [0.        , 0.        , 0.90196078],
                               [0.46666667, 0.04313725, 0.1254902 ]
                               
                               ])
    

    ## save three images. source image, seg from cyclegan, our augmented results
    root = '/home/lthpc/data_calibrator/segimages'
    #source_img_path = os.path.join(root,'source_img_{}.png'.format(id))
    source_out_path = os.path.join(root,'our_gta_{}.png'.format(id))
    #label_out_path = os.path.join(root,'label_out_{}.png'.format(id))
    #save_image(source_img,source_img_path)
    save_image(convert(source_out,color_map),source_out_path)
    #save_image(convert(label,color_map),label_out_path)    
    
'''
def save_seg_results(ori_img,cycle_out,our_out,id):

    ori_img = ori_img.cpu().detach()
    cycle_out = cycle_out.cpu().detach()
    our_out = our_out.cpu().detach()

    #labels = ['road','sidewalk','parking','rail track','building','wall','fence','guard rail','bridge','tunnel','pole','polegroup','traffic light','traffic sign','vegetation','terrain','sky','person','rider','car','truck','bus','caravan','trailer','train','motorcycle','bicycle']
    
    color_map = np.array([
                               [0.50196078, 0.25098039, 0.50196078],
                               [0.95686275, 0.1372549 , 0.90980392],
                               [0.98039216, 0.66666667, 0.62745098],
                               [0.90196078, 0.58823529, 0.54901961],
                               [0.2745098 , 0.2745098 , 0.2745098 ],
                               [0.4       , 0.4       , 0.61176471],
                               [0.74509804, 0.6       , 0.6       ],
                               [0.70588235, 0.64705882, 0.70588235],
                               [0.58823529, 0.39215686, 0.39215686],
                               [0.58823529, 0.47058824, 0.35294118],
                               [0.6       , 0.6       , 0.6       ],
                               [0.6       , 0.6       , 0.6       ],
                               [0.98039216, 0.66666667, 0.11764706],
                               [0.8627451 , 0.8627451 , 0.        ],
                               [0.41960784, 0.55686275, 0.1372549 ],
                               [0.59607843, 0.98431373, 0.59607843],
                               [0.2745098 , 0.50980392, 0.70588235],
                               [0.8627451 , 0.07843137, 0.23529412],
                               [1.        , 0.        , 0.        ],
                               [0.        , 0.        , 0.55686275],
                               [0.        , 0.        , 0.2745098 ],
                               [0.        , 0.23529412, 0.39215686],
                               [0.        , 0.        , 0.35294118],
                               [0.        , 0.        , 0.43137255],
                               [0.        , 0.31372549, 0.39215686],
                               [0.        , 0.        , 0.90196078],
                               [0.46666667, 0.04313725, 0.1254902 ]
                               ])
    

    ## save three images. source image, seg from cyclegan, our augmented results
    root = '/home/lthpc/data_calibrator/segimages'
    ori_img_path = os.path.join(root,'ori_img_{}.png'.format(id))
    cycle_out_path = os.path.join(root,'cycle_out_{}.png'.format(id))
    our_out_path = os.path.join(root,'our_out_{}.png'.format(id))        
    save_image(ori_img,ori_img_path)
    
    save_image(convert(cycle_out,color_map),cycle_out_path)
    save_image(convert(our_out,color_map),our_out_path)    

'''

def fmt_array(arr, fmt=','):
    strs = ['{:.3f}'.format(x) for x in arr]
    return fmt.join(strs)


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def result_stats(hist):
    acc_overall = np.diag(hist).sum() / hist.sum() * 100
    acc_percls = np.diag(hist) / (hist.sum(1) + 1e-8) * 100
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-8) * 100
    freq = hist.sum(1) / hist.sum()
    fwIU = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc_overall, acc_percls, iu, fwIU

def calculate_params(net):
    weight_sum = 0    
    for name,w in net.named_parameters():
        print (name)
        w = w.cpu().detach().numpy()
        weight_sum += w.size
    print ('overall parameter {}'.format(weight_sum))


@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--dataset', default='cityscapes',
              type=click.Choice(dataset_obj.keys()))
@click.option('--datadir', default='',
              type=click.Path(exists=True))
@click.option('--model', default='fcn8s', type=click.Choice(models.keys()))
@click.option('--gpu', default='0')
@click.option('--num_cls', default=19)
def main(path, dataset, datadir, model, gpu, num_cls):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    
    net = get_model('CalibratorNet', model=model, num_cls=num_cls, weights_init=path,cali_model = 'resnet_9blocks',task ='segmentation')
    #net = get_model(model, num_cls=num_cls, weights_init=path)
    #net.load_state_dict(torch.load(path))

    net.eval()

    transform = net.src_net.module.transform if hasattr(net.src_net,'module') else net.src_net.transform
    
    ds = get_fcn_dataset(dataset, datadir, split='val',
                         transform=transform, target_transform=to_tensor_raw)
    #ds = get_fcn_dataset(dataset, datadir, split='val',
    #                     transform=torchvision.transforms.ToTensor(), target_transform=to_tensor_raw)    
    classes = ds.classes
    loader = torch.utils.data.DataLoader(ds, num_workers=8)

    intersections = np.zeros(num_cls)
    unions = np.zeros(num_cls)

    errs = []
    hist = np.zeros((num_cls, num_cls))
    if len(loader) == 0:
        print('Empty data loader')
        return
    iterations = tqdm(enumerate(loader))
    count = 0
    res = []

    
    with torch.no_grad():
        for im_i, (im, label) in iterations:

            im = Variable(im.cuda())
            pert = net.calibrator_T(im)
            #pert = torch.clamp(pert,0,0)
            score = net.src_net(torch.clamp(im+pert,-3,3)).data

            max_score = torch.argmax(score,dim=1)
            #max_ori_score = torch.argmax(net(im).data,dim=1)

            #res.append((max_score,count))
            #save_seg_results(max_ori_score,count)
            #save_seg_results(im,max_score,max_ori_score,count)


            _, preds = torch.max(score, 1)
            hist += fast_hist(label.numpy().flatten(),
                              preds.cpu().numpy().flatten(),
                              num_cls)
            acc_overall, acc_percls, iu, fwIU = result_stats(hist)
            iterations.set_postfix({'mIoU': ' {:0.2f}  fwIoU: {:0.2f} pixel acc: {:0.2f} per cls acc: {:0.2f}'.format(
                np.nanmean(iu), fwIU, acc_overall, np.nanmean(acc_percls))})

            count+=1


    

            

    print()
    print(','.join(classes))
    print(fmt_array(iu))
    print(np.nanmean(iu), fwIU, acc_overall, np.nanmean(acc_percls))
    print()
    print('Errors:', errs)


if __name__ == '__main__':
    main()
