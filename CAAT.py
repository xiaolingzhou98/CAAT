from __future__ import print_function

from utils import *
from pre_resnet10 import PreActResNet18

import argparse
import torch
import torch.optim as optim
from MLP import *
from meta import *
from NLT_CIFAR import *
import os
import time




def main(args):
    time_start = time.time()
    if args.model == 'PreResNet18':
        h_net = PreActResNet18().cuda()
    elif args.model == 'WRN28':
        h_net = WideResNet().cuda()

    h_net.load_state_dict(torch.load('hot_start.pt'))
    
    
    ds_train, ds_valid, ds_test, imbalanced_num_list = build_dataloader(
        seed=args.seed,
        dataset=args.dataset,
        num_meta_total=args.num_meta,
        imbalanced_factor=args.imbalanced_factor,
        corruption_type=args.corruption_type,
        corruption_ratio=args.corruption_ratio,
        batch_size=args.batch_size,
        meta_batch=args.batch_size,
    )
    class_rate = torch.from_numpy(np.array(imbalanced_num_list)/sum(imbalanced_num_list)).cuda().float()
    
    ## other layer optimizer
    optimizer = optim.SGD(h_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    lr = args.lr
    
    
    meta_net = MLP(in_size = 6).cuda()
    meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=0.0001, weight_decay=5e-4)

    ## attack parameters during test
    configs =  {
    'epsilon': 10/255,
    'num_steps': 10,
    'step_size': 2/255, 
    'clip_max': 1,
    'clip_min': 0
    }

    configs1 =  {
    'epsilon': 8/255,
    'num_steps': 20,
    'step_size': 2/255,
    'clip_max': 1,
    'clip_min': 0
    }

    ### main training loop
    maxepoch = args.epoch
    device = 'cuda'
    beta = args.beta
    rate1 = args.rate1
    rate2 = args.rate2
    lim = args.lim
    delta0 = args.bound0 * torch.ones(10)   ## fair constraints 
    delta1 = args.bound1 * torch.ones(10)   ## fair constraints
    lmbda = torch.zeros(30)

    REPORT = []
    REPORT1 = []
    loss_tensor_last = torch.ones([10]).cuda()*5
    for now_epoch in range(1, maxepoch + 1):
        lr = args.lr * ((0.1 ** int(now_epoch >= 40)) * (0.1 ** int(now_epoch >= 80)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        if now_epoch % 40 == 0:
            rate1 = rate1 / 2

        lmbda = frl_train(class_rate, loss_tensor_last, lr, h_net, ds_train, ds_valid, ds_test, meta_net, meta_optimizer, optimizer, now_epoch, configs,
                              configs1, device, delta0, delta1, rate1, rate2, lmbda, beta, lim)
        
        print('................................................................................')
    time_end = time.time()
    time_sum= time_end-time_start
    print(time_sum)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=40)
    argparser.add_argument('--batch_size', type=int, help='batch_size', default=128)
    argparser.add_argument('--seed', type=int, help='random seed', default=100)
    argparser.add_argument('--beta', help='trade off parameter', type = float, default=1.5)
    argparser.add_argument('--model', help='model structure', default='PreResNet18')
    argparser.add_argument('--bound0', type=float, help='fair constraints for clean error', default=0.07)
    argparser.add_argument('--bound1', type=float, help='fair constraints for bndy error', default=0.07)
    argparser.add_argument('--rate1', type=float, help='hyper-par update rate', default=0.05)
    argparser.add_argument('--rate2', type=float, help='hyper-par update rate', default=0.2)
    argparser.add_argument('--lim', type=float, default=0.5)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--imbalanced_factor', type=int, default=1)
    argparser.add_argument('--corruption_type', type=str, default=None)
    argparser.add_argument('--corruption_ratio', type=float, default=0.)
    argparser.add_argument('--dataset', type=str, default='cifar10')
    argparser.add_argument('--num_meta', type=int, default=3000)
    args = argparser.parse_args()
    print(args)
    main(args)
