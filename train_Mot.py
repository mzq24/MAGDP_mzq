from __future__ import print_function
import os
import sys
import shutil
import time
import pandas as pd
from datetime import date
import argparse
import pickle
import pprint
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch_geometric.loader import DataLoader

from dataset_may17 import SimAgn_Dataset, flat_Batch, retrieve_mask
# from mtploss import MTPLoss
from data_utils import convert_goalsBatch_to_goalsets_for_scenarios

from tqdm import tqdm

def train_a_model(model_to_tr, train_loader, loss_func, num_ep=1, num_batch=-1):
    model_to_tr.train()
    running_loss = 0.0
    train_time_tic = time.time()
    for d_idx, data in tqdm(enumerate(train_loader)):
    # for i, data in enumerate(train_loader):
        data = flat_Batch(data).to(args['device'])
        # print(data)
        # print(data.flat_node_type)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        mm_traj_pred = model_to_tr(data)
        batch_loss =loss_func(mm_traj_pred, data.mot_label)

        # batch_loss = sum([loss_func(goal_pred[i], ground_truth_goalsets[i]) for i in range(len(goal_pred))])
        # print(batch_loss)
        batch_loss.backward()
        a = torch.nn.utils.clip_grad_norm_(model_to_tr.parameters(), 1)
        optimizer.step()
        running_loss += batch_loss.item()

        if num_batch>-1 and d_idx>=num_batch:
            break
    return round(running_loss/(d_idx+1), 2)

if __name__ == '__main__':
    from Mot_model import Mot_net
    from eval_Mot import eval_a_model
    # 
    from mtp_loss import mtp_loss
    
    from datetime import datetime
    args={}
    args['enc_hidden_size'] = 128
    args['dec_hidden_size'] = 256
    args['enc_embed_size']  = 256
    args['enc_gat_size']    = 256
    args['num_gat_head'] = 3
    args['num_modes'] = 6
    args['batch_size'] = 64

    args['train_split'] = 0.9
    args['num_batch'] = -1
    args['eval_num_batch'] = -1
    args['device'] = 'cuda:1'
    args['dec_type'] = 'Mot' 
    train_net = Mot_net(args)
    train_net.to(args['device'])

    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d-%m-%Y_%H%M%S")
    f=open(f"./outs/MGA_print_{dt_string}.txt","w+")
    print(train_net, file=f, flush=True)
    print(train_net)

    pytorch_total_params = sum(p.numel() for p in train_net.parameters())
    print('number of parameters: {}'.format(pytorch_total_params), file=f, flush=True)
    print('number of parameters: {}'.format(pytorch_total_params))

    optimizer = torch.optim.AdamW(train_net.parameters(), lr=0.0001, weight_decay=0.01) 
    scheduler = MultiStepLR(optimizer, milestones=[20, 22, 24, 26, 28], gamma=0.5)

    #################################
    myhost = os.uname()[1]
    if myhost == 'AutoManRRCServer':
        data_path = '/disk2/SimAgent_Dataset/pyg_data_may17/training'
    elif myhost == 'asp2a-login-ntu01':
        data_path = '/disk2/SimAgent_Dataset/pyg_data_may17/training'
    else: # NSCC
        data_path = '/home/users/ntu/baichuan/scratch/sim/TGE/pyg_data_full_may17/training'

    full_train_set = SimAgn_Dataset(data_path=data_path, dec_type=args['dec_type']) 
    train_size = int(full_train_set.__len__() * args['train_split'])
    val_size   = full_train_set.__len__() - train_size
    # val_size   = 10000
    print('train_size: {}/{}, val_size: {}/{}'.format(train_size, full_train_set.__len__(), val_size, full_train_set.__len__()), file=f, flush=True)
    print('train_size: {}/{}, val_size: {}/{}'.format(train_size, full_train_set.__len__(), val_size, full_train_set.__len__()))
    train_set, val_set = torch.utils.data.random_split(full_train_set, [train_size, val_size])
    trainDataloader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=4) # num_workers=6, 
    valDataloader   = DataLoader(val_set,   batch_size=args['batch_size'], shuffle=True, num_workers=4) # num_workers=6, 
    #################################

    tic = time.time()
    Val_LOSS = []
    Train_LOSS = []
    min_val_loss = 100.0

    start_ep = 1

    args['train_epoches'] = 50
    # loss_function = MTPLoss(args['num_modes'], 1, 5)

    for ep in range(start_ep, args['train_epoches']+1):
        train_time_tic = time.time()
        train_loss_ep = train_a_model(train_net, trainDataloader, loss_func=mtp_loss, num_ep=ep,  num_batch=args['num_batch'])
        eval_loss_ep  = eval_a_model(train_net,  valDataloader,   loss_func=mtp_loss, num_batch=args['eval_num_batch'])
        scheduler.step()

        model_type = args['dec_type']+'M{}'.format(args['num_modes'])
        if ep%10 == 9:
            torch.save(train_net, f'./models/{model_type}-EP{ep}-Loss{eval_loss_ep[0]}-minFDE{eval_loss_ep[1]}.ckpt')
        if ep ==1:
            torch.save(train_net, f'./models/{model_type}-EP{ep}-Loss{eval_loss_ep[0]}-minFDE{eval_loss_ep[1]}.ckpt')
        

        ep_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(f'{model_type}, ep {ep}, MTP loss train {train_loss_ep}, eval {eval_loss_ep[0]}, minADE {eval_loss_ep[1]}, minFDE {eval_loss_ep[2]}, [ lr= {ep_lr} ]')
        print(f'{model_type}, ep {ep}, MTP loss train {train_loss_ep}, eval {eval_loss_ep[0]}, minADE {eval_loss_ep[1]}, minFDE {eval_loss_ep[2]}, [ lr= {ep_lr} ]', file=f, flush=True)
        # torch.save(train_net, f'./models/mtpgoal-{ep}.ckpt')

        
