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
from tqdm import tqdm

from dataset_may17 import SimAgn_Dataset, flat_Batch, retrieve_mask


def train_a_model(model_to_tr, train_loader, loss_func, num_ep=1, num_batch=-1):
    model_to_tr.train()
    running_loss = 0.0
    data_load_tic = time.time()
    # for i, data in enumerate(train_loader):
    for d_idx, data in tqdm(enumerate(train_loader)):
        # print(i)
        # print('\n', i)
        
        data = flat_Batch(data)
        data = data.to(args['device'])
        data_load_tac = time.time()
        # print(f'used {data_load_tac-data_load_tic} sec to load a batch of {train_loader.batch_size} data')
        # print(torch.max(data.node_feature))

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        forward_tic = time.time()
        fut_pred = model_to_tr(data)
        
        # print(f'used {time.time()-forward_tic} sec to Forward a batch of {train_loader.batch_size} data')
        # batch_loss = goal_est_loss_func(fut_pred, data.agn_goal_set)
        # print(f'fut_pred {fut_pred.shape}, agn_goal_set {data.agn_goal_set.shape}')
        loss_tic = time.time()
        # batch_loss = loss_func(data.agn_goal_set, fut_pred)
        if model_to_tr.args['dec_type']  == 'goal_est_2':
            batch_loss = loss_func(fut_pred, data.agn_goal_set)
        else:
            batch_loss = loss_func(fut_pred.unsqueeze(1), data.agn_goal_set.unsqueeze(1))

        # print(f'batch_loss {batch_loss}, mtr_loss{mtr_loss}')
        # print(f'used {time.time()-loss_tic} sec to Calculate Loss a batch of {train_loader.batch_size} data')
        
        # print(f'\nfut_pred is nan {torch.isnan(fut_pred).any()}, agn_goal_set is nan {torch.isnan(data.agn_goal_set).any()}')
        # print(f'fut_pred {fut_pred}, agn_goal_set {data.agn_goal_set}')
        # print(f'loss: {batch_loss}')
        a = torch.nn.utils.clip_grad_norm_(model_to_tr.parameters(), 1)
        
        backward_tic = time.time()
        batch_loss.backward()
        optimizer.step()
        # print(f'used {time.time()-backward_tic} sec to Backward a batch of {train_loader.batch_size} data\n')
        # if i> 2:
        #     break
        # # print statistics
        running_loss += batch_loss.item()
        # running_loss.append(batch_loss.item())
        data_load_tic = time.time()
        if num_batch>-1 and d_idx>=num_batch:
            break
    return round(running_loss/(d_idx+1),2)

def save_obj_pkl(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    from datetime import datetime
    from TGE_model import TGE_net
    from eval_TGE import eval_a_model
    args={}
    # args['encoder_path'] = '../models/mtpgoal-199.ckpt'
    args['goal_dim'] = 2
    args['enc_hidden_size'] = 128
    args['dec_hidden_size'] = 256
    args['enc_embed_size']  = 256
    args['enc_gat_size']    = 256
    args['num_gat_head'] = 3

    args['batch_size'] = 256
    args['num_gat_head'] = 3
    args['device'] = 'cuda:0'
    args['train_epoches'] = 50
    args['train_split'] = 0.9
    args['eval_num_batch'] = 100
    args['dec_type'] = 'TGE'
    
    train_net = TGE_net(args)
    train_net.to(args['device'])
    print(train_net)

    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d-%m-%Y_%H%M%S")
    f=open(f"./outs/TGE_print_{dt_string}.txt","w+")
    print(train_net, file=f, flush=True)

    pytorch_total_params = sum(p.numel() for p in train_net.parameters())
    print('number of parameters: {}'.format(pytorch_total_params))
    print('number of parameters: {}'.format(pytorch_total_params), file=f, flush=True)
    
    pytorch_total_trainable_params = sum(p.numel() for p in train_net.parameters() if p.requires_grad)
    print('number of trainable parameters: {}'.format(pytorch_total_trainable_params))
    print('number of trainable parameters: {}'.format(pytorch_total_trainable_params))
    

    # pp.pprint(args)
    if args['goal_dim'] == 2:
        goal_est_loss_func = gdp_loss_func = torch.nn.SmoothL1Loss(reduction='mean')
    elif args['goal_dim'] == 5:
        from nll_loss import MTR_nll_loss_gmm_direct
        goal_est_loss_func = MTR_nll_loss_gmm_direct
        ## Initialize optimizer and the scheduler

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
    print('train_size: {}/{}, val_size: {}/{}'.format(train_size, full_train_set.__len__(), val_size, full_train_set.__len__()), file=f, flush=True)
    print('train_size: {}/{}, val_size: {}/{}'.format(train_size, full_train_set.__len__(), val_size, full_train_set.__len__()))
    train_set, val_set = torch.utils.data.random_split(full_train_set, [train_size, val_size])
    trainDataloader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=32) # num_workers=6, 
    valDataloader   = DataLoader(val_set,   batch_size=args['batch_size'], shuffle=True, num_workers=32) # num_workers=6, 
    #################################

    

    tic = time.time()
    Val_LOSS = []
    Train_LOSS = []
    min_val_loss = 100.0


    start_ep = 1

    for ep in range(start_ep, args['train_epoches']+1):
        train_time_tic = time.time()
        train_loss_ep = train_a_model(train_net, trainDataloader, goal_est_loss_func, num_ep=ep, num_batch=-1)
        eval_loss_ep  = eval_a_model( train_net, valDataloader,   goal_est_loss_func,            num_batch=args['eval_num_batch'])
        scheduler.step()

        

        print('-'*66)
        goal_dim235 = args['goal_dim']
        ep_lr = optimizer.state_dict()['param_groups'][0]['lr']
        loss_type = 'SmoothL1Loss' if args['goal_dim'] == 2 else 'NLLloss'
        model_type = args['dec_type']
        print(f'{model_type}-{goal_dim235}, Ep-{ep}, [{loss_type} = train: {train_loss_ep}, val: {eval_loss_ep[0]}, FDE: {eval_loss_ep[1]} ] , [ lr= {ep_lr} ]', file=f, flush=True)
        print(f'{model_type}-{goal_dim235}, Ep-{ep}, [{loss_type} = train: {train_loss_ep}, val: {eval_loss_ep[0]}, FDE: {eval_loss_ep[1]} ] , [ lr= {ep_lr} ]')
        print('-'*66)

        # if ep%10 == 9:
        #     torch.save(train_net, f'./models/TDE{goal_dim235}-EP{ep}-FDE{eval_loss_ep[1]}.ckpt')
        # if ep ==1:
        torch.save(train_net, f'./models/TGE{goal_dim235}-EP{ep}-FDE{eval_loss_ep[1]}.ckpt')




    
