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
sys.path.append('../')

from dataset_may17 import SimAgn_Dataset, flat_Batch, retrieve_mask
from tqdm import tqdm
def train_a_model(model_to_tr, train_loader, loss_func, num_ep=1, num_batch=-1):
    model_to_tr.train()
    running_loss = 0.0
    for d_idx, data in tqdm(enumerate(train_loader)):
        # print(i)
        data = flat_Batch(data)
        data = data.to(args['device'])
        # print(torch.max(data.node_feature))

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        fut_pred = model_to_tr(data)
        # print(f'fut_pred {fut_pred.shape}')
        # print(f'agn_goal_set {data.gdp_label.shape}')
        # loss_func(fut_pred[:,:,model_to_eval.args['out_dim']], data.gdp_label[:,:,model_to_eval.args['out_dim']])
        batch_loss = loss_func(fut_pred, data.gdp_label)
        batch_loss.backward()
        # print(batch_loss, running_loss)
        a = torch.nn.utils.clip_grad_norm_(model_to_tr.parameters(), 1)
        optimizer.step()

        # print statistics
        running_loss += batch_loss.item()
        if num_batch>-1 and d_idx>=num_batch:
            break
        # running_loss.append(batch_loss.item())
    # print(ep, round(running_loss/(d_idx+1),2))
    return round(running_loss/(d_idx+1),2)

def save_obj_pkl(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    from GDP_model import GDP_net
    from eval_GDP import eval_a_model
    args={}
    args['enc_hidden_size'] = 128
    args['dec_hidden_size'] = 256
    args['enc_embed_size']  = 256
    args['enc_gat_size']    = 256
    args['num_gat_head'] = 3
    args['out_dim'] = 6

    args['train_split'] = 0.9
    args['batch_size'] = 32
    args['train_epoches'] = 50
    args['num_batch'] = -1
    args['eval_num_batch'] = -1
    args['dec_type'] = 'GDP' # GDPrnn GDP
    args['dec_feat'] = 'GDLI' #  GDLIT GD : [Goal + Dyn],  GDI : [Goal + Dyn + Int],  GDIT : [Goal + Dyn + Int + TCL]
    args['feat_attention'] = True # True


    #################################
    myhost = os.uname()[1]
    if myhost == 'AutoManRRCServer':
        data_path = '/disk6/SimAgent_Dataset/pyg_data_Jun23/validation'
        args['device'] = 'cuda:2' 
    elif myhost == 'amrrc':
        data_path = '/home/xy/SimAgent_Dataset/pyg_data_Jun23/training'
        args['device'] = 'cuda:0' 
    elif myhost == 'xy-Legion':
        train_data_path = '/home/xy/sim/SimAgent_Dataset/pyg_data_Sep/validation_80'
        eval_data_path  = '/home/xy/sim/SimAgent_Dataset/pyg_data_Sep/validation_5'
        args['device'] = 'cuda:0' 
    else: # NSCC
        data_path = '/home/users/ntu/baichuan/scratch/sim/SimAgent_Dataset/pyg_data_Jun23/training'
        args['device'] = 'cuda:0' 

    

    train_net = GDP_net(args)
    train_net.to(args['device'])
    print(train_net)

    pytorch_total_params = sum(p.numel() for p in train_net.parameters())
    print('number of parameters: {}'.format(pytorch_total_params))

    
    # pp.pprint(args)
    
    ## Initialize optimizer and the scheduler

    optimizer = torch.optim.AdamW(train_net.parameters(), lr=0.0001, weight_decay=0.01) 
    scheduler = MultiStepLR(optimizer, milestones=[20, 22, 24, 26, 28], gamma=0.5)

    train_set = SimAgn_Dataset(data_path=train_data_path, dec_type=args['dec_type']) 
    val_set = SimAgn_Dataset(data_path=eval_data_path, dec_type=args['dec_type']) 
    print('train_size: {}, val_size: {}'.format(train_set.__len__(), val_set.__len__()))
    trainDataloader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=12) # num_workers=6, 
    valDataloader   = DataLoader(val_set,   batch_size=args['batch_size'], shuffle=True, num_workers=12) # num_workers=6, 
    #################################

    tic = time.time()
    Val_LOSS = []
    Train_LOSS = []
    min_val_loss = 100.0

    
    # gdp_loss_func = torch.nn.MSELoss(reduction='mean') # SmoothL1Loss MSELoss
    # from torch.nn import functional as f
    from gdp_loss import weighted_gdp_loss
    gdp_loss_func = weighted_gdp_loss
    # loss_func_ce = cross_entropy_loss
    start_ep = 1
    
    for ep in range(start_ep, args['train_epoches']+1):
        # print(ep)
        train_time_tic = time.time()
        train_loss_ep = train_a_model(train_net, trainDataloader, gdp_loss_func, num_ep=ep, num_batch=args['num_batch'])
        eval_loss_ep  = eval_a_model( train_net, valDataloader,   gdp_loss_func,            num_batch=args['eval_num_batch'])
        scheduler.step()

        # eval_loss_ep = 0
        # if ep%10 == 9:
        #     torch.save(train_net, f'./models/GDP-EP{ep}-Loss{eval_loss_ep}.ckpt')
        model_type = args['dec_type'] + 'out' + str(args['out_dim']) + args['dec_feat']
        # model_type = args['dec_type'] + '-M{}'.format(args['num_modes'])
        if args['feat_attention']:
            model_type += '-attn'
        # if 
        if eval_loss_ep < min_val_loss:
            if os.path.exists(f'./models/{model_type}-Loss{min_val_loss}.ckpt'):
                os.remove(f'./models/{model_type}-Loss{min_val_loss}.ckpt') 
            torch.save(train_net, f'./models/{model_type}-Loss{eval_loss_ep}.ckpt')
            min_val_loss = eval_loss_ep
        print('-'*66)
        

        ep_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(f'{model_type} Ep-{ep}, SmoothL1Loss train {train_loss_ep}, eval {eval_loss_ep} [ lr= {ep_lr} ]')
        print('-'*66)