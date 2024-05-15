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
sys.path.append('../utils')
from dataset_may17 import SimAgn_Dataset, flat_Batch, retrieve_mask
# from mtploss import MTPLoss
from data_utils import convert_goalsBatch_to_goalsets_for_scenarios

from tqdm import tqdm

def train_a_model(model_to_tr, train_loader, loss_func, num_ep=1, num_batch=-1):
    model_to_tr.train()
    running_loss = 0.0
    train_time_tic = time.time()
    for d_idx, data in tqdm(enumerate(train_loader)):
    # for d_idx, data in enumerate(train_loader):
        data = flat_Batch(data).to(args['device'])
        ground_truth_goalsets = convert_goalsBatch_to_goalsets_for_scenarios(data.agn_goal_set, data.num_goal_valid_agn)
        # print(data)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        goal_pred = model_to_tr(data)

        # if model_to_tr.args['loss_type'] == 'Joint':
        #     batch_loss = sum([loss_func(goal_pred[i], ground_truth_goalsets[i], num_modes=model_to_tr.args['num_modes']) for i in range(len(goal_pred))])/len(goal_pred)
        # if model_to_tr.args['loss_type'] == 'Marginal':
        #     batch_loss = loss_func(torch.cat([goal_pred[i] for i in range(len(goal_pred))]), 
        #                            data.agn_goal_set, 
        #                            num_modes=model_to_tr.args['num_modes'])
        
        batch_loss = sum([loss_func(goal_pred[i], ground_truth_goalsets[i], num_modes=model_to_tr.args['num_modes']) for i in range(len(goal_pred))])/len(goal_pred)
        
        # loss_out = [loss_func(goal_pred[i], ground_truth_goalsets[i], num_modes=model_to_tr.args['num_modes'])  for i in range(len(goal_pred))]
        # print(loss_out)
        # batch_loss = sum([torch.min(l) for l in loss_out])/len(goal_pred)

        # best_modes = [l[1] for l in loss_out]
        # batch_loss_ccl  = sum([ Goals_distance_to_CCLs(goal_pred[i][:,best_modes[i],:], data.agn_CCLs[i]) for i in range(len(goal_pred))])/len(goal_pred)

        ## 测试 dist2CCLs loss
        # print(batch_loss_goal)
        # print(f'batch_loss_ccl {batch_loss_ccl}')
        # print(len(data.agn_CCLs[0]))
        # Goals_distance_to_CCLs(goal_pred[0], data.agn_CCLs[0])


        # batch_loss = batch_loss_goal + 0.1*batch_loss_ccl
        # batch_loss = batch_loss_ccl
        # batch_loss = batch_loss_goal
        # print(batch_loss)
        batch_loss.backward()
        a = torch.nn.utils.clip_grad_norm_(model_to_tr.parameters(), 1)
        optimizer.step()
        running_loss += batch_loss.item()

        if num_batch>-1 and d_idx>=num_batch:
            break
    return round(running_loss/(d_idx+1), 2)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss_type", type=str, default='Joint', help="feature used for decoding", choices=['Joint', 'Marginal'])
    opt = parser.parse_args()
    # print(opt)
    from MGA_model import MGA_net
    from eval_MGA import eval_a_model

    from datetime import datetime

    args={}
    args['enc_hidden_size'] = 128
    args['dec_hidden_size'] = 256
    args['enc_embed_size']  = 256
    args['enc_gat_size']    = 256
    args['num_gat_head'] = 3
    args['num_modes'] = 6
    args['batch_size'] = 32

    args['train_split'] = 0.9
    args['num_batch']   =  -1  # -1 # -1
    args['eval_num_batch'] = 300 # 300
    args['dec_type'] = 'MGA' # 'mtpGoal' # 'mtp_goal_veh'
    args['feat_attention'] = True

    args['loss_type'] = opt.loss_type # 'Joint'

    if args['loss_type'] == 'Joint':
        from mtp_goal_loss import mtp_goal_loss
        loss_func_for_training = mtp_goal_loss
    elif args['loss_type'] ==  'Marginal':
        from mtp_goal_loss import marginal_mtp_goal_loss
        loss_func_for_training = marginal_mtp_goal_loss
    else:
        pass
    

    #################################
    myhost = os.uname()[1]
    if myhost == 'AutoManRRCServer':
        data_path = '/disk6/SimAgent_Dataset/pyg_data_Jun23/training'
        args['device'] = 'cuda:1' 
    elif myhost == 'amrrc':
        data_path = '/home/xy/SimAgent_Dataset/pyg_data_Jun23/training'
        args['device'] = 'cuda:1' 
    elif myhost == 'xy-Legion':
        train_data_path = '/home/xy/sim/SimAgent_Dataset/pyg_data_Jun23/validation_0-140'
        eval_data_path  = '/home/xy/sim/SimAgent_Dataset/pyg_data_Jun23/validation_140-145'
        args['device'] = 'cuda:0' 
    else: # NSCC
        data_path = '/home/users/ntu/baichuan/scratch/sim/SimAgent_Dataset/pyg_data_Jun23/training'
        args['device'] = 'cuda:0' 
    # args['use_feat'] = opt.use_feat # N, D, L, DL . N: nothing, D: Dyn only, L: CCLs only, DL: C-A, AA, D+L: Dyn+CCL

    # args['device'] = 'cpu' 
    train_net = MGA_net(args)
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

    

    train_set = SimAgn_Dataset(data_path=train_data_path, dec_type=args['dec_type']) 
    val_set = SimAgn_Dataset(data_path=eval_data_path, dec_type=args['dec_type']) 
    # train_size = int(full_train_set.__len__() * args['train_split'])
    # val_size   = full_train_set.__len__() - train_size
    # val_size   = 10000
    # print('train_size: {}/{}, val_size: {}/{}'.format(train_size, full_train_set.__len__(), val_size, full_train_set.__len__()), file=f, flush=True)
    print('train_size: {}, val_size: {}'.format(train_set.__len__(), val_set.__len__()))
    # train_set, val_set = torch.utils.data.random_split(full_train_set, [train_size, val_size])
    trainDataloader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=12) # num_workers=6, 
    valDataloader   = DataLoader(val_set,   batch_size=args['batch_size'], shuffle=True, num_workers=12) # num_workers=6, 
    #################################

    tic = time.time()
    Val_LOSS = []
    Train_LOSS = []
    min_val_loss = 100.0

    start_ep = 1

    args['train_epoches'] = 50
    for ep in range(start_ep, args['train_epoches']+1):
        train_time_tic = time.time()
        train_loss_ep = train_a_model(train_net, trainDataloader, loss_func=loss_func_for_training, num_ep=ep,  num_batch=args['num_batch'])
        eval_loss_ep  = eval_a_model(train_net,  valDataloader,   loss_func=loss_func_for_training, plot=False, num_batch=args['eval_num_batch'])
        scheduler.step()

        model_type = args['dec_type'] + '-M{}_'.format(args['num_modes']) + args['loss_type']
        if args['feat_attention']:
            model_type += '-attn'
        # if ep%10 == 9:
        if eval_loss_ep[2] < min_val_loss:
            if os.path.exists(f'./models/{model_type}-minFDE{min_val_loss}.ckpt'):
                os.remove(f'./models/{model_type}-minFDE{min_val_loss}.ckpt') 
            torch.save(train_net, f'./models/{model_type}-minFDE{eval_loss_ep[2]}.ckpt')
            min_val_loss = eval_loss_ep[2]
        # if ep ==1:
            # torch.save(train_net, f'./models/{model_type}-EP{ep}-Loss{eval_loss_ep[0]}-minFDE{eval_loss_ep[1]}.ckpt')
        

        ep_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(f'{model_type}, ep {ep}, SmoothL1 loss train {train_loss_ep}, eval {eval_loss_ep[0]}, min-Joint-FDE {eval_loss_ep[1]}, Marginal minFDE {eval_loss_ep[2]}, [ lr= {ep_lr} ]')
        print(f'{model_type}, ep {ep}, SmoothL1 loss train {train_loss_ep}, eval {eval_loss_ep[0]}, min-Joint-FDE {eval_loss_ep[1]}, Marginal minFDE {eval_loss_ep[2]}, [ lr= {ep_lr} ]', file=f, flush=True)
        # torch.save(train_net, f'./models/mtpgoal-{ep}.ckpt')

        
