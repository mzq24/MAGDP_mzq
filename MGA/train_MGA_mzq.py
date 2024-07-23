from __future__ import print_function
from calendar import c
import os
import sys
import time
import argparse
import pickle
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch_geometric.loader import DataLoader
import json
import numpy as np
import logging
from torch import nn
from MGA_model_mzq import MGA_net
from eval_MGA import eval_a_model
from datetime import datetime
from dataset_mzq import SimAgn_Dataset, flat_Batch, retrieve_mask
from utils.data_utils import convert_goalsBatch_to_goalsets_for_scenarios
from tqdm import tqdm
from loss_ma_goal_joint_collision_offroad_position import joint_mag_loss_w_inter_group_position

sys.path.append('../')
sys.path.append('/home/xy/mzq/code/MAGDP')
sys.path.append('/home/xy/mzq/code/MAGDP/MGA')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def train_a_model(model_to_tr, train_loader, loss_func, num_ep=1, num_batch=-1, jp_loss=False):
    model_to_tr.train()
    running_loss = 0.0
    train_time_tic = time.time()
    for d_idx, data in tqdm(enumerate(train_loader)):
        data = flat_Batch(data).to(args['device'])
        ground_truth_goalsets = convert_goalsBatch_to_goalsets_for_scenarios(data.agn_goal_set, data.num_goal_valid_agn)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        goal_pred = model_to_tr(data)
        batch_loss = sum([loss_func(goal_pred[i], ground_truth_goalsets[i], num_modes=model_to_tr.args['num_modes']) for i in range(len(goal_pred))])/len(goal_pred)

        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model_to_tr.parameters(), 1)
        optimizer.step()
        running_loss += batch_loss.item()

        if num_batch>-1 and d_idx>=num_batch:
            break
    return round(running_loss/(d_idx+1), 2)

def set_ckpt_name_old(ckpt_folder_path: str, args: dict, min_val_loss: float, threshold: float, dist_threshold: float):
    model_type = args['dec_type'] + '_M{}_'.format(args['num_modes']) + args['loss_type']
    if args['feat_attention']:
        model_type += '_attn'
    use_a = args['use_attention']
    model_name = f'{ckpt_folder_path}/{model_type}_minJDE{min_val_loss}_thr{threshold}_attenc{int(use_a)}_distthr{dist_threshold}.ckpt'
    return model_name

def set_ckpt_name(ckpt_folder_path: str, file_name: str, min_val_loss: float):
    model_name = f'{ckpt_folder_path}/{file_name}_minJDE{min_val_loss}.ckpt'
    return model_name

def init_log(train_net):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler('logfile.log')
    logger.addHandler(handler)
    train_net.set_logger(logger)
    return logger

def set_log_name_old(log_folder_path: str, args: dict, dt_string: str, threshold: float, dist_threshold: float):
    model_type = args['dec_type'] + '-M{}_'.format(args['num_modes']) + args['loss_type']
    use_a = args['use_attention']
    log_name = f"{log_folder_path}/MGA_print_{dt_string}_{model_type}_thr{threshold}_attenc{int(use_a)}_distthr{dist_threshold}.txt"
    return log_name

def set_log_name(log_folder_path: str, dt_string: str):
    log_name = f"{log_folder_path}/MGA_{dt_string}.txt"
    return log_name

def find_files_with_params(model_path, gnn_type, loss_type, thr_number, attenc_number):
    '''
    param:
        model_path: 'MAGDP/MGA/models'
        gnn_type: 'GATConv', 'GATv2Conv', 'TransformerConv'
        loss_type: 'Joint', 'Marginal'
        thr_number: 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6
        attenc_number: 0, 1
    '''
    folders = [os.path.join(model_path, folder) for folder in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, folder))]
    file_paths = []

    for folder in folders:
        if gnn_type in folder:
            files = os.listdir(folder)
            for file in files:
                filename_without_ext, ext = os.path.splitext(file)
                # Split the file name using '-' and '_'
                parts = filename_without_ext.split('-')
                parts = [elem.split('_') for elem in parts]
                subparts = [subpart for part in parts for subpart in part]
                if check_name_from_subparts(subparts, loss_type=loss_type, thr_number=thr_number, attenc_number=attenc_number):
                    file_paths.append(os.path.join(folder, file))
            else:
                # Add the file path to the list if it does not contain a valid threshold
                continue
    
    return file_paths

def check_name_from_subparts(subparts, loss_type='Joint', thr_number=0.2, attenc_number=0):
    contains_joint = False if loss_type is not None else True
    contains_thr = False if thr_number is not None else True
    contains_attenc = False if attenc_number is not None else True

    for item in subparts:
        if loss_type in item:
            contains_joint = not contains_joint
        if 'thr' in item:
            start_index = item.find('thr')
            tmp_thr_number = float(item[start_index+3:])
            if tmp_thr_number == thr_number:
                contains_thr = not contains_thr
        if 'attenc' in item:
            start_index = item.find('attenc')
            tmp_attenc_number = int(item[start_index+6:])
            if tmp_attenc_number == attenc_number:
                contains_attenc = not contains_attenc

    if contains_joint and contains_thr and contains_attenc:
        return True
    return False

def find_epoch_from_log(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()
        last_line = lines[-1]
        start_index = last_line.find('ep')
        epoch_number = int(last_line[start_index+3:].split(',')[0])
    return epoch_number

def find_minloss_from_ckptname(ckptname):
    start_index = ckptname.find('DE')
    min_loss = float(ckptname[start_index+2:].split('-')[0])
    return min_loss

def init_parse_old_version():
    args={}
    args['train_epoches'] = 50
    args['enc_hidden_size'] = 128
    args['dec_hidden_size'] = 256
    args['enc_embed_size']  = 256
    args['enc_gat_size']    = 256
    args['num_gat_head'] = 4
    args['num_modes'] = 32
    args['batch_size'] = 32
    args['train_split'] = 0.9
    args['num_batch']   =  -1           # -1 # -1
    args['eval_num_batch'] = 300        # 300

    args['train_from_scratch'] = True
    args['dec_type'] = 'MGA'            # 'mtpGoal' # 'mtp_goal_veh'
    args['feat_attention'] = True       # Dyn, CCL, Int whether use att to combine them
    args['loss_type'] = 'Joint'         # 'Joint' # 'Marginal'
    args['gnn_conv'] = 'GATConv'        # 'GATv2Conv' # 'GATConv' # 'TransformerConv' # 'GCNConv'
    args['use_attention'] = False       # trajectories whether use att to handle them
    args['use_gsl'] = True              # whether use gsl
    args['dist_threshold'] = 50.0       # threhold for finding the best edge index
    args['threshold'] = 0.35            # threshold for finding the best value for gsl
    args['norm_seg'] = 1                # 1: [Dyn, CCLs, Int], 2: [Dyn, Int], 3: [Dyn], 4: [Dyn, CCLs]
    args['eval'] = False
    if args['loss_type'] == 'Joint':
        from mtp_goal_loss import mtp_goal_loss
        loss_func_for_training = mtp_goal_loss
    elif args['loss_type'] ==  'Marginal':
        from mtp_goal_loss import marginal_mtp_goal_loss
        loss_func_for_training = marginal_mtp_goal_loss
    else:
        raise ValueError('loss_type should be either Joint or Marginal')
    return args, loss_func_for_training

def info_old_version():
    print(f"threshold: {args['threshold']}", file=f, flush=True)
    print(f"gnn_type: {args['gnn_conv']}", file=f, flush=True)
    print(f"loss_type: {args['loss_type']}", file=f, flush=True)
    print(f"enc_attention: {args['use_attention']}", file=f, flush=True)
    print(f"dist_threshold: {dist_threshold}", file=f, flush=True)
    pytorch_total_params = sum(p.numel() for p in train_net.parameters())
    print('number of parameters: {}'.format(pytorch_total_params), file=f, flush=True)
    print('number of parameters: {}'.format(pytorch_total_params))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-epoch', '--train_epoches', type=int, default=50)
    parser.add_argument('--enc_hidden_size', type=int, default=128)
    parser.add_argument('--dec_hidden_size', type=int, default=256)
    parser.add_argument('--enc_embed_size', type=int, default=256)
    parser.add_argument('--enc_gat_size', type=int, default=256)
    parser.add_argument('--num_gat_head', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_split', type=float, default=0.9)
    parser.add_argument('--num_batch', type=int, default=-1)
    parser.add_argument('--eval_num_batch', type=int, default=300)

    parser.add_argument('-tfs', '--train_from_scratch', default=True)
    parser.add_argument('-det', '--dec_type', type=str, default='MGA')
    parser.add_argument('--num_modes', type=int, default=6)
    parser.add_argument('-lt', '--loss_type', type=str, default='Joint', help="feature used for decoding", choices=['Joint', 'Marginal', 'Joint_position'])
    parser.add_argument('-a', '--feat_attention', action='store_true', default=False)
    parser.add_argument('-gnn', '--gnn_conv', type=str, default='GATConv')
    parser.add_argument('-tr', '--use_attention', action='store_true', default=False)
    parser.add_argument('-g', '--use_gsl', action='store_true', default=False)
    parser.add_argument('-dt', '--dist_threshold', type=float, default=50.0)
    parser.add_argument('-t', '--threshold', type=float, default=0.35)
    parser.add_argument('-seg', '--norm_seg', type=int, default=1)
    parser.add_argument('--eval', action='store_true', default=False)
    args = parser.parse_args()
    
    # Convert args to dictionary
    args = vars(args)
    #args['loss_type'] = 'Joint_position'
    #args['feat_attention'] = True
    #args['train_from_scratch'] = True
    
    # loss
    if args['loss_type'] == 'Joint':
        from mtp_goal_loss import mtp_goal_loss
        loss_func_for_training = mtp_goal_loss
    elif args['loss_type'] == 'Marginal':
        from mtp_goal_loss import marginal_mtp_goal_loss
        loss_func_for_training = marginal_mtp_goal_loss
    elif args['loss_type'] == 'Joint_position':
        loss_func_for_training = joint_mag_loss_w_inter_group_position
    else:
        raise ValueError('loss_type should be either Joint or Marginal')
        
    ##################################################################
    myhost = os.uname()[1]
    if myhost == 'AutoManRRCServer':
        data_path = '/disk6/SimAgent_Dataset/pyg_data_Jun23/training'
        args['device'] = 'cuda:1' 
    elif myhost == 'xy-desktop':
        train_data_path = '/home/xy/mzq/code/MAGDP/process_data/validation_0-140'
        eval_data_path  = '/home/xy/mzq/code/MAGDP/process_data/validation_140-145'
        args['device'] = 'cuda:0'
    elif myhost == 'xy-Legion':
        train_data_path = '/home/xy/sim/SimAgent_Dataset/pyg_data_Jun23/validation_0-140'
        eval_data_path  = '/home/xy/sim/SimAgent_Dataset/pyg_data_Jun23/validation_140-145'
        args['device'] = 'cuda:0' 
    else: # NSCC
        data_path = '/home/users/ntu/baichuan/scratch/sim/SimAgent_Dataset/pyg_data_Jun23/training'
        args['device'] = 'cuda:0' 
    # args['use_feat'] = opt.use_feat # N, D, L, DL . N: nothing, D: Dyn only, L: CCLs only, DL: C-A, AA, D+L: Dyn+CCL
    ##################################################################
    
    norm_seg_dict = {
        1: "D_L_I",
        2: "D_I",
        3: "D",
        4: "D_L"
    }

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H%M%S")     # dd/mm/YY H:M:S

    # make the directory for the output and ckpt
    ckpt_folder_path = f"/home/xy/mzq/code/MAGDP/MGA/models/{dt_string[3:5]}.{dt_string[:2]}/"
    log_folder_path = f"/home/xy/mzq/code/MAGDP/MGA//outs/{dt_string[3:5]}.{dt_string[:2]}/"
    os.makedirs(ckpt_folder_path, exist_ok=True)
    os.makedirs(log_folder_path, exist_ok=True)
   
    # 
    only_eval = False
    threshold = args['threshold']
    dist_threshold = args['dist_threshold']

    model_name = f"{args['dec_type']}_M{args['num_modes']}_{args['loss_type']}_{'a_' if args['feat_attention'] else ''}{args['gnn_conv']}_{'tr_' if args['use_attention'] else ''}{f'g{threshold}_' if args['use_gsl'] else ''}{'' if args['dist_threshold'] == 50 else f'dt{dist_threshold}_'}{norm_seg_dict[args['norm_seg']]}"

    if args['train_from_scratch']:
        train_net = MGA_net(args)
        start_ep = 1
        # datetime object containing current date and time
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H%M%S")     # dd/mm/YY H:M:S
        log_name = set_log_name(log_folder_path, dt_string)
        f = open(log_name, "w+")
        min_val_loss = 100.0
    else:
        if only_eval:
            ckpt_path = './MGA/models/07.20/MGA_M6_Joint_position_a_GATConv_D_L_I_minJDE7.51.ckpt'
            ckpt = torch.load(ckpt_path)
            if isinstance(ckpt, nn.Module):
                train_net = ckpt
            elif isinstance(ckpt, dict):
                train_net = MGA_net(args)
                train_net.load_state_dict(ckpt, strict=False)
            f = open('test.txt', "w+")
            start_ep = 1
            min_val_loss = 0
        else:
            # find the ckpt file and log file
            ckpt_path_list = find_files_with_params('models', args['gnn_conv'], args['loss_type'], args['threshold'], int(args['use_attention']))
            log_path_list = find_files_with_params('outs', args['gnn_conv'], args['loss_type'], args['threshold'], int(args['use_attention']))
            assert len(ckpt_path_list) == 1, print(f"gnn_type:{args['gnn_conv']}, loss_type:{args['loss_type']}, threshold:{args['threshold']}, attenc:{args['use_attention']} has {len(ckpt_path_list)} ckpt files")
            assert len(log_path_list) == 1, print(f"gnn_type:{args['gnn_conv']}, loss_type:{args['loss_type']}, threshold:{args['threshold']}, attenc:{args['use_attention']} has {len(ckpt_path_list)} ckpt files")
            ckpt_path = ckpt_path_list[0]
            ckpt_folder_path = os.path.dirname(ckpt_path)
            min_val_loss = find_minloss_from_ckptname(ckpt_path)
            log_path = log_path_list[0]
            start_ep = find_epoch_from_log(log_path) + 1
            train_net = torch.load(ckpt_path)
            f = open(log_path, "a+")

    print(model_name, file=f, flush=True)
    print(model_name)
    pytorch_total_params = sum(p.numel() for p in train_net.parameters())
    print('number of parameters: {}'.format(pytorch_total_params), file=f, flush=True)
    print('number of parameters: {}'.format(pytorch_total_params))

    train_net.to(args['device'])
    optimizer = torch.optim.AdamW(train_net.parameters(), lr=0.0001, weight_decay=0.01) 
    scheduler = MultiStepLR(optimizer, milestones=[20, 22, 24, 26, 28], gamma=0.5)

    #################################
    train_set = SimAgn_Dataset(data_path=train_data_path, dec_type=args['dec_type']) 
    val_set = SimAgn_Dataset(data_path=eval_data_path, dec_type=args['dec_type']) 
    print('train_size: {}, val_size: {}'.format(train_set.__len__(), val_set.__len__()))
    trainDataloader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=12) # num_workers=6, 
    valDataloader   = DataLoader(val_set,   batch_size=args['batch_size'], shuffle=True, num_workers=12) # num_workers=6, 
    #################################

    tic = time.time()
    Val_LOSS = []
    Train_LOSS = []

    print(f"start_ep: {start_ep}, end_ep:{args['train_epoches'] + start_ep}")
    for ep in range(start_ep, args['train_epoches'] + start_ep):
        if ep == 1:
            args_json = json.dumps(args, indent=4)
            with open('args.json', 'w') as f:
                f.write(args_json)
        # logger.info(f"ep {ep}")
        train_time_tic = time.time()
        # logger.info(f"start training")
        train_loss_ep = train_a_model(train_net, trainDataloader, loss_func=loss_func_for_training, num_ep=ep,  num_batch=args['num_batch'])
        # logger.info(f"start evaluation")
        eval_loss_ep  = eval_a_model(train_net,  valDataloader,   loss_func=loss_func_for_training, plot=False, num_batch=args['eval_num_batch'])
        scheduler.step()
        
        if args['loss_type'] == 'Joint':
            if eval_loss_ep[1] < min_val_loss:
                ckpt_name = set_ckpt_name(ckpt_folder_path, model_name, min_val_loss)
                new_ckpt_name = set_ckpt_name(ckpt_folder_path, model_name, eval_loss_ep[1])
                print(ckpt_name)
                if os.path.exists(ckpt_name):
                    os.remove(ckpt_name)
                torch.save(train_net.state_dict(), new_ckpt_name)
                min_val_loss = eval_loss_ep[1]
        elif args['loss_type'] == 'Marginal':
            if eval_loss_ep[2] < min_val_loss:
                ckpt_name = set_ckpt_name(ckpt_folder_path, model_name, min_val_loss)
                new_ckpt_name = set_ckpt_name(ckpt_folder_path, model_name, eval_loss_ep[2])
                if os.path.exists(ckpt_name):
                    os.remove(ckpt_name)
                torch.save(train_net.state_dict(), new_ckpt_name)
                min_val_loss = eval_loss_ep[2]
        elif args['loss_type'] == 'Joint_position':
            if eval_loss_ep[1] < min_val_loss:
                ckpt_name = set_ckpt_name(ckpt_folder_path, model_name, min_val_loss)
                new_ckpt_name = set_ckpt_name(ckpt_folder_path, model_name, eval_loss_ep[1])
                if os.path.exists(ckpt_name):
                    os.remove(ckpt_name)
                torch.save(train_net.state_dict(), new_ckpt_name)
                min_val_loss = eval_loss_ep[1]
        
        ep_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(f'ep {ep}, SmoothL1 loss train {train_loss_ep}, eval {eval_loss_ep[0]}, min-Joint-FDE {eval_loss_ep[1]}, Marginal minFDE {eval_loss_ep[2]}, [ lr= {ep_lr} ]')
        print(f'ep {ep}, SmoothL1 loss train {train_loss_ep}, eval {eval_loss_ep[0]}, min-Joint-FDE {eval_loss_ep[1]}, Marginal minFDE {eval_loss_ep[2]}, [ lr= {ep_lr} ]', file=f, flush=True)
     

    f.close()

        
