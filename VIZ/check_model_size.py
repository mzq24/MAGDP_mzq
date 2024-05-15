import sys
from torch_geometric.loader import DataLoader
import torch
import numpy as np
import sys
import os
from random import sample
# import tensorflow as tf

# from waymo_open_dataset.utils.sim_agents import visualizations
# from waymo_open_dataset.protos import scenario_pb2
sys.path.append('../')
sys.path.append('../utils')
sys.path.append('../MGA')
sys.path.append('../TGE')
sys.path.append('../GDP')
from dataset_may17 import SimAgn_Dataset, flat_Batch, retrieve_mask
from sub_utils import convert_coords_from_sdc_to_world_frame
DATA_PATH     = '/home/users/ntu/baichuan/scratch/sim/SimAgent_Dataset/pyg_data_Jun23/testing-for_sub'
WORK_PATH     = '/home/users/ntu/baichuan/scratch/sim/Jul08'
ROLLOUTS_PATH = '../SUB/sub_testing_MGAm32_pull_GDP6'

ALL_SCENARIO_IDs = [ scn.split('.pt')[0] for scn in os.listdir(ROLLOUTS_PATH) ]

def load_models_for_rollouts(work_path, eval_device='cpu'):
    # 1. 加载 MGA 模型
    sys.path.append('MGA')
    mga_model_ckpt_path = f'{work_path}/MGA/models/MGA-M32-attn-minFDE2.03.ckpt' #  'MGAM32-EP49-Loss3.76-minFDE6.93.ckpt' MGAM4-EP9-Loss5.79-minFDE10.46.ckpt
    mga_net = torch.load(mga_model_ckpt_path, map_location=eval_device)
    mga_net.eval()
    mga_net.args['device']   = eval_device
    mga_net.args['dec_type'] = 'testMGA'

    # 2. 加载 Goal Estimation 模型
    sys.path.append('TGE')
    tge_ckpt_path = f'{work_path}/TGE/models/TGE-M1-attn-2-minFDE7.11.ckpt'
    tge_net = torch.load(tge_ckpt_path, map_location=eval_device)
    tge_net.eval()
    tge_net.args['device']   = eval_device
    tge_net.args['dec_type'] = 'testTGE'

    ## 加载 GDP 模型
    sys.path.append('GDP')
    gdp_ckpt_path = f'{work_path}/GDP/models/arGDPmaxout6GDLI-Loss0.49.ckpt' # 'GDP-EP9-Loss0.37.ckpt' 'arGDPmax-EP6-Loss0.34.ckpt' 'GDP-EP29-Loss0.29.ckpt
    gdp_net = torch.load(gdp_ckpt_path, map_location=eval_device)
    gdp_net.eval()
    gdp_net.args['device']   = eval_device
    gdp_net.args['dec_type'] = 'testarGDPmax' # 'testarGDPmax' , 'testGDP'

    return mga_net, tge_net, gdp_net 

m,t,g=load_models_for_rollouts(WORK_PATH)

model_list = [m, t, g]
model_name_list = ['MAG', 'TGE', 'GDP']

for i, model in enumerate(model_list):
    model_name = model_name_list[i]
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'{model_name} number of parameters: {pytorch_total_params}')

    model_enc = model.encoder
    pytorch_total_params = sum(p.numel() for p in model_enc.parameters())
    print(f'{model_name} encoder, number of parameters: {pytorch_total_params}')


