import sys
# from torch_geometric.data import Data
from dataset_may17 import SimAgn_Dataset, flat_Batch, retrieve_mask
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import time
import torch
import numpy as np
from tqdm import tqdm 

from utils.sub_utils import Rollouts_Sampler, convert_coords_from_sdc_to_world_frame
from utils.vis_utils import scatter_goal

from multiprocessing import Pool

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--start_idx", type=int, default=0,  help="the number of workders for parallel processing")
parser.add_argument("--end_idx",   type=int, default=20, help="the number of shards to  process")
opt = parser.parse_args()   

import os 
myhost = os.uname()[1]
if myhost == 'AutoManRRCServer':
    DATA_PATH = '/disk2/SimAgent_Dataset/pyg_data_Jun11/validation-for_sub'
elif myhost == 'amrrc':
    DATA_PATH = '/home/xy/SimAgent_Dataset/pyg_data_Jun23/validation-for_sub'
    WORK_PATH = '/home/xy/simAg/Jun23'
else: # NSCC
    DATA_PATH = '/home/users/ntu/baichuan/scratch/sim/SimAgent_Dataset/pyg_data_Jun23/validation-for_sub'
    WORK_PATH = '/home/users/ntu/baichuan/scratch/sim/Jun23'

dataset = SimAgn_Dataset(data_path=DATA_PATH, dec_type='test')
dataset.data_names = dataset.data_names[opt.start_idx : opt.end_idx]
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
print(dataset.__len__())
## 加载各种模型
# eval_device = 'cuda:0'
eval_device = 'cpu'

eval_batch_size = 1

# 1. 加载 MGA 模型
sys.path.append('MGA')
goalset_model_ckpt_path = f'{WORK_PATH}/MGA/models/MGAall-M6-minFDE1.93.ckpt' #  'MGAM32-EP49-Loss3.76-minFDE6.93.ckpt' MGAM4-EP9-Loss5.79-minFDE10.46.ckpt
goalset_net = torch.load(goalset_model_ckpt_path, map_location=eval_device)
goalset_net.eval()
goalset_net.args['device']   = eval_device
goalset_net.args['dec_type'] = 'testMGA'

# 2. 加载 Goal Estimation 模型
sys.path.append('TGE')
goal_est_ckpt_path = f'{WORK_PATH}/TGE/models/TGEall-M6-2-minFDE1.13.ckpt'
goal_est_net = torch.load(goal_est_ckpt_path, map_location=eval_device)
goal_est_net.eval()
goal_est_net.args['batch_size'] = eval_batch_size
goal_est_net.args['device']   = eval_device
goal_est_net.args['dec_type'] = 'testTGE'

## 加载 GDP 模型
sys.path.append('GDP')
gdp_ckpt_path = f'{WORK_PATH}/GDP/models/GDPallout4GDLIT-Loss0.09.ckpt' # 'GDP-EP9-Loss0.37.ckpt' 'arGDPmax-EP6-Loss0.34.ckpt' 'GDP-EP29-Loss0.29.ckpt
gdp_net = torch.load(gdp_ckpt_path, map_location=eval_device)
gdp_net.eval()
gdp_net.args['device']   = eval_device
gdp_net.args['dec_type'] = 'testGDP' # 'testarGDPmax' , 'testGDP'

# gdp_net.args['dec_type'] = 'test'
sim_sampler = Rollouts_Sampler(goalset_net, goal_est_net, gdp_net)

data_list = []
SAVE_TO_PATH = './SUB/sub_val_MGA6_GDP'

for d_idx, d in tqdm(enumerate(loader)):
    tic = time.time()
    # print(d)
    flat_d = flat_Batch(d).to(eval_device)
    # print(flat_d)
    
    scenario_rollouts = sim_sampler.sample_rollouts(flat_d, smoothen=True)

    world_frame_rollouts = []
    for s_r in scenario_rollouts:
        a_traj_world = convert_coords_from_sdc_to_world_frame(s_r.reshape(-1,4), flat_d.sdc_cur_state[0][:2], flat_d.sdc_cur_state[0][2], flat_d.sdc_cur_state[0][3])
        world_frame_rollouts.append(np.expand_dims(a_traj_world.reshape(-1,80,4), axis=0))
    # print(len(scenario_rollouts), [scenario_rollouts[i].shape for i in range(len(scenario_rollouts))])

    scn_id = flat_d.object_id[0][0].split('-')[0]
    obj_indexes = [ int(obj.split('-')[-1]) for obj in flat_d.object_id[0] ]
    valid_obj_id = flat_d.valid_object_id[0]
    # print(scn_id)
    # print(obj_id)
    scenario_rollouts = {'scenario_id': scn_id, 
                        'rollouts':np.concatenate(world_frame_rollouts, axis=0), 
                        'object_id': valid_obj_id, 
                        'track_indexes':obj_indexes}
    
    # print(scenario_rollouts['rollouts'].shape)
    # print(scenario_rollouts['object_id'])
    # print(scenario_rollouts['track_indexes'])

    # break
    pyg_name = f'{SAVE_TO_PATH}/{scn_id}.pt'
    torch.save(scenario_rollouts, pyg_name)
    # if d_idx>1000:
    #     break 
