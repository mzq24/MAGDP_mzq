import sys
sys.path.append('../')
sys.path.append('../utils')
sys.path.append('../MGA')
sys.path.append('../TGE')
sys.path.append('../GDP')

# from torch_geometric.data import Data
from dataset_may17 import SimAgn_Dataset, flat_Batch, retrieve_mask
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import time
import torch
import numpy as np
from tqdm import tqdm 

from rollout_sampler import Rollouts_Sampler, convert_coords_from_sdc_to_world_frame
# from utils.vis_utils import scatter_goal
from waymo_open_dataset.utils.sim_agents import visualizations

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
    DATA_PATH = '/home/users/ntu/baichuan/scratch/sim/SimAgent_Dataset/pyg_data_Jun23/testing-for_sub'
    WORK_PATH = '/home/users/ntu/baichuan/scratch/sim/Jul08'

dataset = SimAgn_Dataset(data_path=DATA_PATH, dec_type='test')
dataset.data_names = dataset.data_names[opt.start_idx : opt.end_idx]
loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
print(dataset.__len__())
## 加载各种模型
# eval_device = 'cuda:0'
eval_device = 'cpu'

eval_batch_size = 1

# 1. 加载 MGA 模型
sys.path.append('MGA')
goalset_model_ckpt_path = f'{WORK_PATH}/MGA/models/MGA-M32-attn-minFDE2.03.ckpt' #  'MGAM32-EP49-Loss3.76-minFDE6.93.ckpt' MGAM4-EP9-Loss5.79-minFDE10.46.ckpt
goalset_net = torch.load(goalset_model_ckpt_path, map_location=eval_device)
goalset_net.eval()
goalset_net.args['device']   = eval_device
goalset_net.args['dec_type'] = 'testMGA'

## 加载 GDP 模型
sys.path.append('GDP')
gdp_ckpt_path = f'{WORK_PATH}/GDP/models/arGDPmaxout6GDLI-Loss0.49.ckpt' # 'GDP-EP9-Loss0.37.ckpt' 'arGDPmax-EP6-Loss0.34.ckpt' 'GDP-EP29-Loss0.29.ckpt
gdp_net = torch.load(gdp_ckpt_path, map_location=eval_device)
gdp_net.eval()
gdp_net.args['device']   = eval_device
gdp_net.args['dec_type'] = 'testarGDPmax' # 'testarGDPmax' , 'testGDP'

# gdp_net.args['dec_type'] = 'test'
sim_sampler = Rollouts_Sampler(goalset_net, goal_est_net, gdp_net, pull_to_TCL=True, Smoothen_GDP=True, use_TGE=False)

data_list = []
if sim_sampler.use_TGE:
    if sim_sampler.pull_to_TCL:
        SAVE_TO_PATH = './SUB/sub_testing_MGAm32_TGEm1_pull_GDP6'
    else:
        SAVE_TO_PATH = './SUB/sub_testing_MGAm32_TGEm1_GDP6'
elif sim_sampler.pull_to_TCL:
    SAVE_TO_PATH = './SUB/sub_testing_MGAm32_pull_GDP6'
else:
    SAVE_TO_PATH = './SUB/sub_testing_MGAm32_GDP6'

import torch
from matplotlib.collections import LineCollection


def scatter_pos(ax, pos):
    ax.scatter(pos[:,0], pos[:,1], s=50, marker='*')

def plot_track_trajectory(ax, track, color='r') -> None:
    valids = np.array([state.valid for state in track.states])[:11]
    if np.any(valids):
        x = np.array([state.center_x for state in track.states])[:11]
        y = np.array([state.center_y for state in track.states])[:11]
        ax.plot(x[valids], y[valids], linewidth=3, alpha=0.5, color=color)

        ax.scatter(x[valids][-1], y[valids][-1])
        # ax.scatter(x[valids], y[valids], cmap='winter' )

def plot_traj_planned(ax, traj, step=1) -> None:
    # Set the values used for colormapping
    # y 的值归一化到[0, 1]
    # 因为 y 大到一定程度超过临界数值后颜色就会饱和不变(不使用循环colormap)。
    traj = traj[step-1::step, :2]
    x, y = traj[:,0], traj[:,1]
    c = np.arange(0,80//step)

    ax.scatter(x, y, c=c, s=10, alpha=0.6)

for d_idx, d in tqdm(enumerate(loader)):
    tic = time.time()
    # print(d)
    flat_d = flat_Batch(d).to(eval_device)
    print(flat_d)
    # print(flat_d.flat_node_name)
    print(flat_d.flat_node_type)

    scenario_rollouts, scenario_rollouts_change_TCL = sim_sampler.sample_rollouts(flat_d)
    print(scenario_rollouts[0].shape, scenario_rollouts_change_TCL[0].shape)

    world_frame_rollouts = []
    for s_r in scenario_rollouts:
        s_r = s_r[:,:,[0,1,2,5]]
        a_traj_world = convert_coords_from_sdc_to_world_frame(s_r.reshape(-1,4), flat_d.sdc_cur_state[0][:2], flat_d.sdc_cur_state[0][2], flat_d.sdc_cur_state[0][3])
        world_frame_rollouts.append(np.expand_dims(a_traj_world.reshape(-1,80,4), axis=0))

    world_frame_rollouts_change_TCL = []
    for s_r in scenario_rollouts_change_TCL:
        s_r = s_r[:,:,[0,1,2,5]]
        a_traj_world = convert_coords_from_sdc_to_world_frame(s_r.reshape(-1,4), flat_d.sdc_cur_state[0][:2], flat_d.sdc_cur_state[0][2], flat_d.sdc_cur_state[0][3])
        world_frame_rollouts_change_TCL.append(np.expand_dims(a_traj_world.reshape(-1,80,4), axis=0))

    scn_id = flat_d.object_id[0][0].split('-')[0]
    obj_indexes = [ int(obj.split('-')[-1]) for obj in flat_d.object_id[0] ]
    valid_obj_id = flat_d.valid_object_id[0]
    # print(scn_id)
    # print(obj_id)
    scenario_rollouts = {'scenario_id': scn_id, 
                        'rollouts': np.concatenate(world_frame_rollouts, axis=0), 
                        'rollouts_change_TCL': np.concatenate(world_frame_rollouts_change_TCL, axis=0), 
                        'object_id': valid_obj_id, 
                        'track_indexes':obj_indexes}
    
    # print(scenario_rollouts['rollouts'].shape)
    # print(scenario_rollouts['object_id'])
    # print(scenario_rollouts['track_indexes'])

    # NUM_SCENE = 10
    NUM_AX = 2



    # Visualize scenario.
    fig, axs = plt.subplots(1, NUM_AX, figsize=(10*NUM_AX, 10))
    ego_tar_pos = world_frame_rollouts[0][0,1,-1:,:2]
    ego_tar_pos_change_TCL = world_frame_rollouts_change_TCL[0][0,1,-1:,:2]
    scatter_pos(axs[0], ego_tar_pos)
    scatter_pos(axs[1], ego_tar_pos_change_TCL)

    # scatter_pos(axs[0], ego_pos)
    # for i in range(NUM_AX):
    #     scatter_pos(axs[0], ego_tar_pos)
        # scatter_pos(axs[1], ego_tar_pos)
    #     axs[i].set_xticks([])
    #     axs[i].set_yticks([])
    ego_tar_mask =  retrieve_mask(type_list=flat_d.flat_node_type, wanted_type_set=('egoAg', 'tarAg'))
    for i in range(world_frame_rollouts[0].shape[1]):
        traj_1 = world_frame_rollouts[0][0,i,:,:2]
        traj_2 = world_frame_rollouts_change_TCL[0][0,i,:,:2]
        # print(traj_1.shape, traj_2.shape)
        plot_traj_planned(axs[0], traj_1)
        plot_traj_planned(axs[1], traj_2)

        if i == sum(ego_tar_mask):
            break
        # visualizations.add_map(axs[i], scenario)
        # for track in scenario.tracks:
        #     plot_track_trajectory(axs[i], track)
    ## 找到 tracks to predict 的 id
    # tracks_to_predict_indeces = [cur_pred.track_index for cur_pred in scenario.tracks_to_predict]
    # objects_to_predict_ids    = [scenario.tracks[idx].id for idx in tracks_to_predict_indeces]
    # print(f'objects_to_predict_ids', objects_to_predict_ids)


    # scn_pt = torch.load(f'./{rollout_save_path}/{scenario.scenario_id}.pt')
    # simulated_states = scn_pt['rollouts']
    # # print(simulated_states.shape)
    # object_id = scn_pt['object_id']

    # for r_i in range(NUM_AX):
    #     rollout_1 = simulated_states[r_i,:,:,:2]
    #     for j, plan in enumerate(rollout_1):
    #         if scn_pt['object_id'][j] in objects_to_predict_ids:
    #             plot_traj_planned(axs[r_i], plan)

    plt.savefig(f'./change_TCL.png', bbox_inches='tight')
    plt.close()





    break
    pyg_name = f'{SAVE_TO_PATH}/{scn_id}.pt'
    torch.save(scenario_rollouts, pyg_name)
    # if d_idx>1000:
    #     break 

