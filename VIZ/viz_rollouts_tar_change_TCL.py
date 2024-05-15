
from torch_geometric.loader import DataLoader
import torch
import numpy as np
import sys
import os
from random import sample
import tensorflow as tf

from waymo_open_dataset.utils.sim_agents import visualizations
from waymo_open_dataset.protos import scenario_pb2

sys.path.append('../')
sys.path.append('../utils')
sys.path.append('../MGA')
sys.path.append('../TGE')
sys.path.append('../GDP')
from dataset_may17 import SimAgn_Dataset, flat_Batch, retrieve_mask
from sub_utils import convert_coords_from_sdc_to_world_frame

DATA_PATH     = '/home/xy/sim/SimAgent_Dataset/pyg_data_Sep/validation'
WORK_PATH     = '/home/xy/sim/Jul08_nscc'

ROLLOUTS_PATH = '../SUB/sub_testing_MGAm32_pull_GDP6'

# ALL_SCENARIO_IDs = [ scn.split('.pt')[0] for scn in os.listdir(ROLLOUTS_PATH) ]
# scenario_id = '53e95e17b15be1f3'

# print(scenario_rollouts.keys())
# print(scenario_rollouts['object_id'])
# print(scenario_rollouts['track_indexes'])
# print(scenario_rollouts['rollouts'].shape)

################################################################################
## 准备好 WOMD 的数据集
DATASET_FOLDER = '/home/xy/sim/Waymo_Dataset/validation'
TEST_FILES = os.path.join(DATASET_FOLDER, 'validation.tfrecord*')
filenames = tf.io.matching_files(TEST_FILES)
dataset = tf.data.TFRecordDataset(filenames)#.shuffle(64)
dataset_iterator = dataset.as_numpy_iterator()
################################################################################

def find_WOMD_scenario(scenario_id, dataset_iterator=dataset_iterator, start_from=0):
    for i in range(45000):
        bytes_example = next(dataset_iterator)
        if i < start_from:
            continue
        scenario = scenario_pb2.Scenario.FromString(bytes_example)
        print(i, 'scn_id: ', scenario.scenario_id)
        if scenario.scenario_id == scenario_id:
            return scenario
            break
    # pass

# print(scenario_pyg.node_name[:3])
# print(scenario_pyg)
class change_TCL_plotter():
    def __init__(self, pyg_data, scenario_rollouts) -> None:
        self.pyg_data = pyg_data
        self.scenario_rollouts = scenario_rollouts
        self.scenario_id = self.scenario_rollouts['scenario_id']
        self.ego_mask = retrieve_mask(type_list=self.pyg_data.node_type, wanted_type_set=('sdcAg'))
        self.tar_mask = retrieve_mask(type_list=self.pyg_data.node_type, wanted_type_set=('tarAg'))

        self.get_ego_id()
        self.get_tar_ids()
        self.get_agn_id_to_idx_dict()
        self.get_agn_idx_to_id_dict()

    def get_agn_id_to_idx_dict(self):
        self.agn_id_to_idx_dict = dict(map(lambda i,j : (i,j) , self.scenario_rollouts['object_id'], self.scenario_rollouts['track_indexes']))

    def get_agn_idx_to_id_dict(self):
        self.agn_idx_to_id_dict = dict(map(lambda i,j : (i,j) , self.scenario_rollouts['track_indexes'], self.scenario_rollouts['object_id']))

    def get_ego_id(self):
        scene_id, ego_track_idx = np.array(self.pyg_data.node_name)[self.ego_mask][0].split('-')
        ego_id = self.scenario_rollouts['object_id'][self.scenario_rollouts['track_indexes'].index(int(ego_track_idx))]
        self.ego_id = ego_id
    
    def get_tar_ids(self):
        tar_node_names_in_pyg = np.array(self.pyg_data.node_name)[self.tar_mask]
        tar_track_idxes       = [tar_name.split('-')[-1] for tar_name in tar_node_names_in_pyg]
        self.tar_ids          = [self.scenario_rollouts['object_id'][self.scenario_rollouts['track_indexes'].index(int(tar_track_idx))] for tar_track_idx in tar_track_idxes]

    def get_agn_ccl_mask(self, agn_id):
        agn_idx = self.agn_id_to_idx_dict[agn_id]
        agn_ccl_mask = [True if node_name.startswith(f'{self.scenario_id}-{agn_idx}CCL') else False for node_name in self.pyg_data.node_name]
        return agn_ccl_mask

    def plot_centerline(self, ax, centerline, color='k', line=':', marker=None, alpha=0.2):
        ax.plot(centerline[:,0], centerline[:,1], linestyle=line, marker=marker, color=color, alpha=alpha)

    def plot_agent_CCLs(self, ax, agn_id):
        agn_ccl_mask = self.get_agn_ccl_mask(agn_id)
        # print(agn_ccl_mask)
        agn_CCLs = self.pyg_data.node_feature[agn_ccl_mask]
        # print(agn_CCLs.shape)
        for ccl in agn_CCLs:
            ccl = convert_coords_from_sdc_to_world_frame(ccl, self.pyg_data.sdc_cur_state[:2], self.pyg_data.sdc_cur_state[2], self.pyg_data.sdc_cur_state[3], state='xy')
            # print(ccl.shape)
            self.plot_centerline(ax, ccl, color='b', alpha=0.5)

    def plot_planned_traj(self, ax, agn_id, rollout_idx=0, step=1):
        agent_index_in_rollouts = self.scenario_rollouts['object_id'].index(agn_id)
        agent_rollout = self.scenario_rollouts['rollouts'][rollout_idx, agent_index_in_rollouts]
        traj = agent_rollout[step-1::step, :2]
        x, y = traj[:,0], traj[:,1]
        c = np.arange(0,80//step)

        ax.scatter(x, y, c=c, s=10, alpha=0.6)
        # ax.text(x[-1], y[-1], agn_id)
    
    def plot_TARs_planned_trajs(self, ax, rollout_idx=0, step=1):
        for tar_id in self.tar_ids:
            self.plot_planned_traj(ax, tar_id, rollout_idx, step)
    
    def plot_SDC_planned_traj(self, ax, rollout_idx=0, step=1):
        self.plot_planned_traj(ax, self.ego_id, rollout_idx, step)
        ax.set_aspect('equal')

# candiate_scenario_ids = ['b7b7be99a9b93d72', '1bc0134805072795', '9828956de59884b4']


def load_models_for_rollouts(work_path, eval_device='cpu'):
    # 1. 加载 MGA 模型
    sys.path.append('MGA')
    mga_model_ckpt_path = f'{work_path}/MGA/models/MGA-M32-attn-minFDE3.54.ckpt' #  'MGAM32-EP49-Loss3.76-minFDE6.93.ckpt' MGAM4-EP9-Loss5.79-minFDE10.46.ckpt
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
    gdp_ckpt_path = f'{work_path}/GDP/models/GDPout6GDLI-attn-Loss1.44.ckpt' # 'GDP-EP9-Loss0.37.ckpt' 'arGDPmax-EP6-Loss0.34.ckpt' 'GDP-EP29-Loss0.29.ckpt
    gdp_net = torch.load(gdp_ckpt_path, map_location=eval_device)
    gdp_net.eval()
    gdp_net.args['device']   = eval_device
    gdp_net.args['dec_type'] = 'testarGDPmax' # 'testarGDPmax' , 'testGDP'

    return mga_net, tge_net, gdp_net 

mga_m, tge_m, gdp_m = load_models_for_rollouts(WORK_PATH)
from rollout_sampler import Rollouts_Sampler
sim_sampler = Rollouts_Sampler(mga_m, tge_m, gdp_m, pull_to_TCL=True, Smoothen_GDP=True, use_TGE=False)

from torch_geometric.loader import DataLoader
from dataset_may17 import SimAgn_Dataset
scenario_id = 'c5df30e1f8c6a798' # '53e95e17b15be1f3' # '27ac0e6fb4802b33' '99584c5fcd3d8186' 'fc7a5da13670752b'
womd_scenario = find_WOMD_scenario(scenario_id)

scenario_pyg_name = f'{DATA_PATH}/2-scn-{scenario_id}.pyg'
scenario_rollouts_name  = f'{ROLLOUTS_PATH}/{scenario_id}.pt'
scenario_pyg = torch.load(scenario_pyg_name)
scenario_rollouts = torch.load(scenario_rollouts_name)

plotter = change_TCL_plotter(scenario_pyg, scenario_rollouts)

# print(f'scenario id {plotter.scenario_id}')
# print(f'ego id {plotter.ego_id}')
# print(f'tar ids {plotter.tar_ids}')


import matplotlib.pyplot as plt
NUM_AX = 1 
Rollout_Index = 0

dataset = SimAgn_Dataset(data_path=DATA_PATH, dec_type='test')
dataset.data_names = [f'/2-scn-{scenario_id}.pyg']
loader = DataLoader(dataset, batch_size=1, shuffle=True)

for d in loader:
    flat_d = flat_Batch(d).to('cpu')
    # print(flat_d.flat_node_type)
    scenario_rollouts, _, flat_with_init_TCL = sim_sampler.sample_rollouts(flat_d)
    # print(flat_with_init_TCL.flat_node_type)
    # print(scenario_rollouts[0].shape)
    world_frame_rollouts = []
    for s_r in scenario_rollouts:
        s_r = s_r[:,:,[0,1,2,5]]
        a_traj_world = convert_coords_from_sdc_to_world_frame(s_r.reshape(-1,4), flat_d.sdc_cur_state[0][:2], flat_d.sdc_cur_state[0][2], flat_d.sdc_cur_state[0][3])
        world_frame_rollouts.append(np.expand_dims(a_traj_world.reshape(-1,80,4), axis=0))

    plotter.scenario_rollouts['rollouts'] = np.concatenate(world_frame_rollouts, axis=0)

    fig, axs = plt.subplots(1, NUM_AX, figsize=(10*NUM_AX, 10))

    ## 画上地图
    visualizations.add_map(axs, womd_scenario)

    plotter.plot_SDC_planned_traj(axs, rollout_idx=Rollout_Index)
    plotter.plot_TARs_planned_trajs(axs, rollout_idx=Rollout_Index)
    plotter.plot_agent_CCLs(axs, plotter.ego_id)
    # plotter.plot_agent_CCLs(axs, plotter.tar_ids[0])

    plt.title(f'Scenario: {plotter.scenario_id} -- Ego: {plotter.ego_id}')
    plt.savefig(f'./tar_change_TCL_{plotter.scenario_id}-{plotter.ego_id}.png', bbox_inches='tight')
    plt.close()
    

    ## 改变 Tar 的 TCL
    print(f'tars: {plotter.tar_ids}')
    tar_id = plotter.tar_ids[2]
    tar_ccl_mask = plotter.get_agn_ccl_mask(tar_id)
    # ego_ccl_type = flat_with_init_TCL.flat_node_type[ego_ccl_mask]
    # print(flat_d.flat_node_type[ego_ccl_mask])
    print(flat_with_init_TCL.flat_node_type[tar_ccl_mask])
    # print(ego_ccl_type)
    flat_with_init_TCL.flat_node_type[tar_ccl_mask] = ['tarCCL', 'tarCCL', 'tarTCL', 'sdcCCL_FAKE'] 
    # flat_with_init_TCL.flat_node_type[ego_ccl_mask] = ['sdcCCL', 'tarTCL', 'sdcCCL_FAKE']
    ## 改变 Goal 
    # print(flat_with_init_TCL.sampled_goalset.shape)
    # print(flat_with_init_TCL.flat_node_type[ego_ccl_mask])
    # print(flat_with_init_TCL.flat_node_type)

    tar_tcl = flat_with_init_TCL.node_feature[tar_ccl_mask][1]
    

    scenario_rollouts_ego_change_TCL, _, _ = sim_sampler.sample_rollouts(flat_with_init_TCL, init_TCL=False, manul_goal=None)
    world_frame_rollouts = []
    for s_r in scenario_rollouts_ego_change_TCL:
        s_r = s_r[:,:,[0,1,2,5]]
        a_traj_world = convert_coords_from_sdc_to_world_frame(s_r.reshape(-1,4), flat_d.sdc_cur_state[0][:2], flat_d.sdc_cur_state[0][2], flat_d.sdc_cur_state[0][3])
        world_frame_rollouts.append(np.expand_dims(a_traj_world.reshape(-1,80,4), axis=0))
    
    plotter.scenario_rollouts['rollouts'] = np.concatenate(world_frame_rollouts, axis=0)
    # print(plotter.scenario_rollouts['rollouts'].shape)

    # NUM_AX = 1 
    fig, axs = plt.subplots(1, NUM_AX, figsize=(10*NUM_AX, 10))
    ## 画上地图
    visualizations.add_map(axs, womd_scenario)
    plotter.plot_SDC_planned_traj(axs, rollout_idx=Rollout_Index)
    plotter.plot_TARs_planned_trajs(axs, rollout_idx=Rollout_Index)
    plotter.plot_agent_CCLs(axs, tar_id)
    # plotter.plot_agent_CCLs(axs, plotter.tar_ids[0])


    plt.title(f'Scenario: {plotter.scenario_id} -- tar: {tar_id}')
    plt.savefig(f'./tar_change_TCL_{plotter.scenario_id}-{tar_id}_changeTCL.png', bbox_inches='tight')
    plt.close()






