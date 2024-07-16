import sys

sys.path.append('../')
sys.path.append('../utils')
sys.path.append('../MGA')
sys.path.append('../TGE')
sys.path.append('../GDP')
sys.path.append('/home/xy/mzq/code/MAGDP')
sys.path.append('/home/xy/mzq/code/MAGDP/MGA')

from torch_geometric.loader import DataLoader
import torch
import numpy as np
import os
from random import sample
import tensorflow as tf
import argparse
from waymo_open_dataset.utils.sim_agents import visualizations
from waymo_open_dataset.protos import scenario_pb2
from dataset_may17 import SimAgn_Dataset, flat_Batch, retrieve_mask
from utils.sub_utils import convert_coords_from_sdc_to_world_frame
from rollout_sampler import Rollouts_Sampler
from torch_geometric.loader import DataLoader
from dataset_may17 import SimAgn_Dataset
from MGA.MGA_model_mzq import MGA_net
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
        # agn_ccl_mask = [True if node_name.startswith(f'{self.scenario_id}-{agn_idx}CCL') else False for node_name in self.pyg_data.node_name]
        agn_ccl_mask = [True if node_name.startswith((f'{self.scenario_id}-{agn_idx}CCL', f'{self.scenario_id}-{agn_idx}TCL', f'{self.scenario_id}-{agn_idx}TCL_FAKE')) else False for node_name in self.pyg_data.node_name]
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
            print(ccl.shape)
            self.plot_centerline(ax, ccl, color='b', alpha=0.5)

    def plot_planned_traj(self, ax, agn_id, rollout_idx=0, step=1):
        agent_index_in_rollouts = self.scenario_rollouts['object_id'].index(agn_id)
        agent_rollout = self.scenario_rollouts['rollouts'][rollout_idx, agent_index_in_rollouts]
        traj = agent_rollout[step-1::step, :2]
        x, y = traj[:,0], traj[:,1]
        c = np.arange(0,80//step)

        ax.scatter(x, y, c=c, s=10, alpha=0.6)
        ax.text(x[-1], y[-1], agn_id)
    
    def plot_TARs_planned_trajs(self, ax, rollout_idx=0, step=1):
        for tar_id in self.tar_ids:
            self.plot_planned_traj(ax, tar_id, rollout_idx, step)
    
    def plot_SDC_planned_traj(self, ax, rollout_idx=0, step=1):
        self.plot_planned_traj(ax, self.ego_id, rollout_idx, step)
        ax.set_aspect('equal')

def load_models_for_rollouts(work_path, model_path, args, eval_device='cpu'):
    # 1. 加载 MGA 模型
    sys.path.append('MGA')
    # mga_model_ckpt_path = f'{work_path}/MGA/models/07.08/MGA_M32_Joint_a_GATConv_D_L_I_minJDE6.25.ckpt' #  'MGAM32-EP49-Loss3.76-minFDE6.93.ckpt' MGAM4-EP9-Loss5.79-minFDE10.46.ckpt
    mga_model_ckpt_path = os.path.join(work_path, model_path)
    ckpt = torch.load(mga_model_ckpt_path, map_location='cpu')
    mga_net = MGA_net(vars(args))
    mga_net.load_state_dict(ckpt, strict=False)
    mga_net.eval()
    # mga_net = torch.load(mga_model_ckpt_path, map_location=eval_device)
    # mga_net.eval()
    mga_net.args['device']   = eval_device
    mga_net.args['dec_type'] = 'testMGA'

    # 2. 加载 Goal Estimation 模型

    ## 加载 GDP 模型
    sys.path.append('GDP')
    gdp_ckpt_path = f'{work_path}/GDP/models/arGDPmaxout6GDLI-Loss0.49.ckpt' # 'GDP-EP9-Loss0.37.ckpt' 'arGDPmax-EP6-Loss0.34.ckpt' 'GDP-EP29-Loss0.29.ckpt
    gdp_net = torch.load(gdp_ckpt_path, map_location=eval_device)
    gdp_net.eval()
    gdp_net.args['device']   = eval_device
    gdp_net.args['dec_type'] = 'testarGDPmax' # 'testarGDPmax' , 'testGDP'

    return mga_net, gdp_net 

def get_model_name(model_name, args):
    # model_path = 'MGA_M6_Joint_a_GATConv_D_L_I_minJDE7.43.ckpt'
    model_name = os.path.splitext(model_name)[0]
    file_name = model_name.split('/')[-1]
    match = re.search('M(\d+)', model_name)
    if match:
        args.num_modes = int(match.group(1))

    match = re.search('DE(\d+\.\d+)', model_name)
    if match:
        min_DE = match.group(1)
    
    args.loss_type = 'Joint' if 'Joint' in model_name else 'Marginal'
    
    args.feat_attention = True if 'a' in model_name else False

    # Define the norm_seg_dict
    norm_seg_dict = {
    1: "D_L_I",
    2: "D_I",
    4: "D_L",
    3: "D"
    }

    for key in norm_seg_dict.keys():
        value = norm_seg_dict[key]
        if value in model_name:
            args.norm_seg = key
            break
    return args, min_DE, file_name

def find_WOMD_scenario(scenario_id, dataset_iterator, start_from=0):
    for i in range(45000):
        bytes_example = next(dataset_iterator)
        if i < start_from:
            continue
        scenario = scenario_pb2.Scenario.FromString(bytes_example)
        # print(i, 'scn_id: ', scenario.scenario_id)
        if scenario.scenario_id == scenario_id:
            print(i, 'scn_id: ', scenario.scenario_id)
            return scenario

def plot_scenario(plotter, scenario_rollouts, flat_d, womd_scenario, Rollout_Index, NUM_AX=1):
    pass

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Evaluate MGA collision')

    # Add the arguments
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

    parser.add_argument('--num_modes', type=int, default=6)
    parser.add_argument('-lt', '--loss_type', type=str, default='Joint', help="feature used for decoding", choices=['Joint', 'Marginal'])
    parser.add_argument('-a', '--feat_attention', action='store_true', default=True)
    parser.add_argument('-gnn', '--gnn_conv', type=str, default='GATConv')
    parser.add_argument('-tr', '--use_attention', action='store_true', default=False)
    parser.add_argument('-g', '--use_gsl', action='store_true', default=False)
    parser.add_argument('-dt', '--dist_threshold', type=float, default=50.0)
    parser.add_argument('-t', '--threshold', type=float, default=0.35)
    parser.add_argument('-seg', '--norm_seg', type=int, default=1)
    parser.add_argument('--eval', action='store_true', default=False)

    parser.add_argument('-p', '--model_path', type=str, default='./MGA/models/07.06/MGA_M6_Joint_a_GATConv_D_L_I_minJDE7.43.ckpt', help='The path to the model')
    parser.add_argument('-d', '--eval_device', type=str, default='0', help='The device to use for evaluation')
    parser.add_argument('-det', '--dec_type', type=str, default='MGA', help='The decoder type')
    parser.add_argument('-vb', '--num_val_batch', type=int, default=500, help='The number of validation batches')

    # Parse the arguments
    args = parser.parse_args()

    DATA_PATH     = '/home/xy/mzq/code/MAGDP/process_data/testing'
    WORK_PATH     = '/home/xy/mzq/code/MAGDP'
    ROLLOUTS_PATH = '/media/xy/Waymo_Submissins/testGD_Plan_8sec_sub_testing_MA_Goal_mM64_RotNopull2TCL_NoSmooth_MAGminFde1.45_GDPFde0.23_0-45000'

    ALL_SCENARIO_IDs = [ scn.split('.pt')[0] for scn in os.listdir(ROLLOUTS_PATH) ]
    # scenario_id = '53e95e17b15be1f3'
    args, min_DE, model_name = get_model_name(args.model_path, args)

    mga_m, gdp_m = load_models_for_rollouts(WORK_PATH, args.model_path, args)

    sim_sampler = Rollouts_Sampler(mga_m, gdp_m, pull_to_TCL=True, Smoothen_GDP=True, use_TGE=False)

    scenario_id   = '1abdbcf3cea023fa' # '1bb32081b806adc'
    ################################################################################
    ## 准备好 WOMD 的数据集
    DATASET_FOLDER = '/home/xy/mzq/code/MAGDP/data/testing/'
    TEST_FILES = os.path.join(DATASET_FOLDER, 'testing.tfrecord*')
    filenames = tf.io.matching_files(TEST_FILES)
    womd_dataset = tf.data.TFRecordDataset(filenames)#.shuffle(64)
    dataset_iterator = womd_dataset.as_numpy_iterator()
    womd_scenario = find_WOMD_scenario(scenario_id, dataset_iterator, start_from=0)
    ################################################################################
    
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
        scenario_rollouts, _, flat_with_init_TCL = sim_sampler.sample_rollouts(flat_d, init_TCL=False)
        # print(flat_with_init_TCL.flat_node_type)
        # print(scenario_rollouts[0].shape)
        world_frame_rollouts = []
        for s_r in scenario_rollouts:
            s_r = s_r[:,:,[0,1,2,5]]
            a_traj_world = convert_coords_from_sdc_to_world_frame(s_r.reshape(-1,4), flat_d.sdc_cur_state[0][:2], flat_d.sdc_cur_state[0][2], flat_d.sdc_cur_state[0][3])
            world_frame_rollouts.append(np.expand_dims(a_traj_world.reshape(-1,80,4), axis=0))

        plotter.scenario_rollouts['rollouts'] = np.concatenate(world_frame_rollouts, axis=0)

        fig, axs = plt.subplots(1, NUM_AX, figsize=(10*NUM_AX, 10))
        plt.title(f'Scenario: {plotter.scenario_id} -- Ego: {plotter.ego_id}')

        ## 画上地图
        visualizations.add_map(axs, womd_scenario)
        plotter.plot_SDC_planned_traj(axs, rollout_idx=Rollout_Index)
        plotter.plot_TARs_planned_trajs(axs, rollout_idx=Rollout_Index)
        plotter.plot_agent_CCLs(axs, plotter.ego_id)
        # plotter.plot_agent_CCLs(axs, plotter.tar_ids[0])
        #axs.set_xlim(200, 300)
        #axs.set_ylim(-1380, -1280)

        
        plt.savefig(f'./VIZ/ego_change_TCL_{plotter.scenario_id}-{plotter.ego_id}.png', bbox_inches='tight')
        plt.close()
        
        
        ## 改变 SDC 的 TCL
        ego_ccl_mask = plotter.get_agn_ccl_mask(plotter.ego_id)
        # ego_ccl_type = flat_with_init_TCL.flat_node_type[ego_ccl_mask]
        # print(flat_d.flat_node_type[ego_ccl_mask])
        print(flat_with_init_TCL.flat_node_type[ego_ccl_mask])
        # print(ego_ccl_type)
        # flat_with_init_TCL.flat_node_type[ego_ccl_mask] = ['sdcCCL', 'tarTCL', 'sdcCCL', 'sdcCCL_FAKE'] # ['sdcCCL', 'tarTCL', 'sdcCCL_FAKE']
        num_ccl = sum(ego_ccl_mask)
        tmp_arr = np.full(num_ccl, 'sdcCCL', dtype='<U11')
        fake_ind = np.where(flat_with_init_TCL.flat_node_type[ego_ccl_mask] == 'sdcTCL_FAKE')[0]
        new_fake_ind = 6
        tmp_arr[new_fake_ind] = 'sdcTCL_FAKE'
        flat_with_init_TCL.flat_node_type[ego_ccl_mask] = tmp_arr
        ## 改变 Goal 
        # print(flat_with_init_TCL.sampled_goalset.shape)
        # print(flat_with_init_TCL.flat_node_type[ego_ccl_mask])
        # print(flat_with_init_TCL.flat_node_type)

        ego_tcl = flat_with_init_TCL.node_feature[ego_ccl_mask][1]
        

        scenario_rollouts_ego_change_TCL, _, _ = sim_sampler.sample_rollouts(flat_with_init_TCL, init_TCL=False) #, manul_goal=ego_tcl[7,:2])
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
        # plotter.plot_TARs_planned_trajs(axs, rollout_idx=Rollout_Index)
        plotter.plot_agent_CCLs(axs, plotter.ego_id)
        #axs.set_xlim(200, 300)
        #axs.set_ylim(-1380, -1280)
        # plotter.plot_agent_CCLs(axs, plotter.tar_ids[0])


        plt.title(f'Scenario: {plotter.scenario_id} -- Ego: {plotter.ego_id}')
        plt.savefig(f'./ego_change_TCL_{plotter.scenario_id}-{plotter.ego_id}_changeTCL-m6.png', bbox_inches='tight')
        plt.close()





