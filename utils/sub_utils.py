from utils.test_utils import get_agn_tcl_index, get_agn_names, get_batch_tcl_indexes, set_TCL_of_an_agent, get_agents_with_many_CCLs, sample_points_from_gaussians
from dataset_may17 import retrieve_mask
import numpy as np
import torch
import copy
from multiprocessing import Pool
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points

import time


def init_TCL_for_agents(flat_data, goalset):
    tcl_indexes = get_batch_tcl_indexes(flat_data.flat_node_name, goalset, flat_data.node_feature)
    # print(len(tcl_indexes))
    flat_data.flat_node_type[tcl_indexes] = 'tarTCL'
    return flat_data

def pull_goal_to_TCL(pred_goalset, pyg_data):
        print(np.unique(pyg_data.flat_node_type))
        tcl_mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('tarTCL', 'tarTCL_FAKE'))
        TCLs = pyg_data.node_feature[tcl_mask][:,:,:2]
        # print(TCLs.shape)
        new_goalset = []
        for i, goal in enumerate(pred_goalset):
            ## 查看是不是 FAKE TCL
            if torch.sum(torch.abs(TCLs[i]))<1:
                new_goal=goal.detach().numpy()
                new_goalset.append(new_goal)
            else:
                new_goal = nearest_points(LineString(TCLs[i].detach().numpy()), Point(goal.detach().numpy()))[0]
                # print(new_goal.coords)
                new_goalset.append(np.array([new_goal.x, new_goal.y]))
        return torch.tensor(np.array(new_goalset), device=pred_goalset.device, dtype=goal.dtype)


class Rollouts_Sampler():
    def __init__(self, MGA_model, TGE_model, GDP_model) -> None:
    # def __init__(self, MGA_model, GDP_model) -> None:
        self.MGA_model = MGA_model
        self.TGE_model = TGE_model
        self.GDP_model = GDP_model

    def init_TCL_for_agents(self, flat_data, goalset):
        tcl_indexes = get_batch_tcl_indexes(flat_data.flat_node_name, goalset, flat_data.node_feature)
        # print(len(tcl_indexes))
        flat_data.flat_node_type[tcl_indexes] = 'tarTCL'
        return flat_data

    def pull_goal_to_TCL(self, pred_goalset, pyg_data):
        tcl_mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('tarTCL'))
        TCLs = pyg_data.node_feature[tcl_mask][:,:,:2]
        # print(TCLs.shape)
        new_goalset = []
        for i, goal in enumerate(pred_goalset):
            ## 查看是不是 FAKE TCL
            if torch.sum(torch.abs(TCLs[i]))<1:
                new_goal=goal.detach().cpu().numpy()
                new_goalset.append(new_goal)
            else:
                new_goal = nearest_points(LineString(TCLs[i].detach().cpu().numpy()), Point(goal.detach().cpu().numpy()))[0]
                # print(new_goal.coords)
                new_goalset.append(np.array([new_goal.x, new_goal.y]))
        return torch.tensor(new_goalset, device=pred_goalset.device, dtype=goal.dtype)

    def parallel_sample(self, flat_data_list, number_worker=-1):
        if number_worker >0:
            with Pool(processes=number_worker) as p:
                p.map(self.sample_rollouts, flat_data_list)

    def sample_rollouts(self, raw_flat_data, smoothen=False):
        # print(raw_flat_data.flat_node_type)
        # print(raw_flat_data)
        Rollouts = []
        # print(f'num_goal_valid_agn {raw_flat_data.num_goal_valid_agn}')
        # 输出 M=32 个 joint goalsets
        goalsets_pred = self.MGA_model(raw_flat_data)

        #################################################
        ## 对 nbr 手动设置 goal 为当前位置
        # nbr_agn_mask_in_data_batch = retrieve_mask(type_list=raw_flat_data.flat_node_type, wanted_type_set=('nbrAg'))
        # nbr_mask_in_agents = retrieve_mask(type_list=raw_flat_data.flat_agent_node_type, wanted_type_set=('nbrAg'))
        # nbr_cur_xy = raw_flat_data.node_feature[nbr_agn_mask_in_data_batch,-1,:2]
        # goalsets_pred[0][nbr_mask_in_agents]=nbr_cur_xy.unsqueeze(1).repeat(1,32,1)
        #################################################


        # 对每一个 goalset 输出 1 个 rollout, 共 6 个 goal set
        for goalset_i, goalset_init in enumerate(goalsets_pred[0].permute(1,0,2)):
            # print(f'goalset_init {goalset_init.shape}')
            ## 初始化 TCLs
            goalset_tic = time.time()
            goalset_flat_data = self.init_TCL_for_agents(copy.deepcopy(raw_flat_data), goalset_init)

            ### TGE mm Goals ########################
            tge_mm_goals = self.TGE_model(goalset_flat_data)
            # print(f'tge_mm_goals {tge_mm_goals.shape}')

            for tge_goals in tge_mm_goals.permute(1,0,2,3):
                ## 将 goalset 调整到 TCL 上面
                new_goalset = self.pull_goal_to_TCL(tge_goals.squeeze(1), goalset_flat_data)
                # new_goalset = tge_goals.squeeze(1)
                # print(f'new_goalset {new_goalset.shape}')
                goalset_flat_data.sampled_goalset = new_goalset

                # 运行 GDP 得到最终轨迹
                gdp_tic = time.time()
                magdp = self.GDP_model(goalset_flat_data.to(self.GDP_model.args['device']))
                if smoothen:
                    magdp = smoothen_a_rollout(magdp.detach().numpy())
                else:
                    magdp = magdp.detach().numpy()
                # print(f'{time.time()-gdp_tic}secs for gdp forward')
                # print(f'magdp {magdp[0].shape} {len(magdp)}')
                Rollouts.append(magdp)
                # break
            # break
        # print(len(Rollouts), Rollouts[0].shape)
        print(Rollouts[0].dtype)
        return Rollouts[:32]

from typing import Dict, Tuple, Any, List, Callable, Union

def smoothen_a_rollout(rollout):
    """ rollout of shape [#Agent, 80, 6]
    """
    rollout_smooth = np.zeros_like(rollout)
    # Set the window size for moving average
    window_size = 11
    for a_i in range(rollout.shape[0]):
        f = rollout[a_i]
        for d_i in range(rollout.shape[2]):
            padded_d = np.pad(f[:,d_i], (window_size//2, window_size-1-window_size//2), mode='edge')
            rollout_smooth[a_i,:,d_i] = np.convolve(padded_d, np.ones(window_size)/window_size, mode='valid')
    return rollout_smooth


def make_2d_rotation_matrix(angle_in_radians: float) -> np.ndarray:
    """
    Makes rotation matrix to rotate point in x-y plane counterclockwise
    by angle_in_radians.
    """
    return np.array([[np.cos(angle_in_radians), -np.sin(angle_in_radians)],
                        [np.sin(angle_in_radians),  np.cos(angle_in_radians)]])

def convert_coords_from_sdc_to_world_frame(coordinates: np.ndarray,
                                            translation: Tuple[float, float], 
                                            altitude: float,
                                            yaw:float, state='xyzh') -> np.ndarray:
    """
    将模型输出的fut 通过平移旋转 转换到 translation 和 yaw 指定的坐标系下。
    :param coordinates: (x, y, z, headings). array of shape [n_steps, 4].
    :param translation: Tuple of (x, y) location that is the center of the new frame.
    :param yaw: yaw angle of the new coords system in radian.
    :return: x,y locations and headings in frame stored in array of share [n_times, 3].
    """
    transform = make_2d_rotation_matrix(angle_in_radians= yaw) # 从 sdc 转回到 world frame 时应该用 yaw （SDC 在原来坐标系中的转角）

    if state == 'xyzh':
        # coords = 
        coords = np.dot(transform, coordinates[:,:2].T).T     # 坐标旋转
        world_coords_xy = coords + np.atleast_2d(np.array(translation)[:2]) # 坐标平移，加上 SDC 在原来坐标系中的位置
        world_coords_z  = coordinates[:,2:3] + altitude   # 垂直平移
        world_coords_h  = coordinates[:,3:]  + yaw         # 角度变化
        return np.concatenate(( world_coords_xy, 
                                world_coords_z, 
                                world_coords_h), axis=1)
    
