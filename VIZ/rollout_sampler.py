from utils.test_utils import get_agn_tcl_index, get_agn_names, get_batch_tcl_indexes, set_TCL_of_an_agent, get_agents_with_many_CCLs, sample_points_from_gaussians
from dataset_may17 import retrieve_mask
import numpy as np
import torch
import copy
from multiprocessing import Pool
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points

import time

class Rollouts_Sampler():
    def __init__(self, MGA_model, GDP_model, pull_to_TCL=True, Smoothen_GDP=True, use_TGE=False) -> None:
    # def __init__(self, MGA_model, GDP_model) -> None:
        self.MGA_model = MGA_model
        self.GDP_model = GDP_model
        self.pull_to_TCL = pull_to_TCL
        self.Smoothen_GDP = Smoothen_GDP
        self.use_TGE = use_TGE

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

    def sample_rollouts(self, raw_flat_data, init_TCL=True, manul_goal=None):
        # print(raw_flat_data.flat_node_name)
        # print(raw_flat_data)
        Rollouts = []
        Rollouts_change_TCL = []
        # print(f'num_goal_valid_agn {raw_flat_data.num_goal_valid_agn}')
        # 输出 M=32 个 joint goalsets
        goalsets_pred = self.MGA_model(raw_flat_data)
        ego_mask =  retrieve_mask(type_list=raw_flat_data.flat_node_type, wanted_type_set=('sdcAg'))
        tar_mask =  retrieve_mask(type_list=raw_flat_data.flat_node_type, wanted_type_set=('tarAg'))
            
        ego_name  = raw_flat_data.flat_node_name[ego_mask][0]
        tar_names = raw_flat_data.flat_node_name[tar_mask]
                
        # print('-'*60)
                
        # print(f'ego name is {ego_name}')
        # print(f'tar names are {tar_names}')

        ego_ccls_mask = [True if node_name.startswith(f'{tar_names[0]}CCL') else False for node_name in raw_flat_data.flat_node_name]
        # print(f'ego_ccls_mask is {ego_ccls_mask}')

        # print('-'*60)

        # 对每一个 goalset 输出 1 个 rollout, 共 6 个 goal set
        # print(goalsets_pred[0].shape)
        for goalset_i, goalset_init in enumerate(goalsets_pred[0].permute(1,0,2)):
            # print(f'goalset_init {goalset_init.shape}')
            ## 初始化 TCLs
            goalset_tic = time.time()
            if init_TCL:
                goalset_flat_data = self.init_TCL_for_agents(copy.deepcopy(raw_flat_data), goalset_init)
            else:
                goalset_flat_data = raw_flat_data
            goalset_flat_data_raw = goalset_flat_data.clone()
            ### TGE mm Goals ########################
            if not self.use_TGE:
                ## 将 goalset 调整到 TCL 上面
                new_goalset = self.pull_goal_to_TCL(goalset_init, goalset_flat_data)
                goalset_flat_data.sampled_goalset = new_goalset
                print(goalset_flat_data.sampled_goalset.shape)
                if manul_goal is not None:
                    goalset_flat_data.sampled_goalset[0] = manul_goal
                    pass
                # 运行 GDP 得到最终轨迹 + Smoothen
                magdp = self.GDP_model(goalset_flat_data.to(self.GDP_model.args['device']))
                magdp = smoothen_a_rollout(magdp.detach().numpy())
                
                Rollouts.append(magdp)
                
                # Shuffle TCL of Ego
                print(f'ego ccls {goalset_flat_data.flat_node_type[ego_ccls_mask]}')
                current_ego_ccls = goalset_flat_data.flat_node_type[ego_ccls_mask]
                ego_ccl_indexes = np.where(ego_ccls_mask)[0]
                # print(f'ego_ccl_indexes {ego_ccl_indexes}')
                for i in range(10):
                    np.random.shuffle(goalset_flat_data.flat_node_type[ego_ccl_indexes[0]: ego_ccl_indexes[-1]+1])
                    if (goalset_flat_data.flat_node_type[ego_ccls_mask] != current_ego_ccls).any():
                        break
                # print(f'ego ccls {goalset_flat_data.flat_node_type[ego_ccls_mask]}')
                
                ## 将 goalset 调整到 新的 TCL 上面
                new_goalset = self.pull_goal_to_TCL(goalset_init, goalset_flat_data)
                goalset_flat_data.sampled_goalset = new_goalset
                # 运行 GDP 得到最终轨迹 + Smoothen
                magdp = self.GDP_model(goalset_flat_data.to(self.GDP_model.args['device']))
                magdp = smoothen_a_rollout(magdp.detach().numpy())
                Rollouts_change_TCL.append(magdp)
                
                print(goalset_flat_data.flat_node_type)
                print(goalset_flat_data.num_wld_agn)

            break
            # else:
            #     tge_mm_goals = self.TGE_model(goalset_flat_data)
            #     for tge_goals in tge_mm_goals.permute(1,0,2,3):

            #         if self.pull_to_TCL:
            #             ## 将 goalset 调整到 TCL 上面
            #             new_goalset = self.pull_goal_to_TCL(tge_goals.squeeze(1), goalset_flat_data)
            #             goalset_flat_data.sampled_goalset = new_goalset
            #         else:
            #             goalset_flat_data.sampled_goalset = tge_goals.squeeze(1)
            #         # 运行 GDP 得到最终轨迹
            #         magdp = self.GDP_model(goalset_flat_data.to(self.GDP_model.args['device']))
            #         if self.Smoothen_GDP:
            #             magdp = smoothen_a_rollout(magdp.detach().numpy())
            #         else:
            #             magdp = magdp.detach().numpy()

            #         Rollouts.append(magdp)
                # break
            # break
        # print(len(Rollouts), Rollouts[0].shape)
        # print(Rollouts[0].dtype)
        return Rollouts, Rollouts_change_TCL, goalset_flat_data_raw

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
    
