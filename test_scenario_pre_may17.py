import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]='-1'
import tensorflow as tf

import numpy as np
import pandas as pd
import networkx as nx
from itertools import chain
from itertools import product
from itertools import starmap
from functools import partial
from scipy.interpolate import interp1d

from torch_geometric.utils.convert import from_networkx

from protobuf_to_dict import protobuf_to_dict, dict_to_protobuf
from tqdm import tqdm
from typing import Dict, Tuple, Any, List, Callable, Union
from map_utils import get_polylines
from shapely.geometry import LineString, Point, Polygon


# import IPython.display as display
from waymo_open_dataset.protos import scenario_pb2

from map_utils import resample_centerline_nonShapely

class Scenario_Pre():
    def __init__(self, tf_record_data) -> None:
        self.tf_record_data = tf_record_data

    def process(self):
        self.cur_scenario = scenario_pb2.Scenario()
        self.cur_scenario.ParseFromString(self.tf_record_data.numpy())
        # print(dir(self.cur_scenario))
        # print(self.cur_scenario.scenario_id)
        # print(self.cur_scenario._extensions_by_name)
        # print(self.cur_scenario._extensions_by_number)
        self.build_map(self.cur_scenario.map_features, self.cur_scenario.dynamic_map_states)
        self.get_traffic_singal_controlled_lanes(10)
        self.get_lane_polylines()
        self.build_lane_graph()
        data_DG, valid_object_ids = self.build_scene_graph()
        # print(data_DG)
        # del data_DG.raw_xy
        data_PyG = from_networkx(data_DG)
        data_PyG.sdc_cur_state = self.get_cur_sdc_state()
        data_PyG.valid_object_id = valid_object_ids

        return data_PyG, self.cur_scenario.scenario_id
    
    def get_cur_sdc_state(self):
        cur_sdc_id = self.cur_scenario.sdc_track_index
        sdc_cur_state = self.cur_scenario.tracks[cur_sdc_id].states[10]
        # print(sdc_cur_state)
        sdc_cur_xyzh = np.array([sdc_cur_state.center_x, sdc_cur_state.center_y, sdc_cur_state.center_z, sdc_cur_state.heading])
        return sdc_cur_xyzh

    def make_2d_rotation_matrix(self, angle_in_radians: float) -> np.ndarray:
        """
        Makes rotation matrix to rotate point in x-y plane counterclockwise
        by angle_in_radians.
        """
        return np.array([[np.cos(angle_in_radians), -np.sin(angle_in_radians)],
                         [np.sin(angle_in_radians),  np.cos(angle_in_radians)]])

    def build_map(self, map_features, dynamic_map_states):
        self.lanes = {}
        self.roads = {}
        self.stop_signs = {}
        self.crosswalks = {}
        self.speed_bumps = {}

        # static map features
        for map in map_features:
            map_type = map.WhichOneof("feature_data")
            map_id = map.id
            if map_type is None:
                continue
            map = getattr(map, map_type)

            if map_type == 'lane':
                self.lanes[map_id] = map
            elif map_type == 'road_line' or map_type == 'road_edge':
                self.roads[map_id] = map
            elif map_type == 'stop_sign':
                self.stop_signs[map_id] = map
            elif map_type == 'crosswalk': 
                self.crosswalks[map_id] = map
            elif map_type == 'speed_bump':
                self.speed_bumps[map_id] = map
            else:
                raise TypeError

        # dynamic map features
        self.traffic_signals = dynamic_map_states

    def build_lane_graph(self):
        lane_graph_dict = {}
        for k,v in self.lanes.items():
            lane_graph_dict[k] = v.exit_lanes
        self.lane_graph_dict = lane_graph_dict
        self.lane_graph_nx = nx.DiGraph(self.lane_graph_dict)
    
    def get_traffic_singal_controlled_lanes(self, timestep):
        # get traffic light controlled lanes and stop sign controlled lanes
        traffic_light_lanes = {}
        stop_sign_lanes = []

        for signal in self.traffic_signals[timestep].lane_states:
            traffic_light_lanes[signal.lane] = (signal.state, signal.stop_point.x, signal.stop_point.y)
            for lane in self.lanes[signal.lane].entry_lanes:
                traffic_light_lanes[lane] = (signal.state, signal.stop_point.x, signal.stop_point.y)

        for i, sign in self.stop_signs.items():
            stop_sign_lanes.extend(sign.lane)
        
        self.traffic_light_lanes = traffic_light_lanes
        self.stop_sign_lanes = stop_sign_lanes

    def get_polyline_dir(self, polyline):
        """ 得到一个 polyline 上的方向"""
        polyline_pre = np.roll(polyline, shift=1, axis=0)
        polyline_pre[0] = polyline[0]
        diff = polyline - polyline_pre
        polyline_dir = diff / np.clip(np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1000000000)
        return polyline_dir

    def get_lane_polylines(self):
        self.lane_polylines = get_polylines(self.lanes)

    def get_road_polylines(self):
        self.road_polylines = get_polylines(self.roads)

    def get_closest_lanes(self, agent_type, agent_trajectory, lane_polylines, radius=2, closest=False):
        closest_lanes = {}
        min_lane_distance_to_agent = 10000
        for lane_id in lane_polylines.keys():
            if lane_polylines[lane_id].shape[0] > 1:
                distance_to_agent = LineString(lane_polylines[lane_id][:,:2]).distance(Point(agent_trajectory[-1, :2]))
                if distance_to_agent < radius:
                    closest_lanes[lane_id] = lane_polylines[lane_id]
                    if distance_to_agent < min_lane_distance_to_agent:
                        min_lane_distance_to_agent = distance_to_agent
                        closest_lane_id = lane_id
        ## 如果 2m 以内没有 closest lane，就之间返回空集
        if len(closest_lanes.keys())<1:
            return closest_lanes
        if closest:
            return {closest_lane_id: closest_lanes[closest_lane_id]}
        # if closest: # 这里会导致第二递归是找不到，2m 内有几个，但是 1m 内就没有了，这时候就返回空。不要递归，老老实实搞 min
        #     new_radius = radius/2
        #     return  self.get_closest_lanes(agent_type, agent_trajectory, lane_polylines, radius=new_radius, closest=closest) \
        #             if len(closest_lanes.keys())>1 else closest_lanes
        # return closest_lanes
    
    def get_start_lanes(self, agent_type, agent_trajectory, lane_polylines):
        closest_lanes = self.get_closest_lanes(agent_type, agent_trajectory, lane_polylines, closest=True)
        # print(f'closest_lanes {closest_lanes.keys()}')
        assert(len(closest_lanes)<=1)
        closest_lane  = list(closest_lanes.keys())
        # print(f'closest_lane {closest_lane}')
        if len(closest_lane)<1:
            return set()
        left_lanes  = [l_n.feature_id for l_n in self.lanes[closest_lane[0]].left_neighbors]
        right_lanes = [l_n.feature_id for l_n in self.lanes[closest_lane[0]].right_neighbors]

        return set(closest_lane+left_lanes+right_lanes)

    def get_target_lane(self, agent_type, agent_fut, lane_polylines, radius=2):
        target_lane = {}
        for lane_id in lane_polylines.keys():
            if lane_polylines[lane_id].shape[0] > 1:
                    distance_to_agent = LineString(lane_polylines[lane_id][:, :2]).distance(Point(agent_fut[-1, :2]))
                    if distance_to_agent < radius:
                        target_lane[lane_id] = lane_polylines[lane_id]
        return target_lane
    
    def get_shortest_path(self, start_lane_id, target_lane_id):
        return list(nx.shortest_path(self.lane_graph_nx, start_lane_id, target_lane_id))

    def convert_path_to_centerline(self, path):
        centerline = np.concatenate([self.lane_polylines[lane] for lane in path])
        resampled_centerline = resample_centerline_nonShapely(centerline[:,:2], 11)
        return centerline, resampled_centerline

    def convert_raw_coords_to_sdc_frame(self, coordinates: np.ndarray,
                                        translation: Tuple[float, float], 
                                        altitude: float,
                                        yaw:float, state='xyzvxvyh') -> np.ndarray:
        """
        将一条轨迹的坐标通过平移旋转 转换到 translation 和 yaw 指定的坐标系下。
        :param coordinates: (x, y, vx, vy, headings). array of shape [n_steps, 3].
        :param translation: Tuple of (x, y) location that is the center of the new frame.
        :param yaw: yaw angle of the new coords system in radian.
        :return: x,y locations and headings in frame stored in array of share [n_times, 3].
        """
        transform = self.make_2d_rotation_matrix(angle_in_radians=-yaw) # 注意这里用的是 -yaw

        if state == 'xy':
            coords = (coordinates[:,:2] - np.atleast_2d(np.array(translation)[:2])).T # 坐标平移
            local_coords   = np.dot(transform, coords).T # 坐标旋转
            return local_coords
        if state == 'xyz':
            coords_xy = (coordinates[:,:2] - np.atleast_2d(np.array(translation)[:2])).T # 水平坐标平移
            local_coords_xy = np.dot(transform, coords_xy).T # 坐标旋转
            local_coords_z  = coordinates[:,2:] - altitude   # 垂直平移
            return np.concatenate((local_coords_xy, local_coords_z), axis=1)
        elif state == 'xyvxvyh':
            coords = (coordinates[:,:2] - np.atleast_2d(np.array(translation)[:2])).T # 坐标平移
            local_coords   = np.dot(transform, coords).T # 坐标旋转
            local_velocities = np.dot(transform, coordinates[:,2:4].T).T # 速度仅旋转
            local_headings = coordinates[:,4:] - yaw # 角度变化
            return np.concatenate((local_coords, local_velocities, local_headings), axis=1)
        elif state == 'xyzvxvyh':
            coords = (coordinates[:,:2] - np.atleast_2d(np.array(translation)[:2])).T # 坐标平移
            local_coords_xy  = np.dot(transform, coords).T               # 坐标旋转
            local_coords_z  = coordinates[:,2:3] - altitude               # 垂直平移
            local_velocities = np.dot(transform, coordinates[:,3:5].T).T # 速度仅旋转
            local_coords_h = coordinates[:,5:] - yaw # 角度变化
            return np.concatenate((local_coords_xy, local_coords_z, local_velocities, local_coords_h), axis=1)
        elif state == 'xyzh':
            coords = (coordinates[:,:2] - np.atleast_2d(np.array(translation)[:2])).T # 坐标平移
            local_coords_xy = np.dot(transform, coords).T     # 坐标旋转
            local_coords_z  = coordinates[:,2:3] - altitude   # 垂直平移
            local_coords_h  = coordinates[:,3:] - yaw         # 角度变化
            return np.concatenate((local_coords_xy, local_coords_z, local_coords_h), axis=1)

    def get_agents_to_predict(self):
        
        return [cur_pred.track_index for cur_pred in self.cur_scenario.tracks_to_predict]
        # return self.cur_scenario.tracks_to_predict
    
    def get_a_frame(self, frame_index):
        """ 从所有 tracks 中得到一个 frame_index 指定的 frame，并转换成 DataFrame 返回。这里的 index 就是 track_id """
        return pd.DataFrame([protobuf_to_dict(self.cur_scenario.tracks[i].states[frame_index]) for i in range(len(self.cur_scenario.tracks))])
    
    def get_neighbors(self, frame, agent_id, radii=50, valid_only=True):
        """得到 agent_id 的邻居节点，返回 [[nbr_id, nbr_x, nbr_y]]"""
        agent_pos = frame.loc[agent_id][['center_x', 'center_y']].values
        all_pos = frame[['center_x', 'center_y']].values
        cur_dist_to_agn = np.linalg.norm(all_pos.astype(float) - agent_pos.astype(float), axis=1)
        frame.insert(10, "dist_to_agent", cur_dist_to_agn)
        frame.insert(0, "agent_index", [i for i in range(frame.shape[0])])
        # print(frame[frame['valid']==True])
        if valid_only:
            return frame[(frame['valid']==True)&(frame['dist_to_agent']<=radii)&(frame['dist_to_agent']>=0.01)][['agent_index', 'center_x', 'center_y']].values
        else:
            return frame[(frame['dist_to_agent']<=radii)&(frame['dist_to_agent']>=0.01)][['agent_index', 'center_x', 'center_y']].values
        
    def get_xyzvxvyh_from_track(self, track):
        return np.array([(state.center_x, state.center_y, state.center_z, state.velocity_x, state.velocity_y, state.heading) for state in track]), np.array([state.valid for state in track])
    
    def get_xyzh_from_track(self, track):
        return np.array([(state.center_x, state.center_y, state.center_z, state.heading) for state in track]), np.array([state.valid for state in track])
    
    def get_hist(self, agn_id, sdc_centered=True):
        agn_track = self.cur_scenario.tracks[agn_id].states
        # assert len(agn_track)==11 
        agn_hist, hist_valid = self.get_xyzvxvyh_from_track(agn_track[:11])
        # print(f'agn_hist {agn_hist.shape}, hist_valid {hist_valid.shape}')
        raw_hist_xy = agn_hist[:,:2]
        valid_raw_hist_xy = agn_hist[hist_valid,:2]
        
        if sdc_centered: # 如果要以 self driving car 为中心处理数据的话，需要转换到 SDC 坐标系下
            cur_sdc_id = self.cur_scenario.sdc_track_index
            sdc_cur_state = self.cur_scenario.tracks[cur_sdc_id].states[10]
            # print(sdc_cur_state)
            sdc_cur_xyzh = np.array([sdc_cur_state.center_x, sdc_cur_state.center_y, sdc_cur_state.center_z, sdc_cur_state.heading])
            ## 坐标转换
            agn_hist = self.convert_raw_coords_to_sdc_frame(agn_hist, sdc_cur_xyzh[:2], sdc_cur_xyzh[2], sdc_cur_xyzh[3], state='xyzvxvyh')
            ## 将 invalid 值都改成 0 
            agn_hist[~hist_valid] = np.zeros((sum(~hist_valid), 6))
            return agn_hist, raw_hist_xy, cur_sdc_id, sdc_cur_xyzh
        else:
            return agn_hist, valid_raw_hist_xy
        
    def get_FAKE_attr(self):
        return np.zeros(4,)

    def get_FAKE_history(self):
        return np.zeros((11,6))
    
    def get_dfs_tree_edges(self, start_lane_id, depth_limit=2):
        edges = []
        dfs_tree = nx.dfs_tree(self.lane_graph_nx, start_lane_id, depth_limit)
        leaves = (v for v, d in dfs_tree.out_degree() if d == 0)
        # all_paths = partial(nx.all_simple_paths, dfs_tree)
        for leaf in leaves:
            for path in nx.all_simple_paths(dfs_tree, source=start_lane_id, target=leaf):
                edges.append(path)
        return edges
        
    def build_scene_graph(self):
        DG = nx.DiGraph()
        valid_object_ids = []
        ## 将 SDC 加入图中
        sdc_id = self.cur_scenario.sdc_track_index
        DG = self.add_an_agent_to_a_graph(dg=DG, agn_id=sdc_id, agn_type='sdcAg')
        valid_object_ids.append(self.cur_scenario.tracks[sdc_id].id)
        ## 将 tracks_to_predict 加入图中
        target_agents = self.get_agents_to_predict()
        # valid_agents = self.get_sim_agent_indexes()
        
        # return DG
        # print(target_agents)
        for tar_ag_id in target_agents:
            if tar_ag_id == sdc_id:
                continue
            DG = self.add_an_agent_to_a_graph(dg=DG, agn_id=tar_ag_id, agn_type='tarAg')
            valid_object_ids.append(self.cur_scenario.tracks[tar_ag_id].id)

        ## 将 SDC 以及　tracks_to_predict 的邻居们加入图中
        set_of_sdc_and_tar = list(DG.nodes)
        for agn_id in set_of_sdc_and_tar:
            cur_frame = self.get_a_frame(self.cur_scenario.current_time_index)
            agn_nbrs = self.get_neighbors(frame=cur_frame, agent_id=agn_id, radii=50, valid_only=True)[:,0].astype(int)
            for nbr_id in agn_nbrs:
                if not DG.has_node(nbr_id):
                    DG = self.add_an_agent_to_a_graph(dg=DG, agn_id=nbr_id, agn_type='nbrAg')
                    valid_object_ids.append(self.cur_scenario.tracks[nbr_id].id)

        ## 将所有的 current valid agent 加入图中
        # 这里应该和前面的 agn index 的顺序对应上，每当 add_an_agent _a_graph 的时候，就找到 agn index 对应的 objectid，并加入到列表中
        valid_agents = self.get_sim_agent_indexes()
        for v_id in valid_agents:
            if not DG.has_node(v_id):
                DG = self.add_an_agent_to_a_graph(dg=DG, agn_id=v_id, agn_type='nbrAg')
                valid_object_ids.append(self.cur_scenario.tracks[v_id].id)

        ## 将 每个车的 Agent 的CCLs 加入图中
        all_agents = list(DG.nodes)
        for agn_id in all_agents:
            agn_type = nx.get_node_attributes(DG, 'node_type')[agn_id]
            # print(f'agn_type {agn_type}')
            DG = self.add_CCLs_of_an_agent_to_a_graph(DG, agn_id, agn_type=agn_type)
        
        return DG, valid_object_ids

    def get_sim_agent_indexes(self):
        """Returns the list of object IDs that needs to be resimulated.

        Internally calls `is_valid_sim_agent` to verify the simulation criteria,
        i.e. is the object valid at `current_time_index`.

        Args:
            scenario: The Scenario proto containing the data.

        Returns:
            A list of int IDs, containing all the objects that need to be simulated.
        """
        track_indexs = []
        CURRENT_TIME_INDEX = 10
        for track_idx, track in enumerate(self.cur_scenario.tracks):
            if track.states[CURRENT_TIME_INDEX].valid:
                track_indexs.append(track_idx)
        return track_indexs

    def get_agent_attr(self, agn_id):
        agn_cur_state = self.cur_scenario.tracks[agn_id].states[10]
        agn_attr = np.array([agn_cur_state.length, agn_cur_state.width, agn_cur_state.height, self.cur_scenario.tracks[agn_id].object_type])
        return agn_attr

    def add_an_agent_to_a_graph(self, dg, agn_id, agn_type):
        ## 将节点和边都要加上，每个节点要链接到它的邻居上
        agn_hist, raw_hist_xy, _, _ = self.get_hist(agn_id, sdc_centered=True)
        # print(f'goal_valid: {goal_valid} {type(goal_valid)}')
        agn_attr = self.get_agent_attr(agn_id)
        agn_node_name = f'{self.cur_scenario.scenario_id}-{agn_id}'
        dg.add_node(agn_id, node_name=agn_node_name, node_feature=agn_hist, raw_xy=raw_hist_xy, fut_valid=True, goal_valid=True, node_type=agn_type, attr=agn_attr) # Other agent node
        dg.add_edge(agn_id, agn_id, edge_name=f'{agn_id}->{agn_id}', edge_type=f'{agn_type}-Loop') # Neighbor A-loop
        ## 将节点链接到当前图中所有的邻居节点上
        cur_frame = self.get_a_frame(self.cur_scenario.current_time_index)
        agn_nbrs  = self.get_neighbors(cur_frame, agn_id, radii=50, valid_only=True)
        agn_nbrs  = agn_nbrs[:,0]
        # print(f'agn_nbrs: {agn_nbrs}')
        for nbr_id in agn_nbrs:
            if dg.has_node(nbr_id):
                nbr_type = nx.get_node_attributes(dg, 'node_type')[nbr_id]
                # print(f'nbr_type {nbr_type}')
                dg.add_edge(nbr_id, agn_id, edge_name=f'{nbr_id}->{agn_id}', edge_type=f'{nbr_type}-{agn_type}') # 只加上邻居到自身的边，后面处理邻居的时候会再加上 自身到邻居的边
        return dg

    def add_CCLs_of_an_agent_to_a_graph(self, dg, agn_id, agn_type):
        agn_CCLs = self.get_CCLs_of_an_agent(agn_id, agn_type)
        ccl_of_goal_valid_agn = nx.get_node_attributes(dg, 'goal_valid')[agn_id]

        for i, agn_ccl in enumerate(agn_CCLs):
            ccl_attr = agn_ccl[2]
            ## 世界坐标系中的 CCL
            ccl_polyline = agn_ccl[1]
            cur_sdc_id = self.cur_scenario.sdc_track_index
            sdc_cur_state = self.cur_scenario.tracks[cur_sdc_id].states[10]
            sdc_cur_xyzh = np.array([sdc_cur_state.center_x, sdc_cur_state.center_y, sdc_cur_state.center_z, sdc_cur_state.heading])
            ## 坐标转换到本地坐标系
            local_ccl = self.convert_raw_coords_to_sdc_frame(ccl_polyline, sdc_cur_xyzh[:2], sdc_cur_xyzh[2], sdc_cur_xyzh[3], state='xy')
            local_ccl_dir = self.get_polyline_dir(local_ccl)
            local_ccl_6 = np.concatenate((local_ccl, local_ccl_dir, np.zeros((11,2))), axis=1)
            ccl_type = f'{agn_type[:3]}CCL'
            ccl_id   = f'{agn_id}CCL{i}'   
            # print(f'ccl_id {ccl_id}, ccl_type {ccl_type}')
            ccl_node_name = f'{self.cur_scenario.scenario_id}-{ccl_id}'
            dg.add_node(ccl_id, node_name=ccl_node_name, node_feature=local_ccl_6, raw_xy=ccl_polyline, fut_valid=False, goal_valid=ccl_of_goal_valid_agn, node_type=ccl_type, attr=ccl_attr) # Other agent node
            dg.add_edge(ccl_id, ccl_id, edge_name=f'{ccl_id}->{ccl_id}', edge_type=f'{ccl_type}-Loop') 
            dg.add_edge(ccl_id, agn_id, edge_name=f'{ccl_id}->{agn_id}', edge_type=f'{ccl_type}-{agn_type}') 
        if len(agn_CCLs)==0 :
            ## 找不到 Agent 的 CCL，就构造一个 Fake CCL 便于处理计算
            fake_ccl_type = f'{agn_type[:3]}CCL_FAKE'
            fake_ccl_id   = f'{agn_id}CCL_FAKE'
            fake_ccl_node_name = f'{self.cur_scenario.scenario_id}-{fake_ccl_id}'
            fake_hist = self.get_FAKE_history()
            fake_ccl_attr = self.get_FAKE_attr()
            dg.add_node(fake_ccl_id, node_name=fake_ccl_node_name, node_feature=fake_hist, raw_xy=fake_hist, fut_valid=False, goal_valid=ccl_of_goal_valid_agn, node_type=fake_ccl_type, attr=fake_ccl_attr) # Other agent node
            dg.add_edge(fake_ccl_id, fake_ccl_id, edge_name=f'{fake_ccl_id}->{fake_ccl_id}', edge_type=f'{fake_ccl_type}-Loop') 
        return dg
    
    def get_ccl_attr(self, agn_ccl):
        ccl_traffic_light = self.traffic_light_lanes[agn_ccl[-1]][0] if agn_ccl[-1] in self.traffic_light_lanes.keys() else 0.0
        ccl_stop_sign   = 1.0 if agn_ccl[-1] in self.stop_sign_lanes else 0.0        
        ccl_speed_limit = self.lanes[agn_ccl[-1]].speed_limit_mph 
        ccl_speed_bump  = 1.0 if agn_ccl[-1] in self.speed_bumps else 0.0
        return np.array([ccl_traffic_light, ccl_stop_sign, ccl_speed_limit, ccl_speed_bump])

    def get_CCLs_of_an_agent(self, agn_id, agn_type):
        """ 1. 得到当前所在 lane 和相邻的 lane （left 和 right）
            2. 从这个几个 lane 出发得到 dfs tree 
            3. 每个lane可能有自己的状态，而一个 CCL 可能有多个 lane，就用最远处的 lane 的状态作为 CCL的状态 """
        CCLs = []
        # if agn_id != self.cur_scenario.sdc_track_index:
        #     return CCLs
        _, agn_traj = self.get_hist(agn_id, sdc_centered=False)
        agn_type = None
        start_lanes = self.get_start_lanes(agent_type=agn_type, agent_trajectory=agn_traj, lane_polylines=self.lane_polylines)
        # print(f'start_lanes {start_lanes}')
        if len(start_lanes) == 0:
            return CCLs
        for lane_id in start_lanes:
            dfs_tree = self.get_dfs_tree_edges(lane_id, depth_limit=2)
            ## 这里怎么能少了缩进！！！
            ## 导致只看最后的一个start lane 出发的 dfs_tree，少了很多 CCL！
            for ccl_path in dfs_tree:
                ccl_line, resampled_ccl_line = self.convert_path_to_centerline(ccl_path)
                ccl_attr = self.get_ccl_attr(ccl_path)
                CCLs.append((ccl_path, resampled_ccl_line, ccl_attr, ccl_line))
        # print(f'CCLs {[ccl[0] for ccl in CCLs]}')
        return CCLs
