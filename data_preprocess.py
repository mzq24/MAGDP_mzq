import os
os.environ["CUDA_VISIBLE_DEVICES"]='-1'
import tensorflow as tf
import torch
import time
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
from shapely.geometry import LineString, Point, Polygon

import sys
sys.path.append('..')


# import IPython.display as display
# from waymo_open_dataset.protos import scenario_pb2
# from utils.feature_description import features_description

# from test_scenario_pre_may17 import Scenario_Pre
from scenario_preprocess import Scenario_Pre

from multiprocessing import Pool
from tqdm import tqdm

from dataset_may17 import retrieve_mask

class simAg_data_pre():
    def __init__(self, tf_scenarios_path='/disk2/Waymo_Dataset/scenario/training/', save_to_path='./pyg_data/training', for_sub=False) -> None:
        self.tf_data_path = tf_scenarios_path
        self.save_to = save_to_path
        self.for_sub = for_sub
        self.tf_data_names = os.listdir(self.tf_data_path)[:]

    def process_all(self, number_worker=8, number_shard=-1):
        pbar = tqdm(total=len(list(self.tf_data_names)))
        pbar.set_description(f"Processing {self.tf_data_path.split('/')[-2]}")
        for i, tf_data_name in enumerate(self.tf_data_names):
            tic = time.time()
            self.process_a_tfrecord(tf_data_name, number_worker=number_worker)
            # print(f'tf record {i} - {tf_data_name}, time: {int(time.time()-tic)} sec')
            pbar.update(1)
            if number_shard>0 and i >number_shard:
                break
        pbar.close()

    def process_a_tfrecord(self, tfrecord_name, number_worker=1):
        cur_tfrecord_dataset = tf.data.TFRecordDataset(self.tf_data_path+tfrecord_name, compression_type='')
        if number_worker >1:
            with Pool(processes=number_worker) as p:
                p.map(self.process_a_scenario, list(cur_tfrecord_dataset))
        else:
            for i, data in enumerate(cur_tfrecord_dataset):
                self.process_a_scenario(data)
                
    
    def process_a_scenario(self, data):
        process_for_sub = self.save_to.split('/')[-1] if self.for_sub else ''
        scenario = Scenario_Pre(data, for_sub=process_for_sub)
        
        pyg_data, scn_id = scenario.process()

        # sdc_mask = retrieve_mask(type_list=pyg_data.node_type, wanted_type_set=('sdcAg'))
        # if np.sum(sdc_mask)<1: ## 怎么会有些数据没有 SDC, 有些数据中 SDC 也是 tracks to predict
        #     assert np.sum(sdc_mask)==1, 'no SDC Agent !!!'
        # print(scenario.cur_scenario.tracks_to_predict)
        pyg_name = f'{self.save_to}/{2}-scn-{scn_id}.pyg'
        # print('not saving', pyg_name)
        # print(pyg_data)
        torch.save(pyg_data, pyg_name)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_split", type=str, default='validation', help="data split: training, validation, or test")
    parser.add_argument("--for_sub",    type=bool, default=False, help="data split: training, validation, or test")
    parser.add_argument("--num_worker", type=int, default=32, help="the number of workders for parallel processing")
    parser.add_argument("--num_shards", type=int, default=-1, help="the number of shards to  process")
    opt = parser.parse_args()
    # print(opt)

    # TF_SCENE_PATH = f'./{opt.data_split}/'
    save_folder_name = f'./{opt.data_split}-for_sub' if opt.for_sub else f'./{opt.data_split}/'
    ## 根据不同服务器设置不同数据地址
    myhost = os.uname()[1]
    if myhost == 'AutoManRRCServer':
        TF_SCENE_PATH = f'/disk2/Waymo_Dataset/scenario/{opt.data_split}/'
        SAVE_TO_PATH  = f'/disk6/SimAgent_Dataset/pyg_data_Jun23/{save_folder_name}'
    elif myhost == 'amrrc':
        TF_SCENE_PATH = f'/home/xy/Waymo_Dataset/{opt.data_split}/'
        SAVE_TO_PATH  = f'/home/xy/SimAgent_Dataset/pyg_data_Jun23/{save_folder_name}'
    else: # NSCC
        TF_SCENE_PATH = f'../Waymo_Dataset/{opt.data_split}/'
        SAVE_TO_PATH  = f'../SimAgent_Dataset/pyg_data_Jun23/{save_folder_name}'
        pass # NSCC platform
    if not os.path.exists(SAVE_TO_PATH):
        os.makedirs(SAVE_TO_PATH)

    # print('..')
    data_pre = simAg_data_pre(tf_scenarios_path=TF_SCENE_PATH, save_to_path=SAVE_TO_PATH, for_sub=opt.for_sub)
    # print('..')
    data_pre.process_all(number_worker=opt.num_worker, number_shard=opt.num_shards)