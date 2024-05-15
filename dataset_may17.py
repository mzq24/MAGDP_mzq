import os
import pickle
import os.path as osp
import time
import torch
import numpy as np
from torch_geometric.data   import Dataset
from torch_geometric.loader import DataLoader

# import functools
# import operator
import sys
sys.path.append('..')
sys.path.append('../may07')


class SimAgn_Dataset(Dataset):
    def __init__(self, data_path='/home/xy/simAgentWaymo2023/may07/pyg_data_may17/training', dec_type='TGE'):
        # Initialization
        self.data_path = data_path
        self.dec_type = dec_type
        self.data_names = os.listdir(self.data_path)
        # print('there are {} data pieces'.format(self.__len__()))

        super(SimAgn_Dataset).__init__()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_names)

    def __getitem__(self, index):        
        try:
            data_item = torch.load(f'{self.data_path}/{self.data_names[index]}')
        except:
            return self.__getitem__(index+1)
        sdc_mask = retrieve_mask(type_list=data_item.node_type, wanted_type_set=('sdcAg'))
        if np.sum(sdc_mask)<1: ## 怎么会有些数据没有 SDC, 有些数据中 SDC 也是 tracks to predict
            # assert np.sum(sdc_mask)==1 , f'{np.sum(sdc_mask)}'
            data_item.node_type[0]='sdcAg'            # return self.__getitem__(index+1)
            
        # print(index, data_item.keys)
        # print(sdc_mask)
        # data_item.sdc_cur_state = data_item.raw_xy[1][-1]
        if self.dec_type != 'test1':
            del data_item.raw_xy
        data_item.node_feature = data_item.node_feature.float()
        data_item.attr = data_item.attr.float()

        agn_mask = retrieve_mask(type_list=data_item.node_type, wanted_type_set=('sdcAg', 'tarAg', 'nbrAg'))
         
        ccl_mask = retrieve_mask(type_list=data_item.node_type, wanted_type_set=('sdcCL', 'sdcTCL', 'sdcTCL_FAKE',
                                                                                 'tarCL', 'tarTCL', 'tarTCL_FAKE',
                                                                                 'nbrCL', 'nbrTCL', 'nbrTCL_FAKE'))
        
        sdc_mask     = retrieve_mask(type_list=data_item.node_type, wanted_type_set=('sdcAg'))
        sdc_tar_mask = retrieve_mask(type_list=data_item.node_type, wanted_type_set=('sdcAg', 'tarAg'))
        wld_agn_mask = retrieve_mask(type_list=data_item.node_type, wanted_type_set=('tarAg', 'nbrAg'))
        
        data_item.num_agn = np.sum(agn_mask)
        data_item.num_ccl = np.sum(ccl_mask)
        data_item.num_wld_agn = torch.sum(torch.from_numpy(wld_agn_mask))
        if data_item.num_wld_agn ==0:
            return self.__getitem__(index+1)
        # 改这里！！！
        # if self.dec_type in ['MGA']:
        #     goal_valid_agn_mask = np.logical_and(sdc_tar_mask, data_item.goal_valid.cpu().numpy())
        #     data_item.num_goal_valid_agn = torch.sum(goal_valid_agn_mask)
        # elif self.dec_type in ['GDP']:
        #     fut_valid_agn_mask  = np.logical_and(sdc_tar_mask, data_item.fut_valid.cpu().numpy())
        #     data_item.num_fut_valid_agn = torch.sum(fut_valid_agn_mask)
        ## 将所有list 转换为 np array
        data_item.node_type = np.array(data_item.node_type)
        data_item.edge_type = np.array(data_item.edge_type)
        data_item.edge_name = np.array(data_item.edge_name)
        data_item.node_name = np.array(data_item.node_name)
        data_item.agent_node_type = data_item.node_type[agn_mask]

        # if not 'test' in self.data_path:
        if not self.dec_type in  ['test']:
            data_item.fut   = data_item.fut.float()
            # data_item.goals = torch.nan_to_num(data_item.fut[:,-1,:2], nan=0.0)
            ## 针对所有的 sdc 和 tar 来做
            if  self.dec_type in  ['TGE', 'DGE', 'TGEall']:
                if self.dec_type in  ['TGE', 'DGE']:
                    goal_valid_agn_mask = np.logical_and(sdc_tar_mask, data_item.goal_valid.cpu().numpy())
                elif self.dec_type == 'TGEall':
                    goal_valid_agn_mask = np.logical_and(agn_mask, data_item.goal_valid.cpu().numpy())
                data_item.agn_goal_set = data_item.fut[goal_valid_agn_mask,-1:,:2].float()
            elif self.dec_type in ['Mot']:
                tar_mask = retrieve_mask(type_list=data_item.node_type, wanted_type_set=('tarAg'))
                fut_valid_tar_mask = np.logical_and(tar_mask, data_item.fut_valid.cpu().numpy())
                data_item.mot_label = data_item.fut[fut_valid_tar_mask,::5,:2].float()#.unsqueeze(1)
            elif self.dec_type in ['MGA', 'MGAall']:
                if self.dec_type == 'MGA':
                    goal_valid_agn_mask = np.logical_and(sdc_tar_mask, data_item.goal_valid.cpu().numpy())
                elif self.dec_type == 'MGAall':
                    goal_valid_agn_mask = np.logical_and(agn_mask, data_item.goal_valid.cpu().numpy())
                data_item.agn_goal_set = data_item.fut[goal_valid_agn_mask,-1:,:2].float()
                data_item.num_goal_valid_agn = torch.sum(torch.from_numpy(goal_valid_agn_mask))
            elif self.dec_type in ['MGAsdc']:
                goal_valid_agn_mask = np.logical_and(sdc_mask, data_item.goal_valid.cpu().numpy())
                data_item.agn_goal_set = data_item.fut[goal_valid_agn_mask,-1:,:2].float()
                data_item.num_goal_valid_agn = torch.sum(torch.from_numpy(goal_valid_agn_mask))
                agn_CCLs_list = []
                for agn_node_name in data_item.node_name[goal_valid_agn_mask]:
                    agn_ccl_mask = [True if (node_name.startswith(f'{agn_node_name}TCL') or node_name.startswith(f'{agn_node_name}CCL')) else False for node_name in data_item.node_name ]
                    agn_CCLs_list.append(data_item.node_feature[agn_ccl_mask][:,:,:2])
                    # print(agn_node_name)
                    # agn_CCLs_dict[agn_node_name] = data_item.node_feature[agn_ccl_mask][:,:,:2]
                data_item.agn_CCLs = (agn_CCLs_list,)
            elif self.dec_type in ['GDP', 'arGDP', 'GDPrnn', 'GDPall']:
                if self.dec_type in ['GDP', 'arGDP', 'GDPrnn']:
                    fut_valid_agn_mask  = np.logical_and(sdc_tar_mask, data_item.fut_valid.cpu().numpy())
                elif self.dec_type == 'GDPall':
                    fut_valid_agn_mask  = np.logical_and(agn_mask, data_item.fut_valid.cpu().numpy())
                data_item.agn_goal_set = data_item.fut[fut_valid_agn_mask, -1,:2].float()
                data_item.num_fut_valid_agn = torch.sum(torch.from_numpy(fut_valid_agn_mask))
                data_item.gdp_label = data_item.fut[fut_valid_agn_mask, :, :].float()
                data_item.goals = torch.ones_like(data_item.fut[:,-1,:2])
                data_item.goals[fut_valid_agn_mask] = data_item.agn_goal_set
            elif self.dec_type in ['arGDPmax']:
                fut_valid_agn_mask  = np.logical_and(sdc_tar_mask, data_item.fut_valid.cpu().numpy())
                data_item.agn_goal_set = data_item.fut[fut_valid_agn_mask, -1,:2].float()
                data_item.num_fut_valid_agn = torch.sum(torch.from_numpy(fut_valid_agn_mask))
                data_item.gdp_label = data_item.fut[fut_valid_agn_mask, :, :].float()
                data_item.goals     = data_item.fut[agn_mask,-1,:2].float()
                data_item.agent_fut_valid = data_item.fut_valid[agn_mask]
                # data_item.goals[fut_valid_agn_mask] = data_item.agn_goal_set
        elif self.dec_type in  ['test']:
                data_item.num_goal_valid_agn = torch.sum(torch.from_numpy(agn_mask))
                data_item.agent_fut_valid = data_item.fut_valid[agn_mask]                
                data_item.object_id = data_item.node_name[agn_mask]
        return data_item
    
# def construct_dec_graph

def flat_Batch(data_batch):
    data_batch.flat_node_name = np.concatenate(data_batch.node_name).flatten()
    data_batch.flat_node_type = np.concatenate(data_batch.node_type).flatten()
    data_batch.flat_agent_node_type = np.concatenate(data_batch.agent_node_type).flatten()
    data_batch.flat_edge_name = np.concatenate(data_batch.edge_name).flatten()
    data_batch.flat_edge_type = np.concatenate(data_batch.edge_type).flatten()
    return data_batch

def retrieve_mask(type_list, wanted_type_set):
    return np.in1d(type_list, wanted_type_set)


if __name__ == '__main__':
    dataset = SimAgn_Dataset(data_path='/disk2/SimAgent_Dataset/pyg_data_may17/validation', dec_type='test') 
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
    print(dataset.__len__())

    for d_idx, d in enumerate(loader):
        # tic = time.time()
        # flat_d = flat_Batch(d)
        # sdc_mask = retrieve_mask(type_list=flat_d.flat_node_type, wanted_type_set=('sdcAg'))
        
        # if sum(sdc_mask) != loader.batch_size: #  , f'{sum(sdc_mask)} {loader.batch_size}'
        #     print(f'{sum(sdc_mask)} {loader.batch_size}', flat_d.flat_node_type[:3])
        #     print(flat_d.flat_edge_type[:3])
        #     break
        print(d_idx)
        # tac = time.time()        