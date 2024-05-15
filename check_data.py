import torch
# from dataset_may17 import SimAgn_Dataset
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
# sys.path.append('../may07')

# from vis_utils  import scatter_goal
# from test_utils import get_agn_tcl_index, get_agn_names, get_batch_tcl_indexes, set_TCL_of_an_agent, get_agents_with_many_CCLs

# dataset = SimAgn_Dataset(data_path='/disk2/SimAgent_Dataset/pyg_data_full_may17/training', dec_type='GDP') 
# loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
# print(dataset.__len__())

# for d_idx, d in enumerate(loader):
#     tic = time.time()
#     # print(d)
#     flat_d = flat_Batch(d)
#     if d_idx>30:
#         print(d_idx)
#     print(flat_d.node_feature[0,:3])
#     print(flat_d.fut[0,:3])
#     print([g.shape for g in flat_d.gdp_label])
from data_utils import normlize_Agn_seq
from dataset_may17 import retrieve_mask
data_name = '/disk2/SimAgent_Dataset/pyg_data_full_may17/validation/'+'2-scn-8c07ce6e0b79d210.pyg'
data_item = torch.load(data_name)
print(data_item)

agn_mask = retrieve_mask(type_list=data_item.node_type, wanted_type_set=('sdcAg', 'tarAg', 'nbrAg'))
agn_Seqs = data_item.node_feature[agn_mask]
# print(f'agn_Seqs {agn_Seqs.shape}')