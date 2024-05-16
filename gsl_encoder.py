import torch
from torch import nn
import sys 
import numpy as np
import time

from gat_encoder_layer import GATEncoderLayer

from utils.data_utils import convert_goalsBatch_to_goalsets_for_scenarios
import torch.nn.functional as F

from gsl_dataset import flat_Batch, retrieve_mask

from polyline_encoder import PointNetPolylineEncoder
from attention import TransformerCombiner
from seq_encoder import Seq_Enc_Net
sys.path.append('../')
class GSL_Enc_Net(torch.nn.Module):
    def __init__(self, args):
        super(GSL_Enc_Net, self).__init__()
        self.args = args
        self.seq_encoder = Seq_Enc_Net(self.args)
        
        if not self.args['dec_feat'] == 'D':
            self.init_GNNConv()
            self.a2a_convs = self.init_GNN_Block(num_gnn_layers=self.args['num_A2A_G_layers'])
            self.l2l_convs = self.init_GNN_Block(2)
            self.l2a_convs = self.init_GNN_Block(num_gnn_layers=self.args['num_A2A_G_layers'])
            self.a2l_convs = self.init_GNN_Block(num_gnn_layers=self.args['num_A2A_G_layers'])

        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)

        self.num_input_feat = 1
        # if self.args['feat_attention']:
        #     self.att_combiner= TransformerCombiner(self.num_input_feat*self.args['enc_embed_size'], num_head=1, num_in=self.num_input_feat, num_layers=1)

        self.out_dim = self.args['agn_output_dim']

    def init_GNNConv(self):
        if self.args['gnn_conv'] == 'GATv2Conv':
            from torch_geometric.nn import GATv2Conv as GNNConv
        elif self.args['gnn_conv'] == 'GATConv':
            from torch_geometric.nn import GATConv as GNNConv
        elif self.args['gnn_conv'] == 'TransformerConv':
            from torch_geometric.nn import TransformerConv as GNNConv
        self.gnn_conv = GNNConv

    def init_GNN_Block(self, num_gnn_layers=1):
        gnn_convs = torch.nn.ModuleList()
        for _ in range(num_gnn_layers):
            gat_hidden_channels = self.args['enc_gat_size'] // self.args['num_gat_head']
            assert self.args['enc_gat_size'] % self.args['num_gat_head'] ==0

            conv = self.gnn_conv(self.args['enc_gat_size'], gat_hidden_channels, 
                                    heads=self.args['num_gat_head'], concat=True,  
                                    add_self_loops=False,
                                    # edge_dim=4,
                                    # beta=True,
                                    dropout=0.0)
            gnn_convs.append(GATEncoderLayer(conv, dropout=0.0))
            # gnn_convs.append(conv)
        return gnn_convs
    
    def GNN_Encoder(self, x, edges):
        for conv in self.a2a_convs:
            x = self.leaky_relu( conv(x, edges) )
        return x
    
    def A2L_Encoder(self, x, edges):
        for conv in self.a2l_convs:
            x = self.leaky_relu( conv(x, edges) )
        return x
    
    def L2L_Encoder(self, x_L, edges):
        for conv in self.l2l_convs:
            x_L = self.leaky_relu( conv(x_L, edges) )
        return x_L
    
    def L2A_Encoder(self, x, edges):
        for conv in self.l2a_convs:
            x_L = self.leaky_relu( conv(x, edges) )
        return x_L

    def Edge_Decoder(self, z, adj_mask):
        # prob_adj = self.sigmoid(z @ z.t() )
        ## batch 越大相当于 序列越长
        assert z.shape[-1]==self.args['enc_hidden_size']
        prob_adj = self.sigmoid(torch.matmul(z , z.t()) / torch.sqrt(torch.tensor(self.args['enc_hidden_size'], dtype=torch.float32)))
        adj_mask=torch.from_numpy(adj_mask)
        prob_adj[~adj_mask]=0
        return (prob_adj > 0.3).nonzero(as_tuple=False).t()

    def forward(self, pyg_data_fwd):
        """ return agn_Int, agn_Dyn, ccl_Seq """
        seq_tic = time.time()
        agn_seq, lan_enc = self.seq_encoder(pyg_data_fwd)
        if self.args['dec_feat'] == 'D':
            sdc_tar_mask = retrieve_mask(pyg_data_fwd.flat_node_type[pyg_data_fwd.flat_agn_mask], wanted_type_set=('sdcAg', 'tarAg'))
            fut_valid_mask = pyg_data_fwd.flat_fut_valid[pyg_data_fwd.flat_agn_mask]
            fut_valid_sdc_tar_mask = torch.logical_and(torch.from_numpy(sdc_tar_mask).to(agn_seq.device), fut_valid_mask)

            # sim_agn_mask = sdc_mask
            sim_agn_mask = fut_valid_sdc_tar_mask
            return agn_seq[sim_agn_mask]
        raw_agn_enc = agn_seq.clone()
        seq_tac = time.time()
        # print('{} sec to get seq enc'.format(seq_tac-seq_tic))

        node_feat_tic = time.time()
        ## 准备 Node Feature
        Agn_Lan_Seq_Feature = torch.empty(size=(agn_seq.shape[0] + lan_enc.shape[0], agn_seq.shape[1]), 
                                          dtype=torch.float32, device=agn_seq.device)
        Agn_Lan_Seq_Feature[pyg_data_fwd.flat_agn_mask] = agn_seq
        Agn_Lan_Seq_Feature[~pyg_data_fwd.flat_agn_mask] = lan_enc

        node_feat_tac = time.time()
        # print('{} sec to get node feature'.format(node_feat_tac-node_feat_tic))

        
        
        ## 1. A2L for traffic
        A2L_edge_mask = retrieve_mask(pyg_data_fwd.flat_edge_type, wanted_type_set=('curAL')) ## 需要加上 self loops
        X = self.A2L_Encoder(Agn_Lan_Seq_Feature, pyg_data_fwd.edge_index[:,A2L_edge_mask])
        # print()
        # print(sum(A2L_edge_mask) ) #, A2L_edge_mask)

        ## 2. L2L
        # X = Agn_Lan_Seq_Feature
        L2L_edge_mask = retrieve_mask(pyg_data_fwd.flat_edge_type, wanted_type_set=('self_lane', 'exit_lane', 'entry_lane', 'left_lane', 'right_lane'))
        X = self.L2L_Encoder(X, pyg_data_fwd.edge_index[:,L2L_edge_mask])
        agn_seq_2 = X.clone()
        # print(sum(l2l_edge_mask)) #, l2l_edge_mask)

        ## 3. L2A
        L2A_edge_mask = retrieve_mask(pyg_data_fwd.flat_edge_type, wanted_type_set=('curLA')) ## 需要加上 self loops
        X = self.L2A_Encoder(X, pyg_data_fwd.edge_index[:,L2A_edge_mask])
        agn_cur_lan = X.clone()
        # print(sum(L2A_edge_mask)) #, l2l_edge_mask)

        gsl_tic = time.time()
        ## 4. GSL + A2A ## 用什么信息来构建 Interaction Graph 值得研究一下，agn_seq only 还是 traffic-aware
        edge_pred   = self.Edge_Decoder(X[pyg_data_fwd.flat_agn_mask], pyg_data_fwd.batch_adj_mask)
        agn_int     = self.GNN_Encoder( X[pyg_data_fwd.flat_agn_mask], edge_pred)
        # print(edge_pred.shape, sum(pyg_data_fwd.flat_agn_mask)**2)
        gsl_tac = time.time()
        # print('{} sec to get GSL outputs'.format(gsl_tac-gsl_tic))

        gnn_feat_tac = time.time()
        # print('{} sec to get gnn outputs'.format(gnn_feat_tac-node_feat_tac))

        mask_tic = time.time()
        # sdc_mask = retrieve_mask(pyg_data_fwd.flat_node_type[pyg_data_fwd.flat_agn_mask], wanted_type_set=('sdcAg'))
        sdc_tar_mask = retrieve_mask(pyg_data_fwd.flat_node_type[pyg_data_fwd.flat_agn_mask], wanted_type_set=('sdcAg', 'tarAg'))
        fut_valid_mask = pyg_data_fwd.flat_fut_valid[pyg_data_fwd.flat_agn_mask]
        fut_valid_sdc_tar_mask = torch.logical_and(torch.from_numpy(sdc_tar_mask).to(agn_seq.device), fut_valid_mask)

        # sim_agn_mask = sdc_mask
        sim_agn_mask = fut_valid_sdc_tar_mask
        
        mask_tac = time.time()
        # print('{} sec to get sdc mask in agn'.format(mask_tac-mask_tic))

        enc_feat_list = [   raw_agn_enc[sim_agn_mask],
                            agn_int[sim_agn_mask], 
                            agn_cur_lan[pyg_data_fwd.flat_fut_valid_sdc_tar_mask],
                            agn_seq_2[  pyg_data_fwd.flat_fut_valid_sdc_tar_mask],
                            ]

        # if self.args['feat_attention']:
        #     enc = self.att_combiner(enc_feat_list)
        # else:
        #     enc = torch.cat([feat for feat in enc_feat_list], dim=1)
        enc = torch.cat([feat for feat in enc_feat_list], dim=1)
        return enc # + agn_tar_lan[pyg_data_fwd.flat_fut_valid_sdc_tar_mask]




