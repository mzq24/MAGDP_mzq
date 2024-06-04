import select
import torch
from torch_geometric.nn import GATConv
from dataset_may17 import retrieve_mask
from model_mzq.base_enc_mzq import Sim_Base_Enc
from model_mzq.gat_encoder_layer import GATEncoderLayer
from scipy.sparse import block_diag

import numpy as np
import copy


class Sim_Enc(Sim_Base_Enc):
    """ Encode the whole scene and return (agn_Int, agn_Dyn) """
    def __init__(self, args):
        super(Sim_Enc, self).__init__(args)
        self.args = args
        self.sigmoid = torch.nn.Sigmoid()
        
        #self.gnn_conv = GATConv
        self.init_GNNConv()
        
        self.agn_Dyn_emb  = torch.nn.Linear(args['enc_hidden_size'], self.args['enc_embed_size'], bias=True)
        self.agn_CCL_emb  = torch.nn.Linear(args['enc_hidden_size'], self.args['enc_embed_size'], bias=True)
        self.agn_Int_emb  = torch.nn.Linear(self.args['enc_gat_size']*self.args['num_gat_head'], self.args['enc_embed_size'], bias=True)
        self.agn_CCLs_emb = torch.nn.Linear(self.args['enc_gat_size']*self.args['num_gat_head'], self.args['enc_embed_size'], bias=True)

        self.CCL_to_Agn_GAT_1  = GATConv(self.args['enc_hidden_size'], self.args['enc_gat_size'], 
                                        heads=self.args['num_gat_head'], concat=True, 
                                        add_self_loops=False,
                                        dropout=0.0)
        
        self.Agn_to_Agn_GAT_1  = GATConv(self.args['enc_gat_size']*self.args['num_gat_head'], self.args['enc_gat_size'], 
                                        heads=self.args['num_gat_head'], concat=True,  
                                        add_self_loops=False,
                                        dropout=0.0)
        
        ## Layer Norms
        self.norm2 = torch.nn.LayerNorm(self.args['enc_embed_size'])

    def forward(self, pyg_data_fwd, agn_cnl=False, use_gsl=False):
        """
        param:
            pyg_data_fwd: a batch of data
            agn_cnl: both agent and centerline are encoded if True, otherwise only agent is encoded
        output:
            return agn_Int, agn_Dyn, ccl_Seq, tcl_Seq(useless for now)
        """

        # Agent & CCL Encoding
        agn_cnl_enc = self.Agn_Cnl_Encoder(pyg_data_fwd.node_feature, pyg_data_fwd.attr, pyg_data_fwd.flat_node_type, use_attention=self.args['use_attention']) # )
        raw_agn_cnl_enc = agn_cnl_enc.clone()

        # get features according to dec_type
        agn_feature = self.get_agn_feature(raw_agn_cnl_enc, pyg_data_fwd)
        # agn_all_feature = self.get_agn_feature(raw_agn_cnl_enc, pyg_data_fwd, types='MGAall_m')
        ccl_feature = self.get_ccl_feature(raw_agn_cnl_enc, pyg_data_fwd)

        # dynamic features and tcl_seq for output
        agn_Dyn = self.leaky_relu(self.agn_Dyn_emb(self.leaky_relu(agn_feature)))
        TCL_Seq = self.leaky_relu(self.agn_CCL_emb(self.leaky_relu(ccl_feature)))

        # get edge mask
        # edge_mask_A2A = self.get_edge_mask(pyg_data_fwd.flat_edge_type, wanted_type=0)
        # edge_mask_C2A = self.get_edge_mask(pyg_data_fwd.flat_edge_type, wanted_type=1)

        # edges = pyg_data_fwd.edge_index     # 2 x E

        # use gsl to get learnable edges
        if use_gsl:
            if not agn_cnl:
                agn_mask = retrieve_mask(type_list=pyg_data_fwd.flat_node_type, wanted_type_set=('sdcAg', 'tarAg', 'nbrAg'))
                if not hasattr(pyg_data_fwd, 'batch_adj_mask'):
                    pyg_data_fwd.batch_adj_mask = torch.from_numpy(block_diag(pyg_data_fwd.adj_mask).toarray().astype(bool))
                learnable_edges_feature_agent = raw_agn_cnl_enc[agn_mask, :]
                pred_edge = self.Edge_Pred(learnable_edges_feature_agent, pyg_data_fwd.batch_adj_mask, \
                                           self.args['threshold'])    # batch_adj_mask is a block diagonal matrix
            else:
                pass
                # both agent and centerline are encoded
                # selected_edges_A2A = edges[:, edge_mask_A2A]
                # selected_edges_C2A = edges[:, edge_mask_C2A]

        # get different distance threshold from dataset 
        diff_dist = False
        if 'dist_threshold' in self.args:
            assert not use_gsl, "use_gsl is True"
            diff_dist = True
            
        # G1 Encoding
        agn_cnl_enc_ca  = self.C_A(raw_agn_cnl_enc, pyg_data_fwd.edge_index, pyg_data_fwd.flat_edge_type)
        agn_CCLs = self.leaky_relu(self.agn_CCLs_emb(self.leaky_relu(self.get_agn_feature(agn_cnl_enc_ca, pyg_data_fwd))))
            
        # G2 Encoding, whether use gsl (gnn structure learning) or not
        if use_gsl:
            agn_enc_ca = agn_cnl_enc_ca[agn_mask, :]    # fetch agent features to align with the pred_edge
            agn_enc_aa_all  = self.A_A(agn_enc_ca, pyg_data_fwd.edge_index, pyg_data_fwd.flat_edge_type, edge_index=pred_edge)
             # get mask in only agent features, not all (including ccl features)
            goal_valid_mask_in_agn = pyg_data_fwd.goal_valid[agn_mask]
            flat_sdc_tar_mask_in_agn = retrieve_mask(type_list=pyg_data_fwd.flat_node_type[agn_mask], wanted_type_set=('sdcAg', 'tarAg'))
            ang_enc_aa_sdc_tar = agn_enc_aa_all[np.logical_and(flat_sdc_tar_mask_in_agn, goal_valid_mask_in_agn.cpu().numpy())].float()
            agn_Int  = self.leaky_relu(self.agn_Int_emb(self.leaky_relu(ang_enc_aa_sdc_tar)))
        else:
            if not diff_dist:
                # normal G2 encoding
                agn_cnl_enc_aa_all = self.A_A(agn_cnl_enc_ca, pyg_data_fwd.edge_index, pyg_data_fwd.flat_edge_type)
                agn_Int = self.leaky_relu(self.agn_Int_emb(self.leaky_relu(self.get_agn_feature(agn_cnl_enc_aa_all,  pyg_data_fwd))))   
            else:
                # distance threshold G2 encoding
                dist_edge_index = self.get_a2a_edge_dist_thr(pyg_data_fwd)
                agn_mask = retrieve_mask(type_list=pyg_data_fwd.flat_node_type, wanted_type_set=('sdcAg', 'tarAg', 'nbrAg'))
                agn_enc_ca = agn_cnl_enc_ca[agn_mask, :]
                agn_enc_aa_all = self.A_A(agn_enc_ca, pyg_data_fwd.edge_index, pyg_data_fwd.flat_edge_type, edge_index=dist_edge_index)
                goal_valid_mask_in_agn = pyg_data_fwd.goal_valid[agn_mask]
                flat_sdc_tar_mask_in_agn = retrieve_mask(type_list=pyg_data_fwd.flat_node_type[agn_mask], wanted_type_set=('sdcAg', 'tarAg'))
                ang_enc_aa_sdc_tar = agn_enc_aa_all[np.logical_and(flat_sdc_tar_mask_in_agn, goal_valid_mask_in_agn.cpu().numpy())].float()
                # agn_Int  = self.leaky_relu(self.agn_Int_emb(self.leaky_relu(ang_enc_aa_sdc_tar)))
                agn_Int = self.leaky_relu(self.agn_Int_emb(self.leaky_relu(ang_enc_aa_sdc_tar)))

        # check the dimension
        assert agn_Dyn.shape[0] == agn_CCLs.shape[0] == agn_Int.shape[0] == TCL_Seq.shape[0], "The first dimensions are not equal, not aligned!"

        ## Layer Norm 一下
        agn_Dyn  = self.norm2(agn_Dyn)
        agn_CCLs = self.norm2(agn_CCLs)
        agn_Int  = self.norm2(agn_Int)
        TCL_Seq  = self.norm2(TCL_Seq)
        
        # if self.args['dec_type'] in ['MGAsdc', 'MGA', 'DGE', 'testMGA', 'MGAall']:
        #     return torch.cat((agn_Dyn, agn_CCLs, agn_Int), dim=1)
        # elif self.args['dec_type'] in ['TGE', 'testTGE', 'TGEall']:
        #     return torch.cat((agn_Dyn, agn_CCLs, agn_Int, TCL_Seq), dim=1)
        # elif self.args['dec_type'] in ['GDP', 'testGDP', 'GDPall']:
        
        return agn_Dyn, agn_CCLs, agn_Int, TCL_Seq

    def init_GNNConv(self):
        if self.args['gnn_conv'] == 'GATv2Conv':
            from torch_geometric.nn import GATv2Conv as GNNConv
        elif self.args['gnn_conv'] == 'GATConv':
            from torch_geometric.nn import GATConv as GNNConv
        elif self.args['gnn_conv'] == 'TransformerConv':
            from torch_geometric.nn import TransformerConv as GNNConv
        self.gnn_conv = GNNConv

    def get_agn_feature(self, agn_cnl_enc, pyg_data, types=None):
        if types is not None:
            selected_type = types
        else:
            selected_type = self.args['dec_type']

        if selected_type in ['MGA', 'TGE', 'DGE']:
            sdc_tar_mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('sdcAg', 'tarAg'))
            return agn_cnl_enc[np.logical_and(sdc_tar_mask, pyg_data.goal_valid.cpu().numpy())].float()
        elif selected_type in ['MGAall_m']:
            agn_mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('sdcAg', 'tarAg', 'nbrAg'))
            return agn_cnl_enc[agn_mask].float()
        elif selected_type in ['MGAall', 'TGEall']:
            agn_mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('sdcAg', 'tarAg', 'nbrAg'))
            return agn_cnl_enc[np.logical_and(agn_mask, pyg_data.goal_valid.cpu().numpy())].float()
        elif selected_type in ['GDPall']:
            agn_mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('sdcAg', 'tarAg', 'nbrAg'))
            return agn_cnl_enc[np.logical_and(agn_mask, pyg_data.fut_valid.cpu().numpy())].float()
        elif selected_type in ['MGAsdc']:
            sdc_mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('sdcAg'))
            return agn_cnl_enc[np.logical_and(sdc_mask, pyg_data.goal_valid.cpu().numpy())].float()
        elif selected_type in ['Mot']:
            tar_mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('tarAg'))
            return agn_cnl_enc[np.logical_and(tar_mask, pyg_data.fut_valid.cpu().numpy())].float()
        elif selected_type in ['GDP', 'GDPrnn']:
            sdc_tar_mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('sdcAg', 'tarAg'))
            return agn_cnl_enc[np.logical_and(sdc_tar_mask, pyg_data.fut_valid.cpu().numpy())].float()
        elif selected_type in ['arGDP', 'testarGDP']:
            return agn_cnl_enc.float()
        elif selected_type in ['testTGE', 'testMGA', 'testGDP', 'arGDPmax', 'testarGDPmax']:
            agn_mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('sdcAg', 'tarAg', 'nbrAg'))
            # print(f'agn_cnl_enc {agn_cnl_enc.shape}, sdc_tar_mask {sdc_tar_mask.shape}')
            return agn_cnl_enc[agn_mask].float()
            # agn_mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('sdcAg', 'tarAg', 'nbrAg'))
            # return agn_cnl_enc[agn_mask].float()
        
    def get_agn_feature_mzq(self, agn_cnl_enc, pyg_data):
        dec_type = self.args['dec_type']
        valid_data = pyg_data.goal_valid if dec_type in ['MGA', 'TGE', 'DGE', 'MGAall', 'TGEall', 'MGAsdc'] else pyg_data.fut_valid
        valid_data = valid_data.cpu().numpy()

        if dec_type in ['MGA', 'TGE', 'DGE', 'GDP', 'GDPrnn']:
            mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('sdcAg', 'tarAg'))
        elif dec_type in ['MGAall', 'TGEall', 'GDPall', 'testTGE', 'testMGA', 'testGDP', 'arGDPmax', 'testarGDPmax']:
            mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('sdcAg', 'tarAg', 'nbrAg'))
        elif dec_type in ['MGAsdc']:
            mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('sdcAg'))
        elif dec_type in ['Mot']:
            mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('tarAg'))
        elif dec_type in ['arGDP', 'testarGDP']:
            return agn_cnl_enc.float()

        return agn_cnl_enc[np.logical_and(mask, valid_data)].float()

    def get_ccl_feature(self, agn_cnl_enc, pyg_data):
        if self.args['dec_type'] in ['MGA', 'TGE', 'DGE']:
            tcl_mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('sdcTCL', 'sdcTCL_FAKE', 
                                                                                         'tarTCL', 'tarTCL_FAKE'))
            # print('tcl_mask', sum(tcl_mask))
            # print(set(pyg_data.flat_node_type))
            return agn_cnl_enc[np.logical_and(tcl_mask, pyg_data.goal_valid.cpu().numpy())].float()
        #  'MGAall', 'TGEall', 'GDPall'
        elif self.args['dec_type'] in ['MGAall', 'TGEall']:
            tcl_mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('sdcTCL', 'tarTCL', 'nbrTCL',
                                                                                         'sdcTCL_FAKE', 'tarTCL_FAKE', 'nbrTCL_FAKE'))
            return agn_cnl_enc[np.logical_and(tcl_mask, pyg_data.goal_valid.cpu().numpy())].float()
        elif self.args['dec_type'] in ['GDPall']:
            tcl_mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('sdcTCL', 'tarTCL', 'nbrTCL',
                                                                                         'sdcTCL_FAKE', 'tarTCL_FAKE', 'nbrTCL_FAKE'))
            return agn_cnl_enc[np.logical_and(tcl_mask, pyg_data.fut_valid.cpu().numpy())].float()
        elif self.args['dec_type'] in ['MGAsdc']:
            tcl_mask = retrieve_mask(pyg_data.flat_node_type, 
                                    ('sdcTCL', 'sdcTCL_FAKE'))
            return agn_cnl_enc[np.logical_and(tcl_mask, pyg_data.goal_valid.cpu().numpy())].float()
        elif self.args['dec_type'] in ['GDP', 'GDPrnn']:
            # print('GDP\n')
            tcl_mask = retrieve_mask(pyg_data.flat_node_type, 
                                    ('sdcTCL', 'sdcTCL_FAKE', 
                                     'tarTCL', 'tarTCL_FAKE'))
            # print(sum(tcl_mask))
            return agn_cnl_enc[np.logical_and(tcl_mask, pyg_data.fut_valid.cpu().numpy())].float()
        elif self.args['dec_type'] in ['Mot']:
            tcl_mask = retrieve_mask(pyg_data.flat_node_type, 
                                    ('tarTCL', 'tarTCL_FAKE'))
            return agn_cnl_enc[np.logical_and(tcl_mask, pyg_data.fut_valid.cpu().numpy())].float()
        elif self.args['dec_type'] in ['arGDP', 'testarGDP']:
            return agn_cnl_enc.float()
        elif self.args['dec_type'] in ['testMGA', 'testTGE', 'testGDP', 'arGDPmax', 'testarGDPmax']:
            tcl_mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('sdcTCL', 'tarTCL', 'nbrTCL',
                                                                                         'sdcTCL_FAKE', 'tarTCL_FAKE', 'nbrTCL_FAKE',))
            return agn_cnl_enc[tcl_mask].float()

    def get_ccl_feature_mzq(self, agn_cnl_enc, pyg_data):
        dec_type = self.args['dec_type']
        valid_data = pyg_data.goal_valid if dec_type not in ['GDPall', 'GDP', 'GDPrnn', 'Mot'] else pyg_data.fut_valid
        valid_data = valid_data.cpu().numpy()

        if dec_type in ['MGA', 'TGE', 'DGE', 'MGAsdc']:
            tcl_mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('sdcTCL', 'sdcTCL_FAKE', 
                                                                                        'tarTCL', 'tarTCL_FAKE'))
        elif dec_type in ['MGAall', 'TGEall', 'GDPall', 'testMGA', 'testTGE', 'testGDP', 'arGDPmax', 'testarGDPmax']:
            tcl_mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('sdcTCL', 'tarTCL', 'nbrTCL',
                                                                                        'sdcTCL_FAKE', 'tarTCL_FAKE', 'nbrTCL_FAKE'))
        elif dec_type in ['GDP', 'GDPrnn']:
            tcl_mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('sdcTCL', 'sdcTCL_FAKE', 
                                                                                        'tarTCL', 'tarTCL_FAKE'))
        elif dec_type in ['Mot']:
            tcl_mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('tarTCL', 'tarTCL_FAKE'))
        elif dec_type in ['arGDP', 'testarGDP']:
            return agn_cnl_enc.float()

        return agn_cnl_enc[np.logical_and(tcl_mask, valid_data)].float()

    def get_goal_valid_agn_tcl_feature(self, agn_cnl_enc, pyg_data, valid_agn_mask):
        valid_agent_names = pyg_data.flat_node_name[valid_agn_mask.cpu()]
        tcl_mask = retrieve_mask(pyg_data.flat_node_type, 
                                ('sdcTCL', 'tarTCL', 'nbrTCL', 
                                 'sdcTCL_FAKE', 'tarTCL_FAKE', 'nbrTCL_FAKE'))
        all_agent_TCL_names = pyg_data.flat_node_name[tcl_mask] # f{agent_id}TCL{number}
        valid_agent_TCL_names = [tcl_name for tcl_name in all_agent_TCL_names if tcl_name.split('TCL')[0] in valid_agent_names]
        valid_agent_TCL_mask  = [True  if node_name in valid_agent_TCL_names else False for node_name in pyg_data.flat_node_name]
        # print(f'valid_agent_names {len(valid_agent_names)}, valid_agent_TCL_names {len(valid_agent_TCL_names)}')
        return agn_cnl_enc[valid_agent_TCL_mask]

    # G1
    def C_A(self, CLA_feat, Edges, edge_type_list_ca):
        ''' 如果没有 centerline loop 的话，一次 gather 后centerline 的feature就为0了 '''
        edge_mask_in_C_A = retrieve_mask(edge_type_list_ca,  wanted_type_set=(  'sdcTCL-sdcAg', 'tarTCL-tarAg', 'nbrTCL-nbrAg',
                                                                                'sdcCCL-sdcAg', 'tarCCL-tarAg', 'nbrCCL-nbrAg',
                                                                                'sdcAg-Loop',   'tarAg-Loop',   'nbrAg-Loop',
                                                                                'sdcCCL-Loop',  'tarCCL-Loop',  'nbrCCL-Loop',
                                                                                'sdcTCL-Loop',  'tarTCL-Loop',  'nbrTCL-Loop' ))
        # print(sum(edge_mask_in_C_A))
        c_a_gat_feature = self.leaky_relu(self.CCL_to_Agn_GAT_1(CLA_feat, Edges[:,edge_mask_in_C_A]))
        return c_a_gat_feature

    def C_A_mzq(self, CLA_feat, Edges, edge_type_list_ca):
        wanted_mask = self.get_edge_type_set(wanted_type=1)
        edge_mask_in_C_A = self.get_edge_mask(edge_type_list_ca, wanted_mask)
        
        c_a_gat_feature = self.leaky_relu(self.CCL_to_Agn_GAT_1(CLA_feat, Edges[:,edge_mask_in_C_A]))
        return c_a_gat_feature
    
    # G2
    def A_A(self, CLA_feat, Edges, edge_type_list_aa, edge_index=None):
        ''' 如果没有 centerline loop 的话，一次 gather 后centerline 的feature就为0了 '''
        if edge_index is None:
            edge_mask_in_A_A = retrieve_mask(edge_type_list_aa,  wanted_type_set=('sdcAg-tarAg', 'sdcAg-nbrAg', 
                                                                              'tarAg-tarAg', 'tarAg-nbrAg', 'tarAg-sdcAg', 
                                                                              'nbrAg-nbrAg', 'nbrAg-tarAg', 'nbrAg-sdcAg',
                                                                              'sdcAg-Loop',  'tarAg-Loop',  'nbrAg-Loop',
                                                                              'sdcCCL-Loop',  'tarCCL-Loop',  'nbrCCL-Loop',
                                                                              'sdcTCL-Loop',  'tarTCL-Loop',  'nbrTCL-Loop'))
            edge_index = Edges[:,edge_mask_in_A_A]
        else:
            edge_index = edge_index
        a_a_gat_feature = self.leaky_relu( self.Agn_to_Agn_GAT_1(CLA_feat, edge_index))
        return a_a_gat_feature

    def A_A_mzq(self, CLA_feat, Edges, edge_type_list_aa, edge_index=None):
        if edge_index is None:
            wanted_mask = self.get_edge_type_set(wanted_type=0)
            edge_mask_in_A_A = self.get_edge_mask(edge_type_list_aa, wanted_mask)
            edge_index = Edges[:,edge_mask_in_A_A]
        else:
            edge_index = edge_index
        a_a_gat_feature = self.leaky_relu( self.Agn_to_Agn_GAT_1(CLA_feat, edge_index))
        return a_a_gat_feature

    # get agn2agn edges from distance threshold
    def get_a2a_edge_dist_thr(self, pyg_data_fwd, dist_threshold=None):
        dist_thr = dist_threshold if dist_threshold is not None else self.args['dist_threshold']
        agn_mask = retrieve_mask(type_list=pyg_data_fwd.flat_node_type, wanted_type_set=('sdcAg', 'tarAg', 'nbrAg'))
        agn_x = pyg_data_fwd.node_feature[agn_mask, 10, :2].float()
        # given positions of N agents, build an adjacent matrix according to their distance with a threshold
        mutual_dist = torch.cdist(agn_x, agn_x, p=2)
        dist_adj = mutual_dist < dist_thr
        if not hasattr(pyg_data_fwd, 'batch_adj_mask'):
            pyg_data_fwd.batch_adj_mask = torch.from_numpy(block_diag(pyg_data_fwd.adj_mask).toarray().astype(bool)).to(dist_adj.device)
        masked_dist_adj = dist_adj & pyg_data_fwd.batch_adj_mask
        dist_edge_index = masked_dist_adj.nonzero(as_tuple=False).t()
        return dist_edge_index

    def get_edge_type_set(self, wanted_type=0):
        # 0: A2A, 1: C2A, 2: only agent2agent
        if wanted_type == 0:
            return set(['sdcAg-tarAg', 'sdcAg-nbrAg', 
                    'tarAg-tarAg', 'tarAg-nbrAg', 'tarAg-sdcAg', 
                    'nbrAg-nbrAg', 'nbrAg-tarAg', 'nbrAg-sdcAg',
                    'sdcAg-Loop',  'tarAg-Loop',  'nbrAg-Loop',
                    'sdcCCL-Loop',  'tarCCL-Loop',  'nbrCCL-Loop',
                    'sdcTCL-Loop',  'tarTCL-Loop',  'nbrTCL-Loop'])
        elif wanted_type == 1:
            return set(['sdcTCL-sdcAg', 'tarTCL-tarAg', 'nbrTCL-nbrAg',
                    'sdcCCL-sdcAg', 'tarCCL-tarAg', 'nbrCCL-nbrAg',
                    'sdcAg-Loop',   'tarAg-Loop',   'nbrAg-Loop',
                    'sdcCCL-Loop',  'tarCCL-Loop',  'nbrCCL-Loop',
                    'sdcTCL-Loop',  'tarTCL-Loop',  'nbrTCL-Loop'])
        elif wanted_type == 2:
            return set(['sdcAg-tarAg', 'sdcAg-nbrAg', 
                    'tarAg-tarAg', 'tarAg-nbrAg', 'tarAg-sdcAg', 
                    'nbrAg-nbrAg', 'nbrAg-tarAg', 'nbrAg-sdcAg',
                    'sdcAg-Loop',  'tarAg-Loop',  'nbrAg-Loop'])

    def get_edge_mask(self, edge_type_list, type_set):  
        # retrieve_mask(type_list=edge_type_list, wanted_type_set=type_set)
        if isinstance(type_set, set):
            type_set = tuple(type_set)
        return np.in1d(edge_type_list, type_set) 

    def GNN_Block(self, num_gnn_layers=1):
        gnn_convs = torch.nn.ModuleList()
        for _ in range(num_gnn_layers):
            # gat_hidden_channels = self.args['enc_gat_size'] // self.args['num_gat_head']
            gat_hidden_channels = self.args['enc_hidden_size'] // self.args['num_gat_head']

            assert self.args['enc_gat_size'] % self.args['num_gat_head'] ==0

            conv = self.gnn_conv(self.args['enc_hidden_size'], gat_hidden_channels, heads=self.args['num_gat_head'], concat=True, 
                                 add_self_loops=True, # edge_dim=4, # beta=True, 
                                 dropout=0.0)
            gnn_convs.append(GATEncoderLayer(conv, dropout=0.0))

        return gnn_convs

    def set_up_GNN_Block(self):
        # gat_hidden_channels = self.args['enc_gat_size'] // self.args['num_gat_head']
        gat_hidden_channels = self.args['enc_hidden_size'] // self.args['num_gat_head']
        conv = self.gnn_conv(self.args['enc_hidden_size'], gat_hidden_channels, heads=self.args['num_gat_head'], concat=True, 
                                 add_self_loops=True, # edge_dim=4, # beta=True, 
                                 dropout=0.0)
        return GATEncoderLayer(conv, dropout=0.0)

    def learnable_edge_GNN_Encoder(self, x, edges, gnn_convs):
        for conv in gnn_convs:
            x = self.leaky_relu(conv(x, edges))
        return x

    def Edge_Pred(self, z, adj_mask, threshold=0.3):
 
        prob_adj = self.sigmoid(torch.matmul(z , z.t()) / torch.sqrt(torch.tensor(self.args['enc_hidden_size'], dtype=torch.float32)))

        prob_adj[~adj_mask]=0

        return (prob_adj > threshold).nonzero(as_tuple=False).t()

    
