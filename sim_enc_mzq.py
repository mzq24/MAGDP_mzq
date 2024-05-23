import torch
from torch_geometric.nn import GATConv
from dataset_may17 import retrieve_mask
from base_enc_mzq import Sim_Base_Enc
import numpy as np
import copy
class Sim_Enc(Sim_Base_Enc):
    """ Encode the whole scene and return (agn_Int, agn_Dyn) """
    def __init__(self, args):
        super(Sim_Enc, self).__init__(args)
        self.args = args
    
        self.agn_Dyn_emb  = torch.nn.Linear(args['enc_hidden_size'], self.args['enc_embed_size'], bias=True)
        self.agn_CCL_emb  = torch.nn.Linear(args['enc_hidden_size'], self.args['enc_embed_size'], bias=True)
        self.agn_Int_emb  = torch.nn.Linear(self.args['enc_gat_size']*self.args['num_gat_head'], self.args['enc_embed_size'], bias=True)
        self.agn_CCLs_emb = torch.nn.Linear(self.args['enc_gat_size']*self.args['num_gat_head'], self.args['enc_embed_size'], bias=True)

        ## 只用 cl-rg2 作为 encoder
        # aa_gat_in_ch = self.args['num_gat_heads']*self.args['gat_out_channels'] if 'g1' in self.args['net_type'] else self.args['encoder_size']

        self.edge_pred = torch.nn.Linear(self.args['enc_embed_size'], 1, bias=True)

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

    def forward(self, pyg_data_fwd):
        """ return agn_Int, agn_Dyn, ccl_Seq """
        ## Agent & CCL Encoding
        agn_cnl_enc = self.Agn_Cnl_Encoder(pyg_data_fwd.node_feature, pyg_data_fwd.attr, pyg_data_fwd.flat_node_type)
        raw_agn_cnl_enc = agn_cnl_enc.clone()
        # print(raw_agn_cnl_enc)

        agn_Dyn = self.leaky_relu(self.agn_Dyn_emb(self.leaky_relu(self.get_agn_feature(raw_agn_cnl_enc, pyg_data_fwd))))
        TCL_Seq = self.leaky_relu(self.agn_CCL_emb(self.leaky_relu(self.get_ccl_feature(raw_agn_cnl_enc, pyg_data_fwd))))

        # G1 Encoding
        agn_cnl_enc_ca  = self.C_A(raw_agn_cnl_enc, pyg_data_fwd.edge_index, pyg_data_fwd.flat_edge_type)
        agn_CCLs = self.leaky_relu(self.agn_CCLs_emb(self.leaky_relu(self.get_agn_feature(agn_cnl_enc_ca, pyg_data_fwd))))

        # G2 Encoding
        agn_cnl_enc_aa  = self.A_A(agn_cnl_enc_ca, pyg_data_fwd.edge_index, pyg_data_fwd.flat_edge_type)
        agn_Int  = self.leaky_relu(self.agn_Int_emb(self.leaky_relu(self.get_agn_feature(agn_cnl_enc_aa,  pyg_data_fwd))))

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


    def get_agn_feature(self, agn_cnl_enc, pyg_data):
        if self.args['dec_type'] in ['MGA', 'TGE', 'DGE']:
            sdc_tar_mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('sdcAg', 'tarAg'))
            return agn_cnl_enc[np.logical_and(sdc_tar_mask, pyg_data.goal_valid.cpu().numpy())].float()
        elif self.args['dec_type'] in ['MGAall', 'TGEall']:
            agn_mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('sdcAg', 'tarAg', 'nbrAg'))
            return agn_cnl_enc[np.logical_and(agn_mask, pyg_data.goal_valid.cpu().numpy())].float()
        elif self.args['dec_type'] in ['GDPall']:
            agn_mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('sdcAg', 'tarAg', 'nbrAg'))
            return agn_cnl_enc[np.logical_and(agn_mask, pyg_data.fut_valid.cpu().numpy())].float()
        elif self.args['dec_type'] in ['MGAsdc']:
            sdc_mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('sdcAg'))
            return agn_cnl_enc[np.logical_and(sdc_mask, pyg_data.goal_valid.cpu().numpy())].float()
        elif self.args['dec_type'] in ['Mot']:
            tar_mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('tarAg'))
            return agn_cnl_enc[np.logical_and(tar_mask, pyg_data.fut_valid.cpu().numpy())].float()
        elif self.args['dec_type'] in ['GDP', 'GDPrnn']:
            sdc_tar_mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('sdcAg', 'tarAg'))
            return agn_cnl_enc[np.logical_and(sdc_tar_mask, pyg_data.fut_valid.cpu().numpy())].float()
        elif self.args['dec_type'] in ['arGDP', 'testarGDP']:
            return agn_cnl_enc.float()
        elif self.args['dec_type'] in ['testTGE', 'testMGA', 'testGDP', 'arGDPmax', 'testarGDPmax']:
            agn_mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('sdcAg', 'tarAg', 'nbrAg'))
            # print(f'agn_cnl_enc {agn_cnl_enc.shape}, sdc_tar_mask {sdc_tar_mask.shape}')
            return agn_cnl_enc[agn_mask].float()
            # agn_mask = retrieve_mask(type_list=pyg_data.flat_node_type, wanted_type_set=('sdcAg', 'tarAg', 'nbrAg'))
            # return agn_cnl_enc[agn_mask].float()

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
        c_a_gat_feature = self.leaky_relu( self.CCL_to_Agn_GAT_1(CLA_feat,        Edges[:,edge_mask_in_C_A]) )
        return c_a_gat_feature

    # G2
    def A_A(self, CLA_feat, Edges, edge_type_list_aa):
        ''' 如果没有 centerline loop 的话，一次 gather 后centerline 的feature就为0了 '''
        edge_mask_in_A_A = retrieve_mask(edge_type_list_aa,  wanted_type_set=('sdcAg-tarAg', 'sdcAg-nbrAg', 
                                                                              'tarAg-tarAg', 'tarAg-nbrAg', 'tarAg-sdcAg', 
                                                                              'nbrAg-nbrAg', 'nbrAg-tarAg', 'nbrAg-sdcAg',
                                                                              'sdcAg-Loop',  'tarAg-Loop',  'nbrAg-Loop',
                                                                              'sdcCCL-Loop',  'tarCCL-Loop',  'nbrCCL-Loop',
                                                                              'sdcTCL-Loop',  'tarTCL-Loop',  'nbrTCL-Loop'))
        a_a_gat_feature = self.leaky_relu( self.Agn_to_Agn_GAT_1(CLA_feat,        Edges[:,edge_mask_in_A_A]) )
        return a_a_gat_feature

        

    
