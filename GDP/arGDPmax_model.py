import sys
import torch

from dataset_may17 import retrieve_mask

from sim_enc_may17 import Sim_Enc
import numpy as np
sys.path.append('../')
from utils.data_utils import convert_goalsBatch_to_goalsets_for_scenarios

class arGDPmax_net(torch.nn.Module):
    def __init__(self, args):
        super(arGDPmax_net, self).__init__()
        self.args = args
        self.encoder = Sim_Enc(self.args)
        
        # trjc
        ''' trajectory completion. Embed the goal using self.agn_ip_emb.
            input: [ Int, Dyn, Goal ] '''
        self.sdc_planner = torch.nn.GRUCell(4*args['enc_embed_size'], 4*args['enc_embed_size'])
        self.wld_planner = torch.nn.GRUCell(4*args['enc_embed_size'], 4*args['enc_embed_size'])

        self.sdc_out = torch.nn.Sequential(
                                        torch.nn.Linear(4*args['enc_embed_size'], args['dec_hidden_size'], bias=True),
                                        torch.nn.LeakyReLU(0.1),
                                        torch.nn.Linear(args['dec_hidden_size'], self.args['out_dim'], bias=True), 
                                        )
        self.wld_out = torch.nn.Sequential(
                                        torch.nn.Linear(4*args['enc_embed_size'], args['dec_hidden_size'], bias=True),
                                        torch.nn.LeakyReLU(0.1),
                                        torch.nn.Linear(args['dec_hidden_size'], self.args['out_dim'], bias=True), 
                                        )

        self.agn_Gol_emb = torch.nn.Linear(2, 128, bias=True)

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
    def arGDP_A_A(self, arGDP_AA_feat, Edges, edge_type_list_aa):
        edge_mask_in_A_A = retrieve_mask(edge_type_list_aa,  wanted_type_set=('sdcAg-tarAg', 'sdcAg-nbrAg', 
                                                                              'tarAg-tarAg', 'tarAg-nbrAg', 'tarAg-sdcAg', 
                                                                              'nbrAg-nbrAg', 'nbrAg-tarAg', 'nbrAg-sdcAg',
                                                                              'sdcAg-Loop',  'tarAg-Loop',  'nbrAg-Loop',
                                                                              'sdcCL-Loop',  'tarCL-Loop',  'nbrCL-Loop'))
        arGDP_a_a_gat_feature = self.leaky_relu( self.arGDP_GAT_1(arGDP_AA_feat,        Edges[:,edge_mask_in_A_A]) )
        return arGDP_a_a_gat_feature

    def forward(self, pyg_data_fwd):
        ## Encoding
        # agn_Int, agn_Dyn, _ = self.encoder(pyg_data_fwd)
        agn_Dyn, agn_CCLs, agn_Int, tcl_Seq = self.encoder(pyg_data_fwd)

        ## Decoding: Goal-directed Planning 
        # dataset 中的 goals 还是有一些 invalid 值的，所以要先选出来再做 goal embedding
        # print(f'pyg_data_fwd.goals is nan {torch.isnan(pyg_data_fwd.goals).any()}')
        if self.args['dec_type'] in ['arGDPmax']:
            agn_Gol = self.agn_Gol_emb(pyg_data_fwd.goals) # 这里改成 goal 方便测试 goal + trjc
        elif self.args['dec_type'] in ['testarGDPmax']:
            # agn_mask = retrieve_mask(type_list=pyg_data_fwd.flat_node_type, wanted_type_set=('sdcAg', 'tarAg', 'nbrAg'))
            # agn_Gol = torch.zeros(agn_Int.shape[0], 256)
            # print(f'agn_Gol {agn_Gol.shape}')
            agn_Gol = self.agn_Gol_emb(pyg_data_fwd.sampled_goalset)

        """ 还是应该把 TCL 信息加进来，作为 reference line, 先不加了 """
        """ 这里 C_0 是所有 Agent 的信息，[ #Agent, * ] 
            测试的时候应该在 sim_enc 里面把所有 node 的 feature 拿出来，然后再在这里筛选 给 SDC 和 World 的信息
        """
        enc_feat_list = [agn_Gol, agn_Dyn, agn_CCLs, agn_Int]

        # print(f'agn_Int {agn_Int.shape}, agn_Dyn {agn_Dyn.shape}, agn_Gol {agn_Gol.shape}, pyg_data_fwd.goal {pyg_data_fwd.goals.shape}')
        arGDP_Enc = torch.cat([feat for feat in enc_feat_list], dim=1)
        # print(f'\narGDP_Enc is nan {torch.isnan(arGDP_Enc).any()}')
        # agn_mask = retrieve_mask(type_list=pyg_data_fwd.flata_node_type, wanted_type_set=('sdcAg', 'tarAg', 'nbrAg'))
        
        sdc_mask_in_arGDPmax = retrieve_mask(type_list=pyg_data_fwd.flat_agent_node_type, wanted_type_set=('sdcAg'))

        SDC_input_0 = arGDP_Enc[ sdc_mask_in_arGDPmax]
        WLD_input_0 = arGDP_Enc[~sdc_mask_in_arGDPmax] # world 里面先取出来所有 非SDC 的feature
        
        assert SDC_input_0.shape[0]+WLD_input_0.shape[0]==arGDP_Enc.shape[0]
        
        if WLD_input_0.shape[0]==0: # 有些情况下只有 Ego 没有任何别的 valid agents
            WLD_input_0 = arGDP_Enc[ sdc_mask_in_arGDPmax]

        
        SDC_h_0 = torch.randn(SDC_input_0.shape[0], 4*self.args['enc_embed_size'], device=agn_Int.device)
        WLD_h_0 = torch.randn(WLD_input_0.shape[0], 4*self.args['enc_embed_size'], device=agn_Int.device)
       
        assert WLD_h_0.shape[0]>0 , f'{SDC_h_0.shape} {WLD_h_0.shape} {arGDP_Enc.shape}'

        SDC_Hidden_Vector_List = []
        WLD_Hidden_Vector_List = []

        ## 得到第一步的 hidden
        SDC_h = self.sdc_planner(SDC_input_0, SDC_h_0)
        # print(f'WLD_h_0 is nan {torch.isnan(WLD_h_0).any()}')
        # print(f'WLD_input_0 is nan {torch.isnan(WLD_input_0).any()}')
        WLD_h = self.wld_planner(WLD_input_0, WLD_h_0)
        SDC_Hidden_Vector_List.append(SDC_h.unsqueeze(0))
        WLD_Hidden_Vector_List.append(WLD_h.unsqueeze(0))
        # print(f'SDC_h is nan {torch.isnan(SDC_h).any()}')
        # print(f'WLD_h is nan {torch.isnan(WLD_h).any()}')

        for i in range(1,80):

            # print(f'\n{i}SDC_h is nan {torch.isnan(SDC_h).any()}')
            # print(f'WLD_h is nan {torch.isnan(WLD_h).any()}')
            # pyg_data_fwd.num_wld_agn = data_item.num_agn
            # print(pyg_data_fwd.num_wld_agn)
            # if (pyg_data_fwd.num_wld_agn == 0).any:
            #     pyg_data_fwd.num_wld_agn[pyg_data_fwd.num_wld_agn==0] = torch.ones(sum(pyg_data_fwd.num_wld_agn==0), dtype=torch.long, device=pyg_data_fwd.num_wld_agn.device)
            WLD_h_sets = convert_goalsBatch_to_goalsets_for_scenarios(WLD_h, pyg_data_fwd.num_wld_agn)

            # print([wld_h.shape for wld_h in WLD_h_sets])
            ## 取最大值给 Ego
            # assert wld_h.shape[0]
            SDC_input_next = torch.cat([torch.max(wld_h, dim=0).values.unsqueeze(0) for wld_h in WLD_h_sets], dim=0)
            
            ## 
            WLD_input_next = torch.cat([SDC_h[i:i+1].repeat(pyg_data_fwd.num_wld_agn[i], 1) for i in range(SDC_h.shape[0])], dim=0)
            # print(f'WLD_input_next is nan {torch.isnan(WLD_input_next).any()}')
            ## 得到新的 hidden
            # print(f'WLD_h {WLD_h.shape}')
            # print(f'SDC_input_next {SDC_input_next.shape}')
            # print(f'SDC_h {SDC_h.shape}')
            SDC_h = self.sdc_planner(SDC_input_next, SDC_h)
            WLD_h = self.wld_planner(WLD_input_next, WLD_h)
            SDC_Hidden_Vector_List.append(SDC_h.unsqueeze(0))
            WLD_Hidden_Vector_List.append(WLD_h.unsqueeze(0))
            
        assert len(SDC_Hidden_Vector_List)==80 and len(WLD_Hidden_Vector_List)==80
        
        ## 从 hidden list 输出轨迹
        Out_All = torch.empty(size=(arGDP_Enc.shape[0], 80, self.args['out_dim']), dtype=torch.float32, device=arGDP_Enc.device)
        Out_All[ sdc_mask_in_arGDPmax] = self.sdc_out(torch.cat(SDC_Hidden_Vector_List, dim=0).permute(1,0,2))
        Out_All[~sdc_mask_in_arGDPmax] = self.wld_out(torch.cat(WLD_Hidden_Vector_List, dim=0).permute(1,0,2))


        if self.args['dec_type'] in ['arGDPmax']:
            ## 得到用于训练的输出
            sdc_tar_mask_arGDPmax = retrieve_mask(type_list=pyg_data_fwd.flat_agent_node_type, wanted_type_set=('sdcAg', 'tarAg'))
            fut_valid_sdc_tar_mask_arGDPmax = np.logical_and(sdc_tar_mask_arGDPmax, pyg_data_fwd.agent_fut_valid.cpu().numpy())
            return Out_All[fut_valid_sdc_tar_mask_arGDPmax]
        elif self.args['dec_type'] in ['testarGDPmax']:
            
            return Out_All
        
        

    
