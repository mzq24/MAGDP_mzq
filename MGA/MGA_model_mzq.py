import torch
from sim_enc_mzq import Sim_Enc
from attention import TransformerCombiner
# from simple_attention import Simple_Attention
from utils.data_utils import convert_goalsBatch_to_goalsets_for_scenarios

class MGA_net(torch.nn.Module):
    def __init__(self, args):
        super(MGA_net, self).__init__()
        self.args = args
        self.encoder = Sim_Enc(self.args)
        
        if self.args['feat_attention']:
            # self.attention= Simple_Attention(self.args['enc_embed_size'], 3, 3*self.args['enc_embed_size'])
            self.att_combiner= TransformerCombiner(self.args['enc_embed_size'], self.args['enc_embed_size'], num_layers=1)

        '''input: [ Int, Dyn] '''
        self.goal_dec = torch.nn.Sequential(
                            torch.nn.Linear(3*self.args['enc_embed_size'], self.args['dec_hidden_size'], bias=True),
                            torch.nn.LeakyReLU(0.1),
                            torch.nn.Linear(self.args['dec_hidden_size'], self.args['num_modes']*2, bias=True),
                            )

    def forward(self, pyg_data_fwd):
        ## Encoding
        # agn_Int, agn_Dyn, agn_CCLs = self.encoder(pyg_data_fwd)
        agn_Dyn, agn_CCLs, agn_Int, tcl_Seq = self.encoder(pyg_data_fwd)

        # print(f'agn_Int is nan {torch.isnan(agn_Int).any()}')
        # print(f'agn_Dyn is nan {torch.isnan(agn_Dyn).any()}')
        # print(f'tcl_Seq is nan {torch.isnan(tcl_Seq).any()}')
        ## Decoding: MTP goalset generation 
        # print(f'agn_Int {agn_Int.shape}, agn_Dyn {agn_Dyn.shape}, agn_CCLs {agn_CCLs.shape}')

        if self.args['feat_attention']:
            enc = self.att_combiner(torch.cat([agn_Dyn.unsqueeze(1), agn_CCLs.unsqueeze(1), agn_Int.unsqueeze(1)], dim=1)).flatten(start_dim=1)
        else:
            enc = torch.cat((agn_Dyn, agn_CCLs, agn_Int), dim=1)
        # enc = torch.cat((agn_Int, agn_Dyn, agn_CCLs), dim=1)
        # enc = agn_CCLs
        # print(enc)
        # enc = agn_Dyn

        goals_in_Batch = self.goal_dec(enc).reshape(-1, self.args['num_modes'], 2)
        # print(f'goals_in_Batch is nan {torch.isnan(goals_in_Batch).any()}')
        ## 将 Batch 中所有的 goals 按照 scenario 分成 goalsets
        Goal_Sets = convert_goalsBatch_to_goalsets_for_scenarios(goals_in_Batch, pyg_data_fwd.num_goal_valid_agn)
        # num_agn_per_scenario = torch.cat((torch.tensor([0], device=goals_in_Batch.args['device']), pyg_data_fwd.num_goal_valid_agn))
        # start_and_end_index = torch.cumsum(num_agn_per_scenario, dim=0)
        # Goal_Sets = []
        # for i in range(len(start_and_end_index)-1):
        #     start_idx, end_idx = start_and_end_index[i], start_and_end_index[i+1]
        #     goalset_in_scenario = goals_in_Batch[start_idx:end_idx]
        #     Goal_Sets.append(goalset_in_scenario)
        return Goal_Sets
