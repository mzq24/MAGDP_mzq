import torch
from sim_enc_may17 import Sim_Enc
from attention import TransformerCombiner

class GDP_net(torch.nn.Module):
    def __init__(self, args):
        super(GDP_net, self).__init__()
        self.args = args
        self.encoder = Sim_Enc(self.args)
        if self.args['feat_attention']:
            self.att_combiner= TransformerCombiner(self.args['enc_embed_size'], self.args['enc_embed_size'], num_layers=1)

        
        ''' trajectory completion. Embed the goal using self.agn_ip_emb.
            input: [ Int, Dyn, Goal ] '''
        if self.args['dec_type'] in ['GDP', 'GDPall']:
            num_feat = len(self.args['dec_feat']) # GD : [Goal + Dyn],  GDI : [Goal + Dyn + Int],  GDIT : [Goal + Dyn + Int + TCL]
            self.traj_planner = torch.nn.Sequential(
                                            torch.nn.Linear(num_feat*args['enc_embed_size'], args['dec_hidden_size'], bias=True),
                                            torch.nn.LeakyReLU(0.1),
                                            torch.nn.Linear(args['dec_hidden_size'], 80*self.args['out_dim'], bias=True), 
                                            )
        elif self.args['dec_type'] in ['GDPrnn']:
            num_feat = len(self.args['dec_feat']) # GD : [Goal + Dyn],  GDI : [Goal + Dyn + Int],  GDIT : [Goal + Dyn + Int + TCL]
            self.traj_planner = torch.nn.GRU(num_feat*self.args['enc_embed_size'], self.args['dec_hidden_size'], 1, batch_first=True)

            self.op = torch.nn.Linear(self.args['dec_hidden_size'], self.args['out_dim'], bias=True)

        self.agn_Gol_emb = torch.nn.Linear(2, 256, bias=True)
        
    def forward(self, pyg_data_fwd):
        ## Encoding
        agn_Dyn, agn_CCLs, agn_Int, tcl_Seq = self.encoder(pyg_data_fwd)
        

        # print(f'tcl_Seq {TCL_Seq.shape}')
        ## Decoding: Goal-directed Planning 
        # dataset 中的 goals 还是有一些 invalid 值的，所以要先选出来再做 goal embedding
        
        if self.args['dec_type'] in ['GDP', 'GDPrnn', 'GDPall']:
            agn_Gol = self.agn_Gol_emb(pyg_data_fwd.agn_goal_set) # 这里改成 goal 方便测试 goal + trjc
        elif self.args['dec_type'] in ['testGDP']:
            agn_Gol = self.agn_Gol_emb(pyg_data_fwd.sampled_goalset)
        
        if self.args['dec_feat'] == 'GD':
            enc_feat_list = [agn_Gol, agn_Dyn]
        elif self.args['dec_feat'] == 'GDI':
            enc_feat_list = [agn_Gol, agn_Dyn, agn_Int]
        elif self.args['dec_feat'] == 'GDLI':
            enc_feat_list = [agn_Gol, agn_Dyn, agn_CCLs, agn_Int]
        elif self.args['dec_feat'] == 'GDLIT':
            enc_feat_list = [agn_Gol, agn_Dyn, agn_CCLs, agn_Int, tcl_Seq]

        # print([f.shape for f in enc_feat_list])

        if self.args['feat_attention']:
            enc = self.att_combiner(torch.cat([feat.unsqueeze(1) for feat in enc_feat_list], dim=1)).flatten(start_dim=1)
        else:
            enc = torch.cat([feat for feat in enc_feat_list], dim=1)

        if self.args['dec_type'] in ['GDPrnn']:
        
            enc = enc.unsqueeze(1)
            enc = enc.repeat(1, 80, 1)

            h_dec, _ = self.traj_planner(enc)
            out_traj = self.op(h_dec)

        elif self.args['dec_type'] in ['GDP', 'testGDP', 'GDPall']:
            out_traj = self.traj_planner(enc).reshape(-1, 80, self.args['out_dim'])

        return out_traj

        

    
