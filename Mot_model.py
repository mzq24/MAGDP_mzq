import torch
from sim_dataset import retrieve_mask
from sim_enc_may17 import Sim_Enc
import numpy as np

class Mot_net(torch.nn.Module):
    def __init__(self, args):
        super(Mot_net, self).__init__()
        self.args = args
        self.encoder = Sim_Enc(self.args)
        
        # trjc
        ''' trajectory completion. Embed the goal using self.agn_ip_emb.
            input: [ Int, Dyn ] '''
        self.traj_predictor = torch.nn.Sequential(
                                        torch.nn.Linear(2*args['enc_embed_size'], args['dec_hidden_size'], bias=True),
                                        torch.nn.LeakyReLU(0.1),
                                        torch.nn.Linear(args['dec_hidden_size'], 16*6*2+6, bias=True), 
                                        )
        
    def forward(self, pyg_data_fwd):
        ## Encoding
        agn_Int, agn_Dyn, _ = self.encoder(pyg_data_fwd)
        # print(f'agn_Int {agn_Int.shape}')
        ## Decoding: multimodal trajectory prediction        
        out_traj = self.traj_predictor(torch.cat((agn_Int, agn_Dyn), dim=1))#.reshape(-1, 16, 2)

        return out_traj

        

    
