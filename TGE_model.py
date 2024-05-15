import argparse
import os
import numpy as np
import math

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from nll_loss import outputActivation

from sim_enc_may17 import Sim_Enc

class TGE_net(nn.Module):
    """ 根据场景信息以及指定 TCL 来输出 goal。Goal可以有以下三种形式：
        1. 仅输出 （X,Y)，[#Agent, 2]
        2. 输出 MTR 中的 Square Gaussian, [#Agent, 3]
        3. 输出 MTR 中的 2-D Gaussian, [#Agent, 5]
        """
    def __init__(self, args):
        super(TGE_net, self).__init__()
        self.args = args

        self.encoder = Sim_Enc(self.args)

        # 使用预训练的 Sim Encoder
        # self.encoder = torch.load(self.args['encoder_path']).encoder
        # self.encoder.args['dec_type'] = self.args['dec_type']
        # self.encoder = self.encoder.eval()
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
            
        self.model = nn.Sequential(
            nn.Linear(3*self.args['enc_embed_size'],  self.args['dec_hidden_size']),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(  self.args['dec_hidden_size'], self.args['dec_hidden_size']),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(  self.args['dec_hidden_size'], self.args['goal_dim']),
            )
    
    def forward(self, pyg_data_fwd):
        """ 
            pyg_data_fwd: 一个 Batch 的 graph.
        """
        ## Encoding
        agn_Int, agn_Dyn, tcl_Seq = self.encoder(pyg_data_fwd)
        # print(f'agn_Int is nan {torch.isnan(agn_Int).any()}')
        # print(f'agn_Dyn is nan {torch.isnan(agn_Dyn).any()}')
        # print(f'tcl_Seq is nan {torch.isnan(tcl_Seq).any()}')
        # raw_agn_cnl_enc = agn_cnl_enc.clone()
        ## TCL feature
        # Sample noise 
        # print(f'agn_Int {agn_Int.shape}, agn_Dyn {agn_Dyn.shape}, tcl_Seq {tcl_Seq.shape}')
        agn_Int = torch.nan_to_num(agn_Int)
        agn_Dyn = torch.nan_to_num(agn_Dyn)
        tcl_Seq = torch.nan_to_num(tcl_Seq)
        # agn_Noise = Variable(FloatTensor(np.random.normal(0, 1, (agn_Int.shape[0], noise_dim)), device=agn_Int.device))
        gen_input = torch.cat((agn_Int, agn_Dyn, tcl_Seq), -1)
        agn_goals = self.model(gen_input)
        agn_goals = outputActivation(agn_goals)
        # agn_goals = agn_goals.unsqueeze(1)
        return agn_goals
