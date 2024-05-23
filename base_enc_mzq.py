import torch
from dataset_may17 import retrieve_mask
import sys 
sys.path.append('../')
# from data_utils import normlize_Agn_seq, normlize_CCL_seq
class Sim_Base_Enc(torch.nn.Module):
    def __init__(self, args):
        super(Sim_Base_Enc, self).__init__()
        self.args = args
        
        self.Agn_Enc_MLP = torch.nn.Sequential(
                                        torch.nn.Linear(11*6, self.args['enc_hidden_size'], bias=True),
                                        torch.nn.LeakyReLU(0.1),
                                        torch.nn.Linear(self.args['enc_hidden_size'], self.args['enc_hidden_size'], bias=True), 
                                        torch.nn.LeakyReLU(0.1),
                                        torch.nn.Linear(self.args['enc_hidden_size'], self.args['enc_hidden_size'], bias=True),
                                        torch.nn.LeakyReLU(0.1),
                                        )
        
        self.CCL_Enc_MLP = torch.nn.Sequential(
                                        torch.nn.Linear(11*4, self.args['enc_hidden_size'], bias=True),
                                        torch.nn.LeakyReLU(0.1),
                                        torch.nn.Linear(self.args['enc_hidden_size'], self.args['enc_hidden_size'], bias=True), 
                                        torch.nn.LeakyReLU(0.1),
                                        torch.nn.Linear(self.args['enc_hidden_size'], self.args['enc_hidden_size'], bias=True),
                                        torch.nn.LeakyReLU(0.1),
                                        )
        self.Agn_Attr_Enc = torch.nn.Sequential(
                                        torch.nn.Linear(4, self.args['enc_hidden_size'], bias=True),
                                        torch.nn.LeakyReLU(0.1),
                                        torch.nn.Linear(self.args['enc_hidden_size'], self.args['enc_hidden_size'], bias=True), 
                                        torch.nn.LeakyReLU(0.1),
                                        )

        self.CCL_Attr_Enc = torch.nn.Sequential(
                                        torch.nn.Linear(4, self.args['enc_hidden_size'], bias=True),
                                        torch.nn.LeakyReLU(0.1),
                                        torch.nn.Linear(self.args['enc_hidden_size'], self.args['enc_hidden_size'], bias=True), 
                                        torch.nn.LeakyReLU(0.1),
                                        )
        
        self.Fuse_Agn_Hist_Attr_MLP = torch.nn.Sequential(
                                        torch.nn.Linear(2*self.args['enc_hidden_size'], self.args['enc_hidden_size'], bias=True),
                                        torch.nn.LeakyReLU(0.1),
                                        torch.nn.Linear(self.args['enc_hidden_size'], self.args['enc_hidden_size'], bias=True), 
                                        torch.nn.LeakyReLU(0.1),
                                        )
        
        self.Fuse_CCL_Seqs_Attr_MLP = torch.nn.Sequential(
                                        torch.nn.Linear(2*self.args['enc_hidden_size'], self.args['enc_hidden_size'], bias=True),
                                        torch.nn.LeakyReLU(0.1),
                                        torch.nn.Linear(self.args['enc_hidden_size'], self.args['enc_hidden_size'], bias=True), 
                                        torch.nn.LeakyReLU(0.1),
                                        )
        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)

        ## Layer Norms
        self.norm1 = torch.nn.LayerNorm(self.args['enc_hidden_size'])


    def Agn_Cnl_Encoder(self, x, attr, node_type_list_ace):
        """ Encode sequential features of all considered vehicles 
            Agn_Seq: 
            Cnl_Seq: 
        """
        Agn_mask = retrieve_mask(type_list=node_type_list_ace, wanted_type_set=('sdcAg', 'tarAg', 'nbrAg'))
        
        ## 除了 Agents 就都是 CCL 了
        CCL_mask = ~Agn_mask
        assert sum(Agn_mask) + sum(CCL_mask)==x.shape[0]
        Agn_Cnl_Seq_Feature = torch.empty(size=(x.shape[0], self.args['enc_hidden_size']), dtype=torch.float32, device=x.device)
        ## 归一化数据
        # x[Agn_mask] = normlize_Agn_seq(x[Agn_mask])
        # x[CCL_mask] = normlize_CCL_seq(x[CCL_mask])
        ## 序列信息
        Agn_Enc_Hist = self.Agn_Enc_MLP(torch.flatten(x[Agn_mask], start_dim=1))
        Cnl_Enc_Seqs = self.CCL_Enc_MLP(torch.flatten(x[CCL_mask,:,:4], start_dim=1))
        ## Attribute 信息
        Agn_Enc_Attr = self.Agn_Attr_Enc(attr[Agn_mask])
        CCL_Enc_Attr = self.CCL_Attr_Enc(attr[CCL_mask])
        Agn_Cnl_Seq_Feature[Agn_mask] = self.Fuse_Agn_Hist_Attr_MLP(torch.cat((Agn_Enc_Hist, Agn_Enc_Attr), dim=-1))
        Agn_Cnl_Seq_Feature[CCL_mask] = self.Fuse_CCL_Seqs_Attr_MLP(torch.cat((Cnl_Enc_Seqs, CCL_Enc_Attr), dim=-1))
        ## Layer Norm 一下
        Agn_Cnl_Seq_Feature = self.norm1(Agn_Cnl_Seq_Feature)
        return Agn_Cnl_Seq_Feature
    
    def forward(self, pyg_data_fwd):
        agn_cnl_enc = self.Agn_Cnl_Encoder(pyg_data_fwd.node_feature, pyg_data_fwd.flat_node_type)

        print('in base model {}'.format(agn_cnl_enc.shape))
        return agn_cnl_enc
