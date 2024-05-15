import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple_Attention(nn.Module):
    def __init__(self, feat_dim, num_feat, hidden_size):
        super(Simple_Attention, self).__init__()
        self.feat_dim = feat_dim
        self.num_feat = num_feat
        self.hidden_size = hidden_size

        self.attention = torch.nn.Sequential(
                                            torch.nn.Linear(self.feat_dim * self.num_feat, self.hidden_size, bias=True),
                                            torch.nn.LeakyReLU(0.1),
                                            torch.nn.Linear(self.hidden_size, self.num_feat, bias=True), 
                                            )

        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, feats):
        
        scores = self.attention(torch.cat(feats, dim=1))
        # print(f'scores {scores.shape}')

        attention = self.num_feat * self.softmax(scores)
        # print(f'attention {attention.shape}')
        # print(attention[0])

        # attention = torch.ones_like(attention, device=attention.device)

        # print(feats[0].shape, attention[:,0].shape)


        weighted = [feats[i]*attention[:,i:i+1] for i in range(len(feats))]


        # print([w.shape for w in weighted])

        # queries = self.query(x)
        # keys = self.key(x)
        # values = self.value(x)
        # scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        # attention = self.softmax(scores)
        # weighted = torch.bmm(attention, values)
        return weighted

        """ 加入 TCL 的 mask """