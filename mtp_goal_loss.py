import torch
from torch.nn import functional as f

# smooth_l1_losstorch.nn.SmoothL1Loss(reduction=None)

def marginal_mtp_goal_loss(goal_set, ground_truth, num_modes=4):
    """ 
        goal set     [#Agent, #Modes, 2] 
        ground_truth [#Agent, 1, 2 ]
    """
    if goal_set.shape[0]==0:
        return torch.tensor(0)

    ground_truth = ground_truth.repeat(1,num_modes,1)

    ## 计算每个 mode 的loss, 然后用最小的作为输出
    loss = f.smooth_l1_loss(goal_set, ground_truth, reduction='none')
    # print(f'loss         is nan {torch.isnan(loss).any()} {loss}\n')
    # print(f'\nloss {loss.shape}')
    loss = torch.mean(loss, dim=2)
    # print(f'loss {loss.shape}')
    loss, _ = torch.min(loss, dim=1)
    # print(f'loss {loss.shape}')
    # print(f'loss_mean is nan {torch.isnan(loss).any()}')
    # print(loss.shape)
    # print (loss)

    return torch.mean(loss)

def mtp_goal_loss(goal_set, ground_truth, num_modes=4):

    """ 计算 Joint MTP Goal loss, 只针对一个 scenario 对应输出的 goal sets
        goal set     [#Agent, #Modes, 2] 
        ground_truth [#Agent, 1, 2 ]
    """
    if goal_set.shape[0]==0:
        return torch.tensor(0)
    # print(f'goal_set {goal_set.shape}, ground_truth {ground_truth.shape}')
    # print(f'goal_set is     nan {torch.isnan(goal_set).any()}')
    # print(f'ground_truth is nan {torch.isnan(ground_truth).any()}')
    goal_set     = goal_set.permute(1, 0, 2)
    ground_truth = ground_truth.permute(1, 0, 2).repeat(num_modes,1,1)

    ## 计算每个 mode 的loss, 然后用最小的作为输出
    loss = f.smooth_l1_loss(goal_set, ground_truth, reduction='none')
    # batch_loss = sum([loss_func(goal_pred[i], ground_truth_goalsets[i], num_modes=model_to_tr.args['num_modes']) for i in range(len(goal_pred))])/len(goal_pred)

    # print(f'loss         is nan {torch.isnan(loss).any()} {loss}\n')
    # print(f'loss {loss}')
    loss = torch.mean(loss, dim=(1,2))
    # print(f'loss_mean is nan {torch.isnan(loss).any()}')
    # print('loss', loss.shape)
    # print(loss, torch.min(loss), torch.argmin(loss))
    # print (loss)
    # return loss
    return torch.min(loss)
    # return torch.min(loss) , torch.argmin(loss)


def Marginal_mtp_goal_minFDE(goal_set, ground_truth, num_modes=4):
    """ 
        计算 Joint-minFDE loss for Multimodal Goal Prediction
        goal set     [#Agent, #Modes, 2] 
        ground_truth [#Agent, 1, 2 ]
    """
    if goal_set.shape[0]==0:
        return torch.tensor(0)
    goal_set     = goal_set.permute(1, 0, 2)
    ground_truth = ground_truth.permute(1, 0, 2).repeat(num_modes,1,1)

    multimodal_fde_loss = torch.sqrt(torch.sum(f.mse_loss(goal_set, ground_truth, reduction='none'), dim=-1))
    # print(f'multimodal_fde_loss: {multimodal_fde_loss.shape}')
    # print(f'multimodal_fde_loss \n{multimodal_fde_loss}')
    min_fde, _ = torch.min(multimodal_fde_loss, dim=0)
    # print(f'min_fde: {min_fde.shape}')
    # print(f'loss is nan {torch.isnan(loss).any()}')
    # print(f'loss {loss}')
    # loss = torch.mean(mse_loss, dim=(1,2))
    return torch.mean(min_fde)

def mtp_goal_FDE(goal_set, ground_truth, num_modes=4):
    """ 
        计算 Joint-minFDE loss for Multimodal Goal Prediction
        goal set     [#Agent, #Modes, 2] 
        ground_truth [#Agent, 1, 2 ]
    """
    if goal_set.shape[0]==0:
        return torch.tensor(0)
    goal_set     = goal_set.permute(1, 0, 2)
    ground_truth = ground_truth.permute(1, 0, 2).repeat(num_modes,1,1)

    ## 计算每个 mode 的loss, 然后用最小的作为输出
    joint_fde_loss = torch.sqrt(torch.sum(f.mse_loss(goal_set, ground_truth, reduction='none'), dim=-1))
    # print(f'\njoint_fde_loss \n{joint_fde_loss}\n{joint_fde_loss.shape}')
    joint_fde_loss = torch.mean(joint_fde_loss,dim=-1)
    # print(f'mseloss: {joint_fde_loss.shape}')
    min_joint_fde = torch.min(joint_fde_loss)
    # print(f'loss is nan {torch.isnan(loss).any()}')
    # print(f'loss {loss}')
    # loss = torch.mean(mse_loss, dim=(1,2))
    return min_joint_fde

def Unimodal_goal_FDE(goal_set, ground_truth):
    """ 
        计算 Joint-minFDE loss for Multimodal Goal Prediction
        goal set     [ #Agent, 2 ] 
        ground_truth [ #Agent, 2 ]
    """
    if goal_set.shape[0]==0:
        return torch.tensor(0)
    um_goal_fde_loss = torch.sqrt(torch.sum(f.mse_loss(goal_set, ground_truth, reduction='none'), dim=-1))
    # print(f'um_goal_fde_loss: {um_goal_fde_loss.shape}')
    return torch.mean(um_goal_fde_loss)

from shapely.geometry import LineString, Point
def calculate_ccl_dists_to_goal(agn_goal, pyg_node_feature, agn_ccl_mask):
    agn_CCLs = pyg_node_feature[agn_ccl_mask].detach().numpy()
    # print(f'agn_goal {agn_goal.shape}')
    CCLs_dist_to_agn = torch.tensor([LineString(ccl[:,:2]).distance(Point(agn_goal.detach().numpy())) for ccl in agn_CCLs])
    return CCLs_dist_to_agn

def is_FAKE_CCL(ccl):
    # print(torch.sum(torch.abs(ccl))<0.0001)
    return True if torch.sum(torch.abs(ccl))<0.0001 else False

def Goals_distance_to_CCLs(goal_set, agn_CCLs_list):
    """ 
    只针对一个scenario 数据计算，而不是整个Batch！！！
    对每个 agent 的 每个 goal 计算其到对应的 CCLs 的最短距离 
    一个 goal set [#Agents, 2]
    """
    

    # print(goal_set.shape)
    # print(pyg_data)
    min_dist_to_CCLs = 0.0 # torch.tensor(0.0)
    for i, agn_goal in enumerate(goal_set):
        # print(f'agn_mm_goal {agn_mm_goal.shape}')
        # print(f'agn_CCLs_list[i][0] {agn_CCLs_list[i][0].view(-1,2).shape}')

        c_dist = torch.cdist(agn_CCLs_list[i][0].view(-1,2), agn_goal.unsqueeze(0))
        # print(f'c_dist {c_dist.shape}')
        d = torch.min(c_dist, dim=0)[0]
        # print('d', d.shape)
        min_dist_to_CCLs += d
        
        ## 找到这个 Agent 对应的 CCLs
        # print(agn_CCLs_list[i][0].shape)
        # print(pyg_data.flat_node_name[:10])
        # CCLs_dist_to_agn = 0.0
        # for agn_goal in agn_mm_goal:
            
            # print(f'agn_goal {agn_goal.shape}')
            # distances = torch.cdist(curve_points.unsqueeze(0), point.unsqueeze(0))
            # print(torch.min( torch.cdist(agn_CCLs_list[i][0][0,:,:2].unsqueeze(0), agn_goal.unsqueeze(0) )) )
            # CCLs_dist_to_agn = [ LineString(ccl[:,:2].cpu()).distance(Point(agn_goal.detach().cpu().numpy())) for ccl in agn_CCLs_list[i][0] if not is_FAKE_CCL(ccl) ]
            # CCLs_dist_to_agn = [ torch.min( agn_CCLs_list[i][0] for  ccl in agn_CCLs_list[i][0] if not is_FAKE_CCL(ccl)  ]
            # for ccl in agn_CCLs_list[i][0]:
            #     if not is_FAKE_CCL(ccl):
            #         ## 如果没有 CCL 怎么办？
            #         CCLs_dist_to_agn += torch.min( torch.cdist(ccl[:,:2].unsqueeze(0), agn_goal.unsqueeze(0) ) ) 

            # print(CCLs_dist_to_agn)
            # if len(CCLs_dist_to_agn)==0:
            #     min_dist_to_CCLs += 0.0 
                # print(torch.tensor(0.))
            # else:
            #     min_dist_to_CCLs += min(CCLs_dist_to_agn) # /len(agn_CCLs_list[i][0])
                # print(torch.min(CCLs_dist_to_agn))
                # print(torch.min(CCLs_dist_to_agn))

            ## 找出 每个goal 到 CCLs 的最小距离，然后取 平均作为 loss
            # pass
    # print(torch.tensor(min_dist_to_CCLs))
    # print(torch.mean(torch.tensor(min_dist_to_CCLs))
    loss = min_dist_to_CCLs / goal_set.shape[0]
    # print('max', loss)
    """ 三种策略：1. 惩罚均值，2. 惩罚最大值，3. 惩罚best mode 对应的 """
    return loss

        

