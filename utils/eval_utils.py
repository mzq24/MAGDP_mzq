import torch
from torch.nn import functional as f

# smooth_l1_losstorch.nn.SmoothL1Loss(reduction=None)

def get_best_goalsets(goal_set, ground_truth, k=1, num_modes=4):
    """ 
        goal set [#Agent, #Modes, 2] 
        ground_truth [#Agent, 1, 2 ]
    """
    goal_set     = goal_set.permute(1, 0, 2)
    ground_truth = ground_truth.permute(1, 0, 2).repeat(num_modes,1,1)

    ## 计算每个 mode 的loss, 然后用最小的作为输出
    # print()
    loss = f.smooth_l1_loss(goal_set, ground_truth, reduction='none')
    loss = torch.mean(loss, dim=(1,2))

    sorted_loss, sorted_idx = torch.sort(loss, descending=False)
    # print(loss.shape)
    # print (loss)

    return sorted_loss[:k], sorted_idx[:k]