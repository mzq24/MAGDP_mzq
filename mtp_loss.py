import torch
from torch.nn import functional as F
from typing import Tuple

def get_trajectory_and_modes(model_prediction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Splits the predictions from the model into mode probabilities and trajectory.
        :param model_prediction: Tensor of shape [batch_size, n_timesteps * n_modes * 2 + n_modes].
        :return: Tuple of tensors. First item is the trajectories of shape [batch_size, n_modes, n_timesteps, 2].
            Second item are the mode probabilities of shape [batch_size, num_modes].
        """
        num_modes = 6
        mode_probabilities = model_prediction[:, -num_modes:].clone()

        desired_shape = (model_prediction.shape[0], num_modes, -1, 2)
        trajectories_no_modes = model_prediction[:, :-num_modes].clone().reshape(desired_shape)

        return trajectories_no_modes, mode_probabilities

def mtp_loss(pred_flat, targets, eval=False):
    '''
    traj: [batch, mode, length, dim]
    targets: [batch, length, dim]
    score: [batch, mode]
    '''
    traj, score = get_trajectory_and_modes(pred_flat)
    # print(f'traj {traj.shape}, score {score.shape}, targets {targets.shape}')
    dist = torch.norm(traj[:, :, :, :2] - targets[:, None, :, :2], dim=-1)
    ade_dist = dist.mean(-1)
    fde_dist = dist[..., -1]
    gt_modes = torch.argmin(0.5*fde_dist + ade_dist, dim=-1)

    classification_loss = F.cross_entropy(score, gt_modes, label_smoothing=0.2)
    B = traj.shape[0]
    selected_trajs = traj[torch.arange(B)[:, None], gt_modes.unsqueeze(-1)].squeeze(1)
    # goal_time = torch.tensor([29, 49, 79])
    goal_time = torch.tensor([5, 9, 15])
    
    ade_loss = F.smooth_l1_loss(selected_trajs, targets[..., :2])
    # print(selected_trajs[..., goal_time, :].shape) #, targets[..., goal_time, :2].shape)
    fde_loss = F.smooth_l1_loss(selected_trajs[..., goal_time, :], targets[..., goal_time, :2])
    # fde_loss = F.smooth_l1_loss(selected_trajs[:, goal_time, :], targets[:, goal_time, :2])
    fde_loss = 0

    loss = ade_loss + 0.5*fde_loss + 2*classification_loss
    if eval:
        ade = torch.mean(torch.norm(selected_trajs - targets[..., :2], dim=-1))
        fde = torch.mean(torch.norm(selected_trajs[..., -1, :] - targets[..., -1, :2], dim=-1))
        return loss, ade, fde, classification_loss
    return loss