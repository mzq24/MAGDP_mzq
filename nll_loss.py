import torch
## Batchwise NLL loss, uses mask for variable output lengths
def maskedNLL_Loss(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    sigX = y_pred[:,:,2]
    sigY = y_pred[:,:,3]
    rho = y_pred[:,:,4]
    ohr = torch.pow(1-torch.pow(rho,2),-0.5)
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]
    # If we represent likelihood in feet^(-1):
    # out = 0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + torch.pow(sigY, 2)*torch.pow(y-muY, 2) - 2*rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379
    # If we represent likelihood in m^(-1):

    out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160
    
    acc[:,:,0] = out
    acc[:,:,1] = out
    acc = acc*mask
    lossVal = torch.sum(acc)/torch.sum(mask)
    return 


def NLL_Loss(y_pred, y_gt):
    # acc = torch.zeros_like(y_gt)
    print(f'y_pred is nan {torch.isnan(y_pred).any()}')
    print(f'y_gt is nan {torch.isnan(y_gt).any()}')
    muX  = y_pred[:,:,0]
    muY  = y_pred[:,:,1]
    sigX = y_pred[:,:,2]
    sigY = y_pred[:,:,3]
    rho  = y_pred[:,:,4]
    print(f'muX is nan {torch.isnan(muX).any()}, min {torch.min(muX)}, max {torch.max(muX)}')
    print(f'muY is nan {torch.isnan(muY).any()}, min {torch.min(muY)}, max {torch.max(muY)}')
    print(f'sigX is nan {torch.isnan(sigX).any()}, min {torch.min(sigX)}, max {torch.max(sigX)}')
    print(f'sigY is nan {torch.isnan(sigY).any()}, min {torch.min(sigY)}, max {torch.max(sigY)}')
    print(f'rho is nan {torch.isnan(rho).any()}, min {torch.min(rho)}, max {torch.max(rho)}')
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]

    eps_rho = torch.tensor(1e-6)
    diff_x = x - muX
    diff_y = y - muY

    ohr = 1/(torch.max(1 - rho * rho, eps_rho)) #avoid infinite values
    out = 0.5*ohr * (diff_x * diff_x / (sigX * sigX) + diff_y * diff_y / (sigY * sigY) -
            2 * rho * diff_x * diff_y / (sigX * sigY)) + torch.log(
            sigX * sigY) - 0.5*torch.log(ohr) + torch.log(torch.tensor(torch.pi)*2)
    
    # eps_rho = torch.tensor(1e-6)
    # ohr = torch.pow(torch.max(1-torch.pow(rho,2), eps_rho),-0.5) #avoid infinite values
    
    # ohr = 1/(torch.max(1 - rho * rho, eps_rho)) #avoid infinite values
    

    print(f'ohr is nan {torch.isnan(ohr).any()}')
    # If we represent likelihood in feet^(-1):
    # out = 0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + torch.pow(sigY, 2)*torch.pow(y-muY, 2) - 2*rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379
    # If we represent likelihood in m^(-1):
    # out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160
    print(f'out is nan {torch.isnan(out).any()}')
    # acc[:,:,0] = out
    # acc = out
    # acc = acc*mask
    # print(f'acc is nan {torch.isnan(acc).any()}')
    print(out)
    lossVal = torch.mean(out)
    print(f'lossVal is nan {torch.isnan(lossVal).any()}')
    return lossVal

def outputActivation(x):
    # print('x in out act', x.shape)
    muX  = x[:,0:1]
    muY  = x[:,1:2]
    sigX = x[:,2:3]
    sigY = x[:,3:4]
    rho  = x[:,4:5]
    # print('rho in out act', rho)
    # sigX = torch.max([sigX, 10]) # sigX = torch.clamp(sigX, 0.001, 10) # sigX = torch.exp(sigX)
    # sigY = torch.max([sigY, 10]) # sigY = torch.clamp(sigY, 0.001, 10) # sigY = torch.exp(sigY)
    log_std_range=(-1.609, 5.0)
    sigX = torch.clamp(sigX, log_std_range[0], log_std_range[1])
    sigY = torch.clamp(sigY, log_std_range[0], log_std_range[1])
    # sigX = torch.exp(sigX)
    # sigY = torch.exp(sigY)
    rho_limit =0.5
    # rho = torch.clamp(torch.tanh(rho), -1+rho_eps, 1-rho_eps)
    rho = torch.clamp(rho, min=-rho_limit, max=rho_limit)
    # print('rho in out act 2', rho)
    out = torch.cat([muX, muY, sigX, sigY, rho], dim=1)
    
    return out


# import torch
import torch.distributions as dist

def gaussian_nll_loss_chatgpt(y_true, y_pred, reg_factor=1e-6):
    """
    Computes the negative log-likelihood loss for a batch of data points using the predicted mean vector and covariance matrix.

    Args:
        y_true: A tensor of shape (batch_size, 2) representing the ground truth labels (X*, Y*) for the batch of data points.
        y_pred: A tensor of shape (batch_size, 5) representing the predicted mean (X, Y) and standard deviations (SigmaX, SigmaY) and correlation coefficient rho.

    Returns:
        A scalar representing the negative log-likelihood loss for the batch of data points.
    """
    X, Y, SigmaX, SigmaY, rho = y_pred.split(1, dim=1)
    # print(f'y_true {y_true.shape}, y_pred {y_pred.shape}')
    covariance = torch.stack((
        torch.stack((SigmaX**2, rho * SigmaX * SigmaY), dim=1),
        torch.stack((rho * SigmaX * SigmaY, SigmaY**2), dim=1)
    ), dim=1).squeeze(-1)
    # print(f'')
    covariance += reg_factor * torch.eye(2, device=y_pred.device).unsqueeze(0)

    # print(f"X shape: {X.shape}")
    # print(f"Y shape: {Y.shape}")
    # print(f"SigmaX shape: {SigmaX.shape}")
    # print(f"SigmaY shape: {SigmaY.shape}")
    # print(f"rho shape: {rho.shape}")
    # print(f"covariance shape: {covariance.shape}")
    # print(f"y_true shape: {y_true.shape}")
    mvn = dist.multivariate_normal.MultivariateNormal(torch.cat([X, Y], dim=1), covariance_matrix=covariance)
    nll = -mvn.log_prob(y_true)
    return nll.mean()

## from MTR
def MTR_nll_loss_gmm_direct(pred_trajs, gt_trajs,
                            use_square_gmm=False, log_std_range=(-1.609, 5.0), rho_limit=0.5):
    """
    GMM Loss for Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
    Written by Shaoshuai Shi 

    Args:
        pred_trajs (batch_size, num_timestamps, 5 or 3)
        gt_trajs   (batch_size, num_timestamps, 2):
    """
    
    assert pred_trajs.shape[-1] == 5

    batch_size = pred_trajs.shape[0]

    nearest_trajs = pred_trajs# (batch_size, num_timestamps, 5)
    res_trajs = gt_trajs - nearest_trajs[:, :, 0:2]  # (batch_size, num_timestamps, 2)
    dx = res_trajs[:, :, 0]
    dy = res_trajs[:, :, 1]

    log_std1 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
    log_std2 = torch.clip(nearest_trajs[:, :, 3], min=log_std_range[0], max=log_std_range[1])
    std1 = torch.exp(log_std1)  # (0.2m to 150m)
    std2 = torch.exp(log_std2)  # (0.2m to 150m)
    rho = torch.clip(nearest_trajs[:, :, 4], min=-rho_limit, max=rho_limit)

    # -log(a^-1 * e^b) = log(a) - b
    reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * torch.log(1 - rho**2)  # (batch_size, num_timestamps)
    reg_gmm_exp = (0.5 * 1 / (1 - rho**2)) * ((dx**2) / (std1**2) + (dy**2) / (std2**2) - 2 * rho * dx * dy / (std1 * std2))  # (batch_size, num_timestamps)

    
    reg_loss = (reg_gmm_log_coefficient + reg_gmm_exp).sum(dim=-1)

    return reg_loss.mean()

# pred_goal = torch.tensor([[[12,12,1,1,0.5]]], dtype=float).repeat(12,1,1)
# gt_goal = torch.tensor([[[20.2,20.2]]], dtype=float).repeat(12,1,1)

# nll = MTR_nll_loss_gmm_direct(pred_goal, gt_goal)
# print(f'nll {nll}')