from torch.nn import functional as f

def weighted_gdp_loss(pred, label, loss_weights=(1,1,1)):
    assert pred.shape == label.shape , f'{pred.shape}, {label.shape}'
    # print(pred.shape, label.shape)
    xyz_loss  = f.smooth_l1_loss(pred[:,:,:3], label[:,:,:3])
    vxvy_loss = f.smooth_l1_loss(pred[:,:,3:5], label[:,:,3:5])
    h_loss    = f.smooth_l1_loss(pred[:,:,5:], label[:,:,5:])

    gdp_loss = loss_weights[0]*xyz_loss + loss_weights[1]*vxvy_loss + loss_weights[2]*h_loss
    # print(gdp_loss.item(), xyz_loss.item(), vxvy_loss.item(), h_loss.item())
    return gdp_loss

