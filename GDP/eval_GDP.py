import torch
from dataset_may17 import SimAgn_Dataset, flat_Batch, retrieve_mask
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.append('../utils/')
from vis_utils import scatter_goal, plot_history, plot_centerline, plot_future


def eval_a_model(model_to_eval, eval_loader, loss_func, plot=False, num_batch=-1):
    model_to_eval.eval()
    running_loss = 0.0
    for d_idx, data in tqdm(enumerate(eval_loader)):
    # for d_idx, data in enumerate(eval_loader):
        data = flat_Batch(data)
        data = data.to(model_to_eval.args['device'])

        # forward + loss calculation
        fut_pred = model_to_eval(data)
        batch_loss = loss_func(fut_pred, data.gdp_label)
        if plot:
            pass
            # fig, axs = plt.subplots(4,4,figsize=(12,12))
            # for ax_i in range(4):
            #     scatter_goal(axs[ax_i, 0],  ground_truth_goalsets[ax_i][:,0].cpu()) # 画出 batch 中每个 pyg 数据的 ground truth goalset
            #     for ax_j in range(1,4):
            #         _, topk_goalset_idx = get_best_goalsets(goal_pred[ax_i], ground_truth_goalsets[ax_i], k=3, num_modes=model_to_eval.args['num_modes'])
            #         scatter_goal( axs[ax_i, ax_j], goal_pred[ax_i].permute(1,0,2).cpu().detach()[topk_goalset_idx[ax_j-1],:,:2] )

            # for ax_col_id, title in enumerate(['GT goalset', 'goalset_top1', 'goalset_top2', 'goalset_top3']):
            #     axs[0, ax_col_id].set_title(title)

            # plt.savefig(f'./mtpgoal_images/{d_idx}_batch.png')
            # plt.close()
        # print statistics
        running_loss += batch_loss.item()
        # running_loss.append(batch_loss.item())
        if num_batch>-1 and d_idx>=num_batch:
            break
    # print(round(running_loss/(i+1),2))
    return round(running_loss/(d_idx+1),2)

if __name__ == '__main__':
    import os
    from torch_geometric.loader import DataLoader

    eval_device = 'cpu'
    model_path = './models/gdp-EP9-Loss0.55.ckpt'
    model = torch.load(model_path, map_location=eval_device)
    model.eval()
    model.args['device'] = eval_device
    model.args['eval_num_batch'] = 50
    print(model.encoder)
    #################################
    myhost = os.uname()[1]
    if myhost == 'AutoManRRCServer':
        data_path = '/disk2/SimAgent_Dataset/pyg_data_full/validation'
    else: # NSCC
        data_path = '/home/users/ntu/baichuan/scratch/sim/may07/pyg_data_full/validation'

    val_set = SimAgn_Dataset(data_path=data_path, dec_type=model.args['dec_type']) 
    print('val_size: {}/{}'.format(0, val_set.__len__()))
    valDataloader = DataLoader(val_set, batch_size=model.args['batch_size'], shuffle=True, num_workers=2)
    #################################
    # gdp_loss_func = torch.nn.SmoothL1Loss(reduction='mean') # SmoothL1Loss MSELoss
    from gdp_loss import weighted_gdp_loss
    gdp_loss_func = weighted_gdp_loss
    eval_loss = eval_a_model(model, valDataloader, gdp_loss_func, num_batch=model.args['eval_num_batch'])
    print(f'eval_loss {eval_loss} ')