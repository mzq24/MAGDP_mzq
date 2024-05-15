import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
sys.path.append('../utils/')
from dataset_may17 import SimAgn_Dataset, flat_Batch, retrieve_mask

from vis_utils import scatter_goal, plot_history, plot_centerline, plot_future
import numpy as np
from sub_utils import smoothen_a_rollout

def eval_a_model(model_list, eval_loader, loss_func, plot=False, num_batch=-1):
    for model in model_list:
        model.eval()
    running_loss = 0.0
    for d_idx, data in tqdm(enumerate(eval_loader)):
    # for d_idx, data in enumerate(eval_loader):
        data = flat_Batch(data)
        data = data.to(model_list[0].args['device'])


        Futs = []
        for m_i, model in enumerate(model_list):

            # forward + loss calculation
            fut_pred = model(data)
            print(f'fut_pred {fut_pred.shape}')
        

            batch_loss = loss_func(fut_pred[:,:,:model_list[0].args['out_dim']], data.gdp_label[:,:,:model_list[0].args['out_dim']])
            print(f'fut_pred {fut_pred.shape}, data.gdp_label {data.gdp_label.shape}')
            Futs.append(fut_pred)
        
        if plot:
            fig, axs = plt.subplots(1,3,figsize=(12,6))
            for ax_i in range(3):
                scatter_goal(axs[ax_i],  data.agn_goal_set.cpu()) # 画出 batch 中每个 pyg 数据的 ground truth goalset
            
            ## 画 Fut GT + GDP 
            tar_mask = retrieve_mask(data.flat_node_type, wanted_type_set=('tarAg'))
            tcl_mask = retrieve_mask(data.flat_node_type, wanted_type_set=('tarTCL'))
            fut_valid_tar_mask = np.logical_and(tar_mask, data.fut_valid.numpy())
            for fut in data.fut[fut_valid_tar_mask]:
                plot_future(axs[0], fut, alpha=0.5)

            for f in Futs[0].detach().numpy():
                plot_future(axs[1], f[::1], alpha=0.5)

            Fut_smooth = smoothen_a_rollout(Futs[1].detach().numpy())
            for f in Fut_smooth:

                # Set the window size for moving average
                # window_size = 11

                # # Apply moving average to smooth the trajectory
                # # Pad the data to preserve length
                # padded_x = np.pad(f[:,0], (window_size//2, window_size-1-window_size//2), mode='edge')
                # padded_y = np.pad(f[:,1], (window_size//2, window_size-1-window_size//2), mode='edge')

                # f[:,0] = np.convolve(padded_x, np.ones(window_size)/window_size, mode='valid')
                # f[:,1] = np.convolve(padded_y, np.ones(window_size)/window_size, mode='valid')
                plot_future(axs[2], f[::1], alpha=0.5)

            ## 画 TCLs + Hist 
            for ax_i in range(3):
                for tcl in data.node_feature[tcl_mask]:
                    plot_centerline(axs[ax_i], tcl)
                for hist in data.node_feature[tar_mask]:
                    non_zero_points = hist[(hist[:, 0] != 0) & (hist[:, 1] != 0)]
                    plot_history(axs[ax_i], non_zero_points, alpha=0.5)
                

            for ax_col_id, title in enumerate(['GT Fut', 'GDP Fut', 'GDP Fut smooth']):
                axs[ax_col_id].set_title(title)
                axs[ax_col_id].set_aspect('equal', 'box')
                # axs[ax_col_id].set_ylim(-50, 50)

            plt.savefig(f'./images/GDP{d_idx}_batch.png', dpi=300)
            plt.close()
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
    GDP_path = './models/arGDPmaxout6GDLI-Loss0.49.ckpt'
    GDPrnn_path = './models/arGDPmaxout6GDLI-Loss0.49.ckpt'
    GDP_model = torch.load(GDP_path, map_location=eval_device)
    GDPrnn_model = torch.load(GDPrnn_path, map_location=eval_device)

    GDP_model.eval()
    GDP_model.args['device'] = eval_device
    GDP_model.args['eval_num_batch'] = 1
    GDP_model.args['batch_size'] = 1
    # GDP_model.args['dec_feat'] = 'GDLIT'

    # print(GDP_model.encoder)
    GDPrnn_model.eval()
    GDPrnn_model.args['device'] = eval_device
    GDPrnn_model.args['eval_num_batch'] = 1
    GDPrnn_model.args['batch_size'] = 1
    # GDPrnn_model.args['dec_feat'] = 'GDLIT'

    #################################
    myhost = os.uname()[1]
    if myhost == 'AutoManRRCServer':
        data_path = '/disk6/SimAgent_Dataset/pyg_data_Jun23/training'
    elif myhost == 'amrrc':
        data_path = '/home/xy/SimAgent_Dataset/pyg_data_Jun23/training'
    else: # NSCC
        data_path = '/home/users/ntu/baichuan/scratch/sim/may07/pyg_data_full/validation'

    val_set = SimAgn_Dataset(data_path=data_path, dec_type=GDP_model.args['dec_type']) 
    print('val_size: {}/{}'.format(0, val_set.__len__()))
    valDataloader = DataLoader(val_set, batch_size=GDP_model.args['batch_size'], shuffle=True, num_workers=2)
    #################################
    gdp_loss_func = torch.nn.SmoothL1Loss(reduction='mean') # SmoothL1Loss MSELoss

    eval_loss = eval_a_model([GDP_model, GDPrnn_model], valDataloader, gdp_loss_func, plot=True, num_batch=GDP_model.args['eval_num_batch'])
    print(f'eval_loss {eval_loss} ')