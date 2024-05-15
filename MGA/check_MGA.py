import sys
import copy
sys.path.append('../')
sys.path.append('../utils/')
from dataset_may17 import SimAgn_Dataset, flat_Batch, retrieve_mask
from data_utils import convert_goalsBatch_to_goalsets_for_scenarios
import torch
from vis_utils import scatter_goal, plot_history, plot_centerline, plot_future
import matplotlib.pyplot as plt
from eval_utils import get_best_goalsets
from mtp_goal_loss import mtp_goal_FDE, Marginal_mtp_goal_minFDE
from tqdm import tqdm 
import numpy as np
from utils.sub_utils import init_TCL_for_agents, pull_goal_to_TCL
def eval_a_model(model_to_eval, loader, loss_func, plot=False, num_batch=-1):
    model_to_eval.eval()
    running_loss = 0.0
    running_minFDE = 0.0
    running_Marginal_minFDE = 0.0
    for d_idx, data in tqdm(enumerate(loader)):
    # for d_idx, data in enumerate(loader):
        data = flat_Batch(data)
        data = data.to(model_to_eval.args['device'])
        ground_truth_goalsets = convert_goalsBatch_to_goalsets_for_scenarios(data.agn_goal_set, data.num_goal_valid_agn)
        print(set(data.flat_edge_type))
        ## 输出预测的 goal set
        goalsets_pred = model_to_eval(data)
        goal_pred = goalsets_pred
       
        if plot:
            fig, axs = plt.subplots(1, 4, figsize=(12,3))
            tar_mask = retrieve_mask(data.flat_node_type, wanted_type_set=('sdcAg'))
            tcl_mask = retrieve_mask(data.flat_node_type, wanted_type_set=('sdcTCL', 'sdcTCL_FAKE', 'sdcCCL'))
            fut_valid_tar_mask = np.logical_and(tar_mask, data.fut_valid.numpy())

            for ax_i in range(4):
                for hist in data.node_feature[tar_mask]:
                    non_zero_points = hist[(hist[:, 0] != 0) & (hist[:, 1] != 0)]
                    plot_history(axs[ax_i], non_zero_points, alpha=0.5)
                
                for fut in data.fut[fut_valid_tar_mask]:
                    plot_future(axs[ax_i], fut, alpha=0.5)

                scatter_goal(axs[ax_i],  ground_truth_goalsets[0][:,0,:].detach().cpu()) # 画出 batch 中每个 pyg 数据的 ground truth goalset
                for tcl in data.node_feature[tcl_mask]:
                    plot_centerline(axs[ax_i], tcl)
                axs[ax_i].set_aspect('equal', 'box')
                # scatter_goal(axs[1],  ground_truth_goalsets[0][:,0,:].detach().cpu()) # 画出 batch 中每个 pyg 数据的 ground truth goalset
                # scatter_goal(axs[2],  ground_truth_goalsets[0][:,0,:].detach().cpu()) # 画出 batch 中每个 pyg 数据的 ground truth goalset
        # 对每一个 goalset 输出 1 个 rollout, 共 32 个 goal set
        for goalset_i, goalset_init in enumerate(goalsets_pred[0].permute(1,0,2)):
            # print(f'goalset_init {goalset_init.shape}')
            ## 初始化 TCLs
            ## 将 goalset 调整到 TCL 上面
            # new_goalset = pull_goal_to_TCL(goalset_init.cpu(), data.to('cpu'))
            
            # print(f'new_goalset {new_goalset.shape}')
            # data.sampled_goalset = new_goalset

            if plot:
                # fig, axs = plt.subplots(1, 3, figsize=(12,5))
                # tar_mask = retrieve_mask(data.flat_node_type, wanted_type_set=('tarAg'))
                # tcl_mask = retrieve_mask(data.flat_node_type, wanted_type_set=('tarTCL', 'tarTCL_FAKE'))
                # print(f'{sum(tar_mask)}, {sum(tcl_mask)}')

                # for ax_i in range(4):
                #     for tcl in data.node_feature[tcl_mask]:
                #         plot_centerline(axs[ax_i], tcl)
                #         axs[ax_i].set_aspect('equal', 'box')
                # print('ground_truth_goalsets[0]', ground_truth_goalsets[0][:,0,:].shape)
                # print(goalset_i)
                if goalset_i>=2:
                    scatter_goal(axs[3],  goalset_init.detach().cpu(),  color='b') # 画出 batch 中每个 pyg 数据的 ground truth goalset
                    continue
                
                scatter_goal(axs[goalset_i+1],  goalset_init.detach().cpu(),  color='b') # 画出 batch 中每个 pyg 数据的 ground truth goalset
                scatter_goal(axs[3],  goalset_init.detach().cpu(),  color='b') # 画出 batch 中每个 pyg 数据的 ground truth goalset
                if axs[goalset_i+1].get_ylim()[1] -  axs[goalset_i+1].get_ylim()[0]<40:
                    axs[goalset_i+1].set_ylim(-30, 30)

            # if goalset_i>=2:
            #     break
        for ax_col_id, title in enumerate(['GT goalset', 'pred goalset 1', 'pred goalset 2', 'pred goalset 3']):
            axs[ax_col_id].set_title(title)
        plt.savefig(f'./image/{d_idx}_batch.png')
        plt.close()

        print(f'Batch_size: {len(goal_pred)}', 'goal_pred_0: ', {goal_pred[0].shape})
        # print(f' {len(goal_pred)} pred goal sets: {[goalset.shape for goalset in goal_pred]}\n ground truth goal sets: {[goalset.shape for goalset in ground_truth_goalsets]}')
        batch_loss = sum([loss_func(goal_pred[i], ground_truth_goalsets[i], num_modes=model_to_eval.args['num_modes']) for i in range(len(goal_pred))])/len(goal_pred)
        # loss_out = [loss_func(goal_pred[i], ground_truth_goalsets[i], num_modes=model_to_eval.args['num_modes']) for i in range(len(goal_pred))]
        # batch_loss = sum([l[0] for l in loss_out])/len(goal_pred)
        # print(batch_loss)
        # batch_minFDE  = 
        batch_minFDE = sum([mtp_goal_FDE(goal_pred[i], ground_truth_goalsets[i], num_modes=model_to_eval.args['num_modes']) for i in range(len(goal_pred))])/len(goal_pred)
        batch_marginal_minFDE = Marginal_mtp_goal_minFDE(torch.cat(goal_pred, dim=0), data.agn_goal_set, num_modes=model_to_eval.args['num_modes'])
        # batch_loss = Marginal_mtp_goal_minFDE(torch.cat(goal_pred, dim=0), data.agn_goal_set, num_modes=model_to_eval.args['num_modes'])
        # print(f'batch_marginal_minFDE {batch_marginal_minFDE}')
        
        running_loss += batch_loss.item()
        running_minFDE += batch_minFDE.item() #.item()
        running_Marginal_minFDE += batch_marginal_minFDE.item()
        if num_batch>-1 and d_idx>=num_batch:
            break
    return round(running_loss/(d_idx+1),2), round(running_minFDE/(d_idx+1),2), round(running_Marginal_minFDE/(d_idx+1),2)


def save_obj_pkl(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    from torch_geometric.loader import DataLoader
    from mtp_goal_loss import mtp_goal_loss, mtp_goal_FDE
    
    model_path = './models/MGAsdcM32-EP48-Loss1.16-minFDE2.44.ckpt' # MGAsdcM6-EP12-Loss2.72-minFDE5.18.ckpt MGAsdcM6-EP11-Loss2.55-minFDE4.93.ckpt
    eval_device = 'cpu'
    model = torch.load(model_path, map_location=eval_device)
    model.eval()
    model.args['batch_size'] = 1
    model.args['device'] = eval_device
    # model.args['dec_type'] = 'mtp_goal_veh' # 'mtp_goal_veh' # 'mtpGoal'
    model.args['num_val_batch'] = 0

    print(model.encoder)
    #################################
    import os
    myhost = os.uname()[1]
    if myhost == 'AutoManRRCServer':
        data_path = '/disk2/SimAgent_Dataset/pyg_data_Jun11/validation'
    elif myhost == 'asp2a-login-ntu01':
        data_path = '/disk2/SimAgent_Dataset/pyg_data_full/validation'
    else: # NSCC
        data_path = '/home/users/ntu/baichuan/scratch/sim/may07/pyg_data_full/validation'
    dataset = SimAgn_Dataset(data_path=data_path, dec_type=model.args['dec_type']) 
    loader  = DataLoader(dataset, batch_size=model.args['batch_size'], shuffle=True, num_workers=2)
    #################################

    # loss_function = mtp_goal_loss
    eval_loss = eval_a_model(model, loader, loss_func=mtp_goal_loss, plot=True, num_batch=model.args['num_val_batch'])
    print(f'eval SmoothL1 loss {eval_loss[0]}, Joint-minFDE {eval_loss[1]}, Marginal-minFDE {eval_loss[2]}')
    print(model_path)
#     // The object states for a single object through the scenario.
#     message Track {
#     enum ObjectType {
#     TYPE_UNSET = 0;  // This is an invalid state that indicates an error.
#     TYPE_VEHICLE = 1;
#     TYPE_PEDESTRIAN = 2;
#     TYPE_CYCLIST = 3;
#     TYPE_OTHER = 4;
#   }