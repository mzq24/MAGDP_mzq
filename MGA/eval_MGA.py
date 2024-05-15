from dataset_may17 import SimAgn_Dataset, flat_Batch, retrieve_mask
from data_utils import convert_goalsBatch_to_goalsets_for_scenarios
import torch
from vis_utils import scatter_goal, plot_history, plot_centerline, plot_future
import matplotlib.pyplot as plt
from eval_utils import get_best_goalsets
from mtp_goal_loss import mtp_goal_FDE, Marginal_mtp_goal_minFDE
from tqdm import tqdm 
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

        # forward + loss calculation
        goal_pred = model_to_eval(data)
        
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
        if plot:
            fig, axs = plt.subplots(4,4,figsize=(12,12))
            for ax_i in range(4):
                scatter_goal(axs[ax_i, 0],  ground_truth_goalsets[ax_i][:,0].cpu()) # 画出 batch 中每个 pyg 数据的 ground truth goalset
                for ax_j in range(1,4):
                    _, topk_goalset_idx = get_best_goalsets(goal_pred[ax_i], ground_truth_goalsets[ax_i], k=3, num_modes=model_to_eval.args['num_modes'])
                    scatter_goal( axs[ax_i, ax_j], goal_pred[ax_i].permute(1,0,2).cpu().detach()[topk_goalset_idx[ax_j-1],:,:2] )

            for ax_col_id, title in enumerate(['GT goalset', 'goalset_top1', 'goalset_top2', 'goalset_top3']):
                axs[0, ax_col_id].set_title(title)

            plt.savefig(f'./mtpgoal_images/{d_idx}_batch.png')
            plt.close()
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
    
    model_path = './models/MGAM32-EP1-Loss6.57-minFDE11.69.ckpt'
    eval_device = 'cuda:2'
    model = torch.load(model_path, map_location=eval_device)
    model.eval()
    model.args['batch_size'] = 4
    model.args['device'] = eval_device
    model.args['dec_type'] = 'mtp_goal_veh' # 'mtp_goal_veh' # 'mtpGoal'
    model.args['num_val_batch'] = 0

    print(model.encoder)
    #################################
    import os
    myhost = os.uname()[1]
    if myhost == 'AutoManRRCServer':
        data_path = '/disk2/SimAgent_Dataset/pyg_data_full/validation'
    elif myhost == 'asp2a-login-ntu01':
        data_path = '/disk2/SimAgent_Dataset/pyg_data_full/validation'
    else: # NSCC
        data_path = '/home/users/ntu/baichuan/scratch/sim/may07/pyg_data_full/validation'
    dataset = SimAgn_Dataset(data_path=data_path, dec_type=model.args['dec_type']) 
    loader  = DataLoader(dataset, batch_size=model.args['batch_size'], shuffle=True, num_workers=2)
    #################################

    # loss_function = mtp_goal_loss
    eval_loss = eval_a_model(model, loader, loss_func=mtp_goal_loss, plot=False, num_batch=model.args['num_val_batch'])
    print(f'eval SmoothL1 loss {eval_loss[0]}, Joint-minFDE {eval_loss[1]}, Marginal-minFDE {eval_loss[2]}')

#     // The object states for a single object through the scenario.
#     message Track {
#     enum ObjectType {
#     TYPE_UNSET = 0;  // This is an invalid state that indicates an error.
#     TYPE_VEHICLE = 1;
#     TYPE_PEDESTRIAN = 2;
#     TYPE_CYCLIST = 3;
#     TYPE_OTHER = 4;
#   }