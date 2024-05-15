import torch
from dataset_may17 import SimAgn_Dataset, flat_Batch, retrieve_mask
from tqdm import tqdm

def eval_a_model(model_to_eval, eval_loader, loss_func, num_batch=-1):
    model_to_eval.eval()
    running_loss = 0.0
    running_ade  = 0.0
    running_fde  = 0.0
    for d_idx, data in tqdm(enumerate(eval_loader)):
    # for d_idx, data in enumerate(eval_loader):
        data = flat_Batch(data)
        data = data.to(model_to_eval.args['device'])

        # forward + loss calculation
        fut_pred = model_to_eval(data)
        mot_loss, ade, fde, _ = loss_func(fut_pred, data.mot_label, eval=True)

        # print statistics
        running_loss += mot_loss.item()
        running_ade  += ade.item()
        running_fde  += fde.item()
        # running_loss.append(batch_loss.item())
        if num_batch>-1 and d_idx>=num_batch:
            break
    # print(round(running_loss/(i+1),2))
    return round(running_loss/(d_idx+1),2), round(running_ade/(d_idx+1),2), round(running_fde/(d_idx+1),2)

# if __name__ == '__main__':
#     import os
#     from torch_geometric.loader import DataLoader

    # eval_device = 'cpu'
    # model_path = './models/gdp-EP9-Loss0.55.ckpt'
    # model = torch.load(model_path, map_location=eval_device)
    # model.eval()
    # model.args['device'] = eval_device
    # model.args['eval_num_batch'] = 50
    # print(model.encoder)
    # #################################
    # myhost = os.uname()[1]
    # if myhost == 'AutoManRRCServer':
    #     data_path = '/disk2/SimAgent_Dataset/pyg_data_full/validation'
    # elif myhost == 'asp2a-login-ntu01':
    #     data_path = '/disk2/SimAgent_Dataset/pyg_data_full/validation'
    # else: # NSCC
    #     data_path = '/home/users/ntu/baichuan/scratch/sim/may07/pyg_data_full/validation'

    # val_set = SimAgn_Dataset(data_path=data_path, dec_type=model.args['dec_type']) 
    # print('val_size: {}/{}'.format(0, val_set.__len__()))
    # valDataloader = DataLoader(val_set, batch_size=model.args['batch_size'], shuffle=True, num_workers=2)
    # #################################
    # gdp_loss_func = torch.nn.SmoothL1Loss(reduction='mean') # SmoothL1Loss MSELoss
    # eval_loss = eval_a_model(model, valDataloader, gdp_loss_func, num_batch=model.args['eval_num_batch'])
    # print(f'eval_loss {eval_loss} ')