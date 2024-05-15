## 将 Batch 中所有的 goals 按照 scenario 分成 goalsets
import torch
def convert_goalsBatch_to_goalsets_for_scenarios(goals_in_Batch, num_goal_valid_agn):
    num_agn_per_scenario = torch.cat((torch.tensor([0], device=goals_in_Batch.device),num_goal_valid_agn))
    start_and_end_index = torch.cumsum(num_agn_per_scenario, dim=0)
    Goal_Sets = []
    for i in range(len(start_and_end_index)-1):
        start_idx, end_idx = start_and_end_index[i], start_and_end_index[i+1]
        goalset_in_scenario = goals_in_Batch[start_idx:end_idx]
        # print(f'goalset_in_scenario {goalset_in_scenario.shape}')
        Goal_Sets.append(goalset_in_scenario)
    return Goal_Sets

## 将输入数据归一化一下，不至于太大
def normlize_Agn_seq(agn_seq):
    """ agn_seq: [11, 6] , (x,y,z,vx,vy,h) """
    agn_seq[0:3] = agn_seq[0:3] / 200
    agn_seq[3:5] = agn_seq[3:5] / 30
    agn_seq[5: ] = agn_seq[5: ] / 3.14
    return agn_seq

def normlize_CCL_seq(ccl_seq):
    """ ccl_seq: [11, 4], () """
    ccl_seq[0:2] = ccl_seq[0:2] / 200
    ccl_seq[2:4] = ccl_seq[2:4] / 3.14
    return ccl_seq