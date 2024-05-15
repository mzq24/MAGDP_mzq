import torch
from shapely.geometry import LineString, Point, Polygon
import numpy as np

def get_agn_ccl_mask(agn_name, node_name_list):
    """
    return agn_ccl_mask, agn_ccl_index, agn_ccl_names
    """
    agn_ccl_mask = torch.tensor([True if node_name.startswith(f'{agn_name}CCL') or node_name.startswith(f'{agn_name}TCL')\
                    else False \
                    for node_name in node_name_list ])
    agn_ccl_names = node_name_list[agn_ccl_mask]
    agn_ccl_index = agn_ccl_mask.nonzero().squeeze(-1)
    return agn_ccl_mask, agn_ccl_index, agn_ccl_names

def calculate_ccl_dists_to_goal(agn_goal, pyg_node_feature, agn_ccl_mask):
    agn_CCLs = pyg_node_feature[agn_ccl_mask].detach().cpu().numpy()
    # print(f'agn_goal {agn_goal.shape}')
    CCLs_dist_to_agn = torch.tensor([LineString(ccl[:,:2]).distance(Point(agn_goal.detach().cpu().numpy())) for ccl in agn_CCLs])
    return CCLs_dist_to_agn

def get_agn_tcl_index(agn_name, agn_goal, pyg_node_name_list, pyg_node_feature):
    """
    agn_name 是 scenario id + agn id，不然不同的 scenario 中 agent id 会重复。
    """
    agn_ccl_mask, agn_ccl_index, _ = get_agn_ccl_mask(agn_name, pyg_node_name_list)
    ccl_dists = calculate_ccl_dists_to_goal(agn_goal, pyg_node_feature, agn_ccl_mask)
    agn_tcl_index = agn_ccl_index[torch.argmin(ccl_dists)].item()
    return agn_tcl_index

def get_agn_names(pyg_batch_node_name_list):
    """
    列出一个 Batch 中所有的 node_name
    """
    return [node_name for node_name in pyg_batch_node_name_list if 'CL' not in node_name ]
    
def get_batch_tcl_indexes(pyg_batch_node_name_list, predicted_agn_goal_set, batch_pyg_node_feature):
    """
    给定预测出的 Agents Joint Goalset, 找出他们对应的 TCLs 的 indexes
    """
    agn_names_in_batch = get_agn_names(pyg_batch_node_name_list)
    assert len(agn_names_in_batch) == predicted_agn_goal_set.shape[0], f'{len(agn_names_in_batch)}  {predicted_agn_goal_set.shape[0]}'

    tcl_indexes = []
    for i in range(len(agn_names_in_batch)):
        tcl_idx = get_agn_tcl_index(agn_names_in_batch[i], predicted_agn_goal_set[i], pyg_batch_node_name_list, batch_pyg_node_feature)
        tcl_indexes.append(tcl_idx)
    return tcl_indexes

def set_TCL_of_an_agent(agn_name, flat_data):
    """ 
    给一个 Agent 随机设定 TCL
    先把 agend id 对应的 CCL 都设置为 CCL (tarCCL), 然后再随机选一个作为 TCL (tarTCL). 
    """   
    agn_ccl_mask, agn_ccl_index, agn_ccl_names = get_agn_ccl_mask(agn_name, flat_data.flat_node_name)

    flat_data.flat_node_type[agn_ccl_mask] = 'tarCCL'
    # print(f'before tcl assignment {pyg_batch_node_type_list}')
    ## 从 agn_ccl_index 中 sample 一个出来，设置为 tarTCL
    # print(f'agn_ccl_index {agn_ccl_index}')
    tcl_idx = np.random.choice(agn_ccl_index.numpy())
    # print(f'tcl_idx {tcl_idx}')
    flat_data.flat_node_type[tcl_idx] = 'tarTCL'
    # print(f'after tcl assignment {pyg_batch_node_type_list}')
    return flat_data

def get_agents_with_many_CCLs(pyg_batch_node_name_list):
    agn_names = get_agn_names(pyg_batch_node_name_list)
    agn_name_to_CCL_mask = {}
    for ag_name in agn_names:
        ag_ccl_mask, _, _ = get_agn_ccl_mask(ag_name, pyg_batch_node_name_list)
        # print(f'ag_ccl_mask {ag_ccl_mask}')
        ## 只把 #CCL >2 的加进来
        if sum(ag_ccl_mask)>=2:
            agn_name_to_CCL_mask[ag_name] = ag_ccl_mask
    return agn_name_to_CCL_mask

def sample_2d_gaussian(X, Y, SigmaX, SigmaY, Rho, num_points):
    # Step 1: Generate two sets of independent samples from a standard normal distribution
    Z1 = np.random.normal(size=num_points)
    Z2 = np.random.normal(size=num_points)

    # Step 2: Calculate intermediate variables
    X_samples = X + SigmaX * Z1
    Y_samples = Y + SigmaY * (Rho * Z1 + np.sqrt(1 - Rho**2) * Z2)

    # Return the sampled points as arrays
    return X_samples, Y_samples

def sample_points_from_gaussians(gaussians, num_points=1):
    """
    gaussians: [#Agent, 5] 
    可以考虑一次性采样32组，然后对每一组来只改变 TCL 有变化的。
    """
    sampled_points = []
    
    for i in range(gaussians.shape[0]):
        X, Y , SigmaX, SigmaY, Rho = gaussians[i]
        X_sample, Y_sample = sample_2d_gaussian(X, Y, SigmaX, SigmaY, Rho, num_points)
        sampled_points.append((X_sample[0], Y_sample[0]))
    return np.array(sampled_points)