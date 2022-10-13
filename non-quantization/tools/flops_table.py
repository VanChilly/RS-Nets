# 32, 26, 20, 14, 8
flops_table = {
    'resnet18': [0.03769, 0.03463, 0.02672, 0.01596, 0.01228],
    'resnet20': [0.04158, 0.02884, 0.01624, 0.00873, 0.00260],
}

# 224, 192, 160, 128, 96
flops_table_224 = {
    'resnet18': [1.82, 1.34, 0.93, 0.6, 0.34]
}

import torch

def get_flops_loss(gumbel_tensor, flops_list, alpha=0.03, loss_type='DRNet'):
    """Get flops loss based on gumbel_tensor
    Implementation of Paper 
    "Dynamic Low-Resolution Distillation for Cost-Efficient End-to-End Text Spotting"

    Parameters
    ----------
    gumbel_tensor : torch.Tensor
        [N, reso_sizes]
    flops_list : List
        flops_list for a certain network
    loss_gamma : float
        weight for flops loss
    """
    flops_loss = 0
    # print(f"Decision Prob: {gumbel_tensor[0].data}")
    assert len(flops_list) == gumbel_tensor.shape[-1] != 0

    n_sizes = len(flops_list)
    for j in range(n_sizes):
        flops_loss += torch.sum(gumbel_tensor[:, j] * flops_list[j])
        
    if loss_type == 'DRNet':
        # DRNet
        E_F = flops_loss / gumbel_tensor.shape[0]
        zero = torch.tensor(0).to(gumbel_tensor.device)
        L_reg = torch.max(zero, (E_F - alpha) / (max(flops_list) - min(flops_list)))
        return L_reg
    elif loss_type == 'DLD':
        return flops_loss
    else:
        raise NotImplementedError(f"Unknown Loss type: {loss_type}")