# 32, 26, 20, 14, 8
flops_table = {
    'resnet18': [0.03769, 0.03463, 0.02672, 0.01596, 0.01228]
}
# 224, 192, 160, 128, 96
flops_table_224 = {
    'resnet18': [1.82, 1.34, 0.93, 0.6, 0.34]
}

import torch

def get_flops_loss(gumbel_tensor, flops_list):
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
    assert len(flops_list) == gumbel_tensor.shape[-1] != 0

    n_sizes = len(flops_list)
    for j in range(n_sizes):
        flops_loss += torch.sum(gumbel_tensor[:, j] * flops_list[j])
    return flops_loss 