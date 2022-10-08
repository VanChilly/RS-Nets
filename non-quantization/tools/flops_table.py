# 224, 192, 160, 128, 96
flops_table = {
    'resnet18': [1.82, 1.34, 0.93, 0.6, 0.34]
}

def get_flops_loss(gumbel_tensor, flops_list, loss_gamma):
    """Get flops loss based on gumbel_tensor

    Parameters
    ----------
    gumbel_tensor : torch.Tensor
        [N, reso_sizes]
    flops_list : List
        flops_list for a certain network
    loss_gamma : float
        weight for flops loss
    """
    

