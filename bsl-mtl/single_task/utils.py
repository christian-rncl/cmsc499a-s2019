'''
Stolen from 
https://github.com/LaceyChen17/neural-collaborative-filtering/blob/master/src/utils.py
'''
import torch

def use_optimizer(network, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(), lr = params['sgd_lr'],  momentum=params['sgd_momentum'], weight_decay=1e-5)
    elif params['optimizer'] == 'adam':
        # optimizer = torch.optim.sparseAdam(network.parameters(), lr=params['adam_lr'], weight_decay=params['l2_regularization'])
        optimizer = torch.optim.sparseAdam(network.parameters())
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer


def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)
