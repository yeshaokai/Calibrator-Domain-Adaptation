import torch

models = {}


def register_model(name):
    #print('resgitered {}'.format(name)) 

    def decorator(cls):
        models[name] = cls
        return cls

    return decorator


def get_model(name, **kwargs):
    net = models[name](**kwargs)
    if torch.cuda.is_available():
        net = net.cuda()
    return net
