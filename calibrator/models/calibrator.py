import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from .util import init_weights
from .models import register_model, get_model
from .backbones import define_G
@register_model('Calibrator')
class Calibrator(nn.Module):
    "Defines and Calibrator Network."

    def __init__(self, cfg):
        super(Calibrator, self).__init__()
        self.name = 'Calibrator'
        self.arch = cfg['CALIBRATOR']['ARCH']
        self.width_factor = cfg['CALIBRATOR']['CALI_WIDTH_FACTOR']
        self.L_infty_norm = cfg['CALIBRATOR']['L_INFTY_NORM']
        self.norm_type = cfg['CALIBRATOR']['NORM_TYPE']
        self.setup_net()

    def forward(self, x_t):
        """Pass source and target images through their
        respective networks."""

        pert = self.cali(x_t)

        # because output of calibrator is within (-1,1)
        # we change pert range from (-1,1) to (0,1)

        pert = (pert + 1) / 2        

        pert  = torch.clamp(pert, 0, self.L_infty_norm)
        
        return pert

    def setup_net(self):
        """Setup source, target and discriminator networks."""

        self.output_nc = 3

        self.gen_input_nc = 3

        
        #We might need stronger calibrator. But none of them works well yet
        
        self.cali = define_G(self.gen_input_nc,self.output_nc,self.width_factor,self.arch,norm = self.norm_type)   



    def load(self, init_path):
        "Loads full src and tgt models."
        net_init_dict = torch.load(init_path)
        self.load_state_dict(net_init_dict)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)


if __name__ == '__main__':
    print(networks)
