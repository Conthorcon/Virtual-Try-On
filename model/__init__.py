import torch.nn as nn
import torch
from .cpvton import GMM, UnetGenerator

def build_model(opt):
    gmm = GMM(opt)
    tom = UnetGenerator(26, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d) 

    gmm.load_state_dict(torch.load(opt.gmm_checkpoint))
    tom.load_state_dict(torch.load(opt.tom_checkpoint))

    return gmm, tom


