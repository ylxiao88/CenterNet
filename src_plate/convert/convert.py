import sys
#import __init_path
import torch
#from torch import nn
#from torch.autograd import Variable
#import torch.nn.functional as F
#import torch.utils.data
import torchvision

#sys.path.append('./demo_det/')
from opts import opts
#from self_det import CtdetDetector
from dlav0 import get_pose_net as get_dlav0
#from decode import ctdet_decode
#import numpy as np


def main(model_path, opt, name ):
    model = get_dlav0(num_layers=34, heads={'hm':1,'wh':2, 'reg':2}, head_conv=256)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model.eval()

    _input = torch.rand(1,512,512,3)
    ts = torch.jit.trace(model, _input)
    ts.save('det.pt')


if __name__ == '__main__':
    opt = opts().init()
    #model_path = '/home/pengcheng/workspace/pytorchtocaffe/example/model_centeNet_mp.pth'
    #name = 'centerNetMP'
    model_path = '/home/ubuntu/project/CenterNet/exp/ctdet/plate/model_best.pth'
    name = 'centernet'
    main(model_path=model_path, opt=opt, name=name)
