import torch
from utils import dequantize_tensor, quantize_tensor
from struct import pack, unpack
from torchvision import datasets, transforms
import logging
import copy
import torch
from torch.optim.optimizer import Optimizer, required
import torch.distributed as dist


def quantize(x, num_bits):
    qmin = 0.
    qmax = 2.**num_bits - 1.
    min_val, max_val = x.min(), x.max()
    scale = (qmax - qmin) / (max_val - min_val)
    scale = 1 if scale == 0 else scale
    print(scale)
    q_x = qmin + (x - min_val) * scale
    q_x.clamp_(qmin, qmax).round_()

    b = pack('f', scale)
    c = unpack('i', b)[0]
    x1 = 0x000000FF & c
    x2 = (0X0000FF00 & c) >> 8
    x3 = (0x00FF0000 & c) >> 16
    x4 = (0xFF000000 & c) >> 24

    m_parameter = torch.Tensor([x1, x2, x3, x4, min_val])
    m_parameter = torch.cat((q_x, m_parameter))
    m_parameter = m_parameter.round().to(torch.int16)
    return m_parameter


def dequantize(m_parameter):
    scale1 = int(m_parameter[-5].item())
    scale2 = int(m_parameter[-4].item())
    scale3 = int(m_parameter[-3].item())
    scale4 = int(m_parameter[-2].item())

    scale = 0
    scale = scale1 | scale
    scale = (scale2 << 8) | scale
    scale = (scale3 << 16) | scale
    scale = (scale4 << 24) | scale
    scale = pack('i', scale)
    scale = unpack('f', scale)[0]
    print(scale)
    min_val = int(m_parameter[-1].item())
    q_x = m_parameter[:-5]
    result = q_x.float() / scale + min_val
    return result




shabi = datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
                       )

fuck = len(shabi)
print(fuck)
fuck1 = torch.utils.data.random_split(shabi, [40000, 20000])
print(len(fuck1[0]))
print(len(fuck1[1]))

a = 1
