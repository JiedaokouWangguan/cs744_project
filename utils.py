import torch
from struct import pack, unpack


class MessageCode(object):
    """Different types of messages between client and server that we support go here."""
    PullTilde = 0
    UpdateTilde = 1
    WorkerTerminate = 2
    ParameterRequest = 0
    GradientUpdate = 1


def quantize_tensor(x, num_bits):
    qmin = 0.
    qmax = 2.**num_bits - 1.
    min_val, max_val = x.min(), x.max()
    scale = (max_val - min_val) / (qmax - qmin)
    scale = 1 if scale == 0 else scale
    initial_zero_point = qmin - min_val / scale

    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()

    b = pack('f', scale)
    c = unpack('I', b)[0]
    x1 = 0x000000FF & c
    x2 = (0X0000FF00 & c) >> 8
    x3 = (0x00FF0000 & c) >> 16
    x4 = (0xFF000000 & c) >> 24

    m_parameter = torch.Tensor([x1, x2, x3, x4, zero_point])
    m_parameter = torch.cat((q_x, m_parameter))

    m_parameter = m_parameter.round().byte()

    return m_parameter


def dequantize_tensor(m_parameter):

    scale1 = int(m_parameter[-5].item())
    scale2 = int(m_parameter[-4].item())
    scale3 = int(m_parameter[-3].item())
    scale4 = int(m_parameter[-2].item())

    scale = 0
    scale = scale1 | scale
    scale = (scale2 << 8) | scale
    scale = (scale3 << 16) | scale
    scale = (scale4 << 24) | scale

    scale = pack('I', scale)
    scale = unpack('f', scale)[0]
    zero_point = int(m_parameter[-1].item())
    q_x = m_parameter[:-5]
    return scale * (q_x.float() - zero_point)
