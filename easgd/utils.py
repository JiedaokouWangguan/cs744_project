import torch


class MessageCode(object):
    """Different types of messages between client and server that we support go here."""
    PullTilde = 0
    UpdateTilde = 1


def quantize_tensor(x, num_bits=8):
    qmin = 0.
    qmax = 2.**num_bits - 1.
    min_val, max_val = x.min(), x.max()
    print("min: {}, max: {}".format(min_val, max_val))
    scale = (max_val - min_val) / (qmax - qmin)
    print("scale: {}".format(scale))
    initial_zero_point = qmin - min_val / scale
    print("init_zero_point: {}".format(initial_zero_point))
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
    q_x = q_x.round().byte()

    m_parameter = torch.Tensor([scale, zero_point])
    m_parameter = torch.cat((m_parameter, q_x))

    return m_parameter


def dequantize_tensor(m_parameter):
    scale = float(m_parameter[0].item())
    zero_point = int(m_parameter[1].item())
    q_x = m_parameter[2:]
    return scale * (q_x.float() - zero_point)
