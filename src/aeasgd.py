import logging
import copy
import torch
from torch.optim.optimizer import Optimizer, required
import torch.distributed as dist
from utils import MessageCode

_LOGGER = logging.getLogger(__name__)


class AEASGD(Optimizer):
    """DownpourSGD"""

    def __init__(self, params, lr=required, tau=required, rho=required, model=required):

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, tau=tau, rho=rho)

        self.model = model
        # this sets the initial model parameters
        self.idx = 0
        super(AEASGD, self).__init__(params, defaults)

    @staticmethod
    def send_message(message_code, payload, dst=0):
        """Sends a message to a destination
        Concatenates both the message code and destination with the payload into a single tensor and then sends that as a tensor
        """
        _LOGGER.info("SENDING MESSAGE: {} RANK: {}".format(message_code, dist.get_rank()))
        m_parameter = torch.Tensor([dist.get_rank(), message_code.value])
        m_parameter = torch.cat((m_parameter, payload))
        dist.send(tensor=m_parameter, dst=dst)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # send parameter request every N iterations
        if self.idx % self.param_groups[0]['tau'] == 0:
            self.idx = 0
            self.send_message(MessageCode.PullTilde, torch.zeros(self.squash_model(self.model).numel()))

            # pull x tilde
            m_parameter = torch.zeros(self.squash_model(self.model).numel() + 2)
            dist.recv(tensor=m_parameter)

            # build alpha term
            current_index = 0  # keep track of where to read from parameter_update
            delta = copy.deepcopy(self.model)
            alpha = self.param_groups[0]['rho'] * self.param_groups[0]['lr']
            for parameter in delta.parameters():
                numel = parameter.data.numel()
                size = parameter.data.size()
                parameter.data.add_(-1, m_parameter[current_index:current_index + numel].view(size))
                parameter.data.mul_(alpha)
                current_index += numel
            # delta = delta * self.param_groups[0]['rho'] * self.param_groups[0]['lr']

            # update x
            for cur_parameter, cur_delta in zip(self.model.parameters(), delta.parameters()):
                cur_parameter.data.add_(-1, cur_delta.data)

            # push delta to update x tilde
            self.send_message(MessageCode.UpdateTilde, self.squash_model(delta))

        else:
            self.idx += 1

        # internal sgd update
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(-group['lr'], d_p)

        return loss

    @staticmethod
    def squash_model(model):
        """
        Squash model parameters into a single tensor.
        """
        m_parameter = torch.Tensor([0])
        for parameter in list(model.parameters()):
            m_parameter = torch.cat((m_parameter, parameter.data.view(-1)))
        return m_parameter[1:]
