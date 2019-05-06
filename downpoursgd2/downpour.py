import logging
import torch
from torch.optim.optimizer import Optimizer, required
import torch.distributed as dist
from utils import MessageCode

_LOGGER = logging.getLogger(__name__)


class DownPourSGD(Optimizer):
    """DownpourSGD"""

    def __init__(self, params, lr=required, n_push=required, n_pull=required, model=required):

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, n_push=n_push, n_pull=n_pull)
        self.model = model
        self.accumulated_gradients = torch.zeros(self.squash_model(self.model).numel())
        # this sets the initial model parameters
        self.idx = 0
        self.lr = lr
        self.n_push = n_push
        self.n_pull = n_pull
        super(DownPourSGD, self).__init__(params, defaults)

    @staticmethod
    def send_message(message_code, payload, dst=0):
        """Sends a message to a destination
        Concatenates both the message code and destination with the payload into a single tensor and then sends that as a tensor
        """
        _LOGGER.info("SENDING MESSAGE: {} RANK: {}".format(message_code, dist.get_rank()))
        m_parameter = torch.Tensor([dist.get_rank(), message_code])
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

        if self.idx % self.n_pull == 0:
            self.send_message(MessageCode.ParameterRequest, self.accumulated_gradients)

            m_parameter = torch.zeros(self.squash_model(self.model).numel() + 2)
            dist.recv(tensor=m_parameter)

            # build alpha term
            current_index = 2
            for parameter in self.model.parameters():
                numel = parameter.data.numel()
                size = parameter.data.size()
                parameter.data.copy_(m_parameter[current_index:current_index + numel].view(size))
                current_index += numel

        # keep track of accumulated gradients so that we can send
        gradients = self.squash_model(self.model, grad=True)
        self.accumulated_gradients.add_(-self.lr, gradients)

        # send parameter request every N iterations
        if self.idx % self.n_push == 0:
            self.send_message(MessageCode.GradientUpdate, self.accumulated_gradients)
            self.accumulated_gradients.zero_()

        self.idx += 1

        # internal sgd update
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(-self.lr, d_p)

        return loss

    @staticmethod
    def squash_model(model, grad=False):
        """
        Squash model parameters into a single tensor.
        """
        m_parameter = torch.Tensor([0])
        for parameter in list(model.parameters()):
            if grad:
                m_parameter = torch.cat((m_parameter, parameter.grad.view(-1)))
            else:
                m_parameter = torch.cat((m_parameter, parameter.data.view(-1)))
        return m_parameter[1:]

