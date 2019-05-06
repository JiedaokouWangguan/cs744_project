import logging
import torch
from torch.optim.optimizer import Optimizer, required
import torch.distributed as dist
from utils import MessageCode

_LOGGER = logging.getLogger(__name__)


class DownPourSGD(Optimizer):
    """DownpourSGD"""

    def __init__(self, params, lr=required, n_push=required, n_pull=required, rho=required, model=required):

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, n_push=n_push, n_pull=n_pull, rho=rho)
        self.model = model
        self.accumulated_gradients = self.squash_model(self.model)
        # this sets the initial model parameters
        self.idx = 0
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

        if self.idx % self.param_groups[0]['n_pull'] == 0:
            self.send_message(MessageCode.ParameterRequest, self.accumulated_gradients)

        # keep track of accumulated gradients so that we can send
        gradients = self.squash_model(self.model, grad=True)
        self.accumulated_gradients.add_(-self.param_groups[0]['lr'], gradients)

        # send parameter request every N iterations
        if self.idx % self.param_groups[0]['n_push'] == 0:
            self.send_message(MessageCode.GradientUpdate, self.accumulated_gradients)
            self.accumulated_gradients.zero_()

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

