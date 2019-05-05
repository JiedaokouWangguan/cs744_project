#
"""
Parameter server for distbelief
"""
import logging
import torch
import torch.optim
import torch.distributed as dist
from utils import MessageCode

_LOGGER = logging.getLogger(__name__)


class ParameterServer(object):

    def __init__(self, model):
        _LOGGER.info("Creating ParameterServer")
        self.running = True
        self.model = model
        self.parameter_shard = torch.randn(self.squash_model(self.model).numel())
        self.m_parameter = torch.zeros(self.squash_model(self.model).numel() + 2)

    def start(self):
        _LOGGER.info("Started Running!")
        while self.running:
            _LOGGER.info("Polling for message...")
            dist.recv(tensor=self.m_parameter)
            self.receive(int(self.m_parameter[0].item()),
                         MessageCode(self.m_parameter[1].item()),
                         self.m_parameter[2:])

    def receive(self, sender, message_code, parameter):
        _LOGGER.info("Processing message: {} from sender {}".format(message_code.name, sender))

        if message_code == MessageCode.PullTilde:
            self.send_message(MessageCode.PullTilde, self.parameter_shard, dst=sender)

        elif message_code == MessageCode.UpdateTilde:
            self.parameter_shard.add_(parameter)

    @staticmethod
    def send_message(message_code, payload, dst=0):
        """Sends a message to a destination
        Concatenates both the message code and destination with the payload into a single tensor and then sends that as a tensor
        """
        _LOGGER.info("SENDING MESSAGE: {} RANK: {}".format(message_code, dist.get_rank()))
        m_parameter = torch.Tensor([dist.get_rank(), message_code.value])
        m_parameter = torch.cat((m_parameter, payload))
        dist.send(tensor=m_parameter, dst=dst)

    @staticmethod
    def squash_model(model):
        """
        Squash model parameters into a single tensor.
        """
        m_parameter = torch.Tensor([0])
        for parameter in list(model.parameters()):
            m_parameter = torch.cat((m_parameter, parameter.data.view(-1)))
        return m_parameter[1:]
