#
"""
Parameter server for distbelief
"""
import logging
import torch
import torch.optim
import torch.distributed as dist
from ..utils import MessageCode
from ..utils import dequantize_tensor, quantize_tensor


_LOGGER = logging.getLogger(__name__)


class ParameterServer(object):

    def __init__(self, model, world_size, quantize_num_bits=0):
        _LOGGER.info("Creating ParameterServer")
        self.running = True
        self.model = model
        self.world_size = world_size
        self.num_terminate = 0
        self.quantize_num_bits = quantize_num_bits
        self.parameter_shard = torch.randn(self.squash_model(self.model).numel())

    def start(self):
        _LOGGER.info("Started Running!")
        while self.running:
            _LOGGER.info("Polling for message...")
            m_parameter = torch.zeros(self.squash_model(self.model).numel() + 7).to(torch.int16)
            print("1")
            dist.recv(tensor=m_parameter)
            print("2")
            m_parameter = dequantize_tensor(m_parameter)
            self.receive(int(m_parameter[0].item()),
                         int(m_parameter[1].item()),
                         m_parameter[2:])
            print("3")
            if self.num_terminate == self.world_size-1:
                self.running = False
        print("parameter server terminated.")

    def receive(self, sender, message_code, parameter):
        _LOGGER.info("Processing message: {} from sender {}".format(message_code, sender))

        if message_code == MessageCode.PullTilde:
            self.send_message(MessageCode.PullTilde, self.parameter_shard, dst=sender)

        elif message_code == MessageCode.UpdateTilde:
            self.parameter_shard.add_(parameter)
        elif message_code == MessageCode.WorkerTerminate:
            self.num_terminate += 1

    def send_message(self, message_code, payload, dst=0):
        """Sends a message to a destination
        Concatenates both the message code and destination with the payload into a single tensor and then sends that as a tensor
        """
        _LOGGER.info("SENDING MESSAGE: {} RANK: {}".format(message_code, dist.get_rank()))
        m_parameter = torch.Tensor([dist.get_rank(), message_code])
        m_parameter = torch.cat((m_parameter, payload))
        m_parameter = quantize_tensor(m_parameter, self.quantize_num_bits)
        print("----------bs")
        dist.send(tensor=m_parameter, dst=dst)
        print("----------as")

    @staticmethod
    def squash_model(model):
        """
        Squash model parameters into a single tensor.
        """
        m_parameter = torch.Tensor([0])
        for parameter in list(model.parameters()):
            m_parameter = torch.cat((m_parameter, parameter.data.view(-1)))
        return m_parameter[1:]
