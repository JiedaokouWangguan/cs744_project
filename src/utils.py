from enum import Enum


class MessageCode(Enum):
    """Different types of messages between client and server that we support go here."""
    PullTilde = 0
    UpdateTilde = 1