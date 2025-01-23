# sleep_federated/federated/base.py
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import logging

class FederatedComponent:
    """Base class for federated learning components"""
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)