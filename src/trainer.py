import torch

class Trainer:
    def __init__(self, config: dict):
        self.__dict__ |= config
        self.config = config


