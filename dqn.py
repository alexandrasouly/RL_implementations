from dataclasses import dataclass
from torch import nn
import numpy as np


def make_env(env_name):
    pass
    return env


def preprocess_frames(frame):
    pass
    return frame


def stack_frames(frames):
    pass
    return stacked_frames


@dataclass
class HyperParams(object):
    pass


class DQN(object):
    def __init__(self, input_shape) -> None:
        """
        CNN network from 2015 paper Human-level control through deep reinforcement learning
        https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
        """

        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(????, 512),  ##TODO stable-baselinesV3 probably missed of this step???
            nn.ReLU(),
            nn.Linear(512, n_features),
        )

    def calc_conv_layer_size(size, stride, filter):
        return np.floor((size-filter)/stride) + 1


class Memory(object):
    def __init__(self) -> None:
        self._prepopulate_memory()

    def _prepopulate_memory(self):
        pass

    def add(self):
        pass

    def sample(self):
        pass


def choose_action():
    pass


def train():
    pass


def test():
