import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch


def hidden_init(layer):
    """
    Provides limits for Uniform distribution which reinitializes parameters
    for neural network layers.

    Args:
        layer: Neural network layer to be reinitialized.

    Returns:
        limits: Upper and lower limits used for Uniform distribution.
    """

    # Calculate limits for Uniform distribution sampling.
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class Actor(nn.Module):
    """
    Initializes an Actor (Policy) Model.

    Arguments:
        state_size: An integer count of dimensions for each state.
        action_size: An integer count of dimensions for each action.
        seed: An integer random seed.
        fc1_units: An integer number of nodes in first hidden layer.
        fc2_units: An integer number of nodes in second hidden layer.
        state: An instance of state gathered for the environent.
    """

    def __init__(self, state_size, action_size, seed, fc1_units, fc2_units):
        """Initializes parameters and builds model."""

        # Initialize inheritance and relevant variables.
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.bn1 = nn.LayerNorm(fc1_units)
        self.bn2 = nn.LayerNorm(fc2_units)
        self.bn3 = nn.LayerNorm(action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitializes the parameters of each hidden layer."""

        # Reinitialize parameters for each NN layer.
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Builds an actor (policy) network that maps states to actions."""

        # Build Actor neural network architecture.
        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = torch.tanh(self.fc3(x))
        x = self.bn3(x) 
        return x


class Critic(nn.Module):
    """
    Initializes a Critic (Value) Model.

    Arguments:
        state_size: An integer count of dimensions for each state.
        action_size: An integer count of dimensions for each action.
        seed: An integer random seed.
        fc1_units: An integer number of nodes in first hidden layer.
        fc2_units: An integer number of nodes in second hidden layer.
        state: An instance of state gathered for the environent.
        action: An instance of action predicted by the corresponding
            Actor model.
    """

    def __init__(self, state_size, action_size, seed, fc1_units, fc2_units):
        """Initializes parameters and builds model."""

        # Initialize inheritance and relevant variables.
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.bn1 = nn.LayerNorm(fc1_units)
        self.bn2 = nn.LayerNorm(fc2_units)
        self.bn3 = nn.LayerNorm(1)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitializes the parameters of each hidden layer."""

        # Reinitialize parameters for each NN layer.
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Builds a critic (value) network that maps (state, action) pairs
        to Q-values.
        """

        # Build Critic neural network architecture.
        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        return x