from collections import namedtuple
import torch.nn.functional as F
from collections import deque
import numpy as np
import random
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Agent:
    """
    A class to implement an Agent which interacts with and learns from the
    environment!

    Attributes:
        seed: Random seed specified to diversify results.
        memory: ReplayBuffer object which stores memories for agents.
        batch_size: An integer for minibatch size used for training.
        actor_local: The main network used to train agent behavior.
        actor_target: Target network used to stabilize agent training.
        actor_optimizer: Optimizer used when training Actor network.
        critic_local: The main critic network used to improve actor network.
        critic_target: Target network used to stabilize critic training.
        critic_optimizer: Optimizer used when training Critic network.
        noise: Random noise added to input data to improve training.
        gamma: A float designating the discount factor.
        tau: A float designating multiplication factor for soft update of
            target parameters.
    """

    def __init__(self, seed, memory, batch_size, actor_local,
                 actor_target, actor_optimizer, critic_local, critic_target,
                 critic_optimizer, noise, gamma, tau):
        """Initializes an Agent object."""

        # Initialize random seed.
        random.seed(seed)

        # Initialize Actor optimizer and networks for stabilized training.
        self.actor_local = actor_local
        self.actor_target = actor_target
        self.actor_optimizer = actor_optimizer

        # Initialize Critic optimizer and networks for stabilized training.
        self.critic_local = critic_local
        self.critic_target = critic_target
        self.critic_optimizer = critic_optimizer

        # Initialize replay memory.
        self.memory = memory
        self.batch_size = batch_size

        # Initialize noise to add to actions for bias reduction.
        self.noise = noise

        # Initialize discount factor and multiplication factor for target
        # soft update.
        self.gamma = gamma
        self.tau = tau

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""

        # Preprocess state data for action prediction.
        state = torch.from_numpy(state).float().to(device)

        # Generate action predictions based on local actor network.
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        # Add relevant noise if desired.
        if add_noise:
            action += self.noise.sample()

        return np.clip(action, -1, 1)

    def add_experience(self, states, actions, rewards, next_states, dones):
        """
        Saves experience in replay memory, and uses a random sample from
        buffer to learn.
        """

        # Save experience and reward information to replay memory.
        self.memory.add_batch(states, actions, rewards, next_states, dones)

    def step(self):
        """Initializes learning if enough memories are available."""

        # Learn if enough samples are available in memory.
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

    def learn(self, experiences, gamma):
        """
        Update policy and value parameters using batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        Where . . .
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Parameters:
            experiences: A tuple of (s, a, r, s', done) tuples.
            gamma: A float designating the discount factor.
        """

        # Extract relevant arrays from experiences.
        states, actions, rewards, next_states, dones = experiences

        # Get predicted next-state actions and Q values from target models.
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current states.
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute and minimize the loss on the local Critic network.
        Q_current = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_current, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute and minimize the loss on the local Actor network.
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the target networks based on the local network updates.
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """
        Executes a soft update on the target model parameters using the
        following equation:
            θ_target = τ * θ_local + (1 - τ) * θ_target

        Parameters:
            local_model: PyTorch model from which weights will be copied.
            target_model: PyTorch model which weights will be copied to.
            tau: A float designating the interpolation parameter.
        """

        # Soft update target model parameters.
        for target_param, local_param in \
                zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )


class ReplayBuffer:
    """
    A class to implement a fixed-size buffer to store experience tuples.

    Attributes:
        seed: Random seed specified to diversify results.
        buffer_size: An integer for replay buffer size.
        batch_size: An integer for minibatch size.
    """

    def __init__(self, seed, buffer_size, batch_size):
        """Initializes a ReplayBuffer object."""

        # Initialize variables for ReplayBuffer.
        random.seed(seed)
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            'Experience',
            field_names=['state', 'action', 'reward', 'next_state', 'done']
        )

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""

        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def add_batch(self, states, actions, rewards, next_states, dones):
        """Add a new experience batch to memory."""

        for state, action, reward, next_state, done in \
                zip(states, actions, rewards, next_states, dones):
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)

    def create_tensor(self, data):
        """Formats data as a tensor for model training."""

        return torch.from_numpy(np.vstack(data)).float().to(device)

    def sample(self):
        """Randomly samples a batch of experiences from memory."""

        # Randomly sample experiences and extract relevant variables.
        experiences = random.sample(self.memory, k=self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        # Reformat arrays as tensors for training.
        states = self.create_tensor(states)
        actions = self.create_tensor(actions)
        rewards = self.create_tensor(rewards)
        next_states = self.create_tensor(next_states)
        dones = self.create_tensor(dones)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Returns the current size of internal memory."""

        return len(self.memory)


class GaussianNoise:
    """
    A class to implement GaussianNoise used on the actions in order
    to reduce generalization error during training.

    Attributes:
        size: Integer of dimensionality of the action space.
        seed: Integer random seed specified to diversify results.
        mu: Float mean or expected value of action distribution.
        sigma: Float standard deviation value which designates how much noise
            we would like to add to our chosen action values.
    """
    def __init__(self, size, seed, mu, sigma):
        """Initialize parameters and noise process."""

        # Initialize relevant variables for distribution.
        random.seed(seed)
        self.size = size
        self.mu = mu
        self.sigma = sigma

    def sample(self):
        """
        Returns noise sampled from a Normal distribution with specified mu and
        sigma values.
        """

        return np.random.normal(self.mu, self.sigma, self.size)