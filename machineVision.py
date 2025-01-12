import gymnasium as gym
import ale_py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
from scipy.ndimage import zoom
import cv2
from base64 import b64encode
import time
from torchvision import datasets, models
import os
from typing import Dict, List, Tuple
import time
import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import cv2

env = gym.make("ALE/Boxing-v5", render_mode="rgb_array")

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: tuple, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, 2, 210,160], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, 2, 210,160], dtype=np.float32)
        self.obs_buf = np.zeros([size, 2, 105,80], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, 2, 105,80], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size

from torchvision.models.alexnet import alexnet

class Network(nn.Module):

    def reset_network(self):
      self.hidden = torch.zeros(1, self.hidden.shape[-1]).to(device)
      self.cell = torch.zeros(1, self.hidden.shape[-1]).to(device)

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        atom_size: int,
        support: torch.Tensor
    ):
        """Initialization."""
        super(Network, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten()
        )
        self.hidden = torch.zeros( 1, 128).to(device)
        self.cell = torch.zeros( 1, 128).to(device)
        self.out_dim = out_dim
        self.atom_size = atom_size
        self.support = support.to(device)
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2816, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim * atom_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        if(len(x.shape) == 3):
          x = x.unsqueeze(0)
        if(x.shape[1] != 3):
          x.permute(0,3,1,2)
        x = x.to(device)
        x = self.conv(x)
        q_atoms = self.layers(x).view(-1, self.out_dim, self.atom_size)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans

        return dist
    def reset(self):
      self.hidden = torch.zeros(1, self.hidden.shape[-1]).to(device)
      self.cell = torch.zeros(1, self.hidden.shape[-1]).to(device)

class DQNAgent:
    """DQN Agent interacting with environment.

    Attribute:
        env (gym.Env): openAI Gym environment
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        epsilon (float): parameter for epsilon greedy policy
        epsilon_decay (float): step size to decrease epsilon
        max_epsilon (float): max value of epsilon
        min_epsilon (float): min value of epsilon
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including
                           state, action, reward, next_state, done
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
        support (torch.Tensor): support for categorical dqn
    """

    def __init__(
        self,
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        epsilon_decay: float,
        seed: int,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.2,
        gamma: float = 0.99,
        # Categorical DQN parameters
        v_min: float = 0.0,
        v_max: float = 200.0,
        atom_size: int = 51,
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            epsilon_decay (float): step size to decrease epsilon
            lr (float): learning rate
            max_epsilon (float): max value of epsilon
            min_epsilon (float): min value of epsilon
            gamma (float): discount factor
            v_min (float): min value of support
            v_max (float): max value of support
            atom_size (int): the unit number of support
        """
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self.env = env
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.seed = seed
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma
        self.reset_countdown = int(1/self.epsilon_decay)
        self.state = None

        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        # networks: dqn, dqn_target
        self.dqn = Network(
            obs_dim, action_dim, atom_size, self.support
        ).to(self.device)
        self.dqn_target = Network(
            obs_dim, action_dim, atom_size, self.support
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.001)

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # epsilon greedy policy
        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.dqn(
                torch.FloatTensor(state).to(self.device),
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        if not self.is_test:
            self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        next_state = self.env.render()
        next_state = self.rgb2gray(next_state)
        next_state = cv2.resize(next_state, (0, 0), fx = 0.5, fy = 0.5)
        next_state = np.expand_dims(next_state, axis=0)
        next_state = np.concatenate((next_state, self.state[1:,:,:]), axis = 0)
        reward = np.clip(reward, -1.0, 1.0)

        done = terminated or truncated

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)

        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.memory.sample_batch()
        self.dqn.reset()
        self.dqn_target.reset()
        loss = self._compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, num_frames: int, plotting_interval: int = 10000):
        """Train the agent."""
        self.is_test = False

        self.state, _ = self.env.reset(seed=self.seed)
        self.state = self.env.render()
        self.state = self.rgb2gray(self.state)
        self.state = cv2.resize(self.state, (0, 0), fx = 0.5, fy = 0.5)
        self.state = np.expand_dims(self.state, axis=0)
        self.state = np.concatenate((self.state, self.state), axis = 0)
        plt.imshow(self.state[0,:,:], cmap='gray')
        plt.show()
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0
        timeStart = time.time()
        for frame_idx in range(1, num_frames + 1):
            if(frame_idx % 100000 == 0):
              torch.save(self.dqn.state_dict(), "categorical_dqn.pth")
              print(str(frame_idx) + ' saved!')
              print(f'Time: {time.time()-timeStart}')
            action = self.select_action(self.state)
            next_state, reward, done = self.step(action)


            state = next_state
            score += reward

            # if episode ends
            if done:
                self.state, _ = self.env.reset(seed=self.seed)
                self.state = self.env.render()
                self.state = self.rgb2gray(self.state)
                self.state = cv2.resize(self.state, (0, 0), fx = 0.5, fy = 0.5)
                self.state = np.expand_dims(self.state, axis=0)
                self.state = np.concatenate((self.state, self.state), axis = 0)
                scores.append(score)
                score = 0

            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1

                # linearly decrease epsilon
                self.epsilon = max(
                    self.min_epsilon, self.epsilon - (
                        self.max_epsilon - self.min_epsilon
                    ) * self.epsilon_decay
                )
                if(self.epsilon == self.min_epsilon):
                  self.reset_countdown -= 1
                  if(self.reset_countdown <=0):
                    self.reset_countdown = int(1/self.epsilon_decay)
                    #self.epsilon = self.max_epsilon
                epsilons.append(self.epsilon)

                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

            # plotting
            if frame_idx % plotting_interval == 0:
                self._plot(frame_idx, scores, losses, epsilons)

        self.env.close()

    def test(self, video_folder=None):
        """Test the agent."""
        self.is_test = True

        # for recording a video
        naive_env = self.env
        if(video_folder):
            self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        self.state, _ = self.env.reset(seed=self.seed)
        self.state = self.env.render()
        self.state = self.rgb2gray(self.state)
        self.state = cv2.resize(self.state, (0, 0), fx = 0.5, fy = 0.5)
        self.state = np.expand_dims(self.state, axis=0)
        self.state = np.concatenate((self.state, self.state), axis = 0)
        done = False
        score = 0

        while not done:
            action = self.select_action(self.state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        #print("score: ", score)
        self.env.close()

        # reset
        self.env = naive_env
        return score


    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            next_action = self.dqn_target(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]
            #print(f'Reward: {reward.shape}')
            #print(f'Support: {self.support.shape}')
            t_z = reward + (1 - done) * self.gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])

        loss = -(proj_dist * log_p).sum(1).mean()

        return loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _plot(
        self,
        frame_idx: int,
        scores: List[float],
        losses: List[float],
        epsilons: List[float],
    ):
        """Plot the training progresses."""
        #plt.figure(figsize=(20, 5))
        #plt.subplot(131)
        scoresFig = plt.figure()
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        scoresFig.savefig('scoresPlot.png')
        plt.close(scoresFig)
        lossesFig = plt.figure()
        plt.title('loss')
        plt.plot(losses)
        plt.xlabel('Step')
        lossesFig.savefig('LossPlot.png')
        plt.close(lossesFig)
        #plt.subplot(133)
        #plt.title('epsilons')
        #plt.plot(epsilons)
        #plt.show()
    def rgb2gray(self,rgb):

      r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
      gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
      return gray
env = gym.make("ALE/Boxing-v5", render_mode="rgb_array")
#env = gym.make("CartPole-v1", render_mode="rgb_array")
seed = 777

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

np.random.seed(seed)
seed_torch(seed)

num_frames = 3000000
memory_size = 30000
batch_size = 32
target_update = 2500
epsilon_decay = 1 / 20000

# train
print(device)
agent = DQNAgent(env, memory_size, batch_size, target_update, epsilon_decay, seed)
agent.epsilon = 1
print('Training')
#agent.train(num_frames)
#torch.save(agent.dqn.state_dict(), "categorical_dqn_final.pth")
agent.dqn.load_state_dict(torch.load("categorical_dqn_final3_mill.pth", weights_only=True))
video_folder="categorical_dqn_video"
#agent.test()
testScores = [0 for i in range(500)]
for i in range(500):
    testScores[i] = agent.test()
print(f'Max: {max(testScores)}')
print(f'Min: {min(testScores)}')
print(f'Avg: {np.average(testScores)}')
print(f'std: {np.std(testScores)}')
