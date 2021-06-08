# Copyright 2021 Christian Leininger
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dynamics Prediction and Training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as d
import gym
from sklearn.model_selection import train_test_split


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class SkillDynamic(nn.Module):
    """skilldynamic Model."""

    def __init__(self, state_size, obs_size, fix_var=False, fc1_units=256, fc2_units=256, seed=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(SkillDynamic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.mean = nn.Linear(fc2_units, obs_size)
        self.std = nn.Linear(fc2_units, obs_size)
        self.apply(weights_init_)
        self.training = True

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.std(x)
        mean = self.reparam(mean, log_std)
        return mean, log_std

    def reparam(self, mean, logvar):
        """   """
        if self.training:
            std = logvar.mul(0.5).exp_()  # numerisch stabiler
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mean)
        else:
            return mean

    def sample(self, state):
        mean, log_std = self.forward(state)
        mu = self.reparam(mean, log_std)
        return mu, log_std


def gaussian_nll(
    pred_mean: torch.Tensor,
    pred_logvar: torch.Tensor,
    target: torch.Tensor,
    reduce: bool = True,
    ) -> torch.Tensor:
    """Negative log-likelihood for Gaussian distribution

    Args:
        pred_mean (tensor): the predicted mean.
        pred_logvar (tensor): the predicted log variance.
        target (tensor): the target value.
        reduce (bool): if ``False`` the loss is returned w/o reducing.
            Defaults to ``True``.

    Returns:
        (tensor): the negative log-likelihood.
    """
    l2 = F.mse_loss(pred_mean, target, reduction="none")
    inv_var = (-pred_logvar).exp()
    losses = l2 * inv_var + pred_logvar
    if reduce:
        return losses.sum(dim=1).mean()
    return losses


class SkillDynamicModel(nn.Module):
    def __init__(self, state_size, config):
        super(SkillDynamicModel, self).__init__()
        self.model = SkillDynamic(state_size, state_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"])
        self.batch_size = config["batch_size"]

    def train(self, states, actions, next_states):
        """ update dynamic model   """
        predict, log = self.model(torch.Tensor(states))
        delta = next_states - states
        self.optimizer.zero_grad()
        loss = gaussian_nll(predict, log, torch.Tensor(delta))
        loss.backward()
        self.optimizer.step()

    def test(self, X_train, X_valid, y_train, y_vaild):
        """ For testing the model

            Args:
                param1(numpy array): train data
                param2(numpy array): vaild data
                param3(numpy array): train label
                param4(numpy array): valid label
        """
        epochs = 100000
        batch_size = 32
        loss_list = []
        print("train size state ", X_train.shape)
        print("train size next_state ", y_train.shape)
        print("valid size state ", X_valid.shape)
        print("valid size next_state ", y_vaild.shape)
        for i in range(epochs):
            idxs = np.random.randint(0, X_train.shape[0], size=batch_size)
            state_batch = X_train[idxs]
            next_state_batch = y_train[idxs]
            predict, log = self.model(torch.Tensor(state_batch))
            delta = next_state_batch - state_batch
            self.optimizer.zero_grad()
            loss = gaussian_nll(predict, log, torch.Tensor(delta))
            loss.backward()
            self.optimizer.step()
            loss_list.append(loss.item())
            diff = 0
            if i % 2500 == 0:
                model.save_model("epoch_{}".format(i))
                model.load_model("epoch_{}".format(i))

                print("Epoch {} Loss {}".format(i, np.mean(loss_list)))
                for i in range(20):
                    idxs = np.random.randint(0, X_valid.shape[0], size=batch_size)
                    state_batch_valid = X_valid[idxs]
                    next_state_batch_valid = y_vaild[idxs]
                    self.model.training = False
                    predict, log = self.model(torch.Tensor(state_batch_valid))
                    self.model.training = True
                    a = predict.detach().numpy() + state_batch_valid
                    diff += np.sum(next_state_batch_valid - a)
                diff /= 20
                print("difference ", diff)

    def save_model(self, name, base="test"):
        """ """
        # create directory if not exist
        path = os.path.join(base, name)
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.model.state_dict(), path + "_dynamic_model.pth")
        print("saved model at {}".format(path))

    def load_model(self, name, base="test"):
        """ """
        path = os.path.join(base, name)
        try:
            self.model.load_state_dict(torch.load(path + "_dynamic_model.pth"))
            print("load model at {}".format(path))
        except Exception:
            print("Exception ", sys.exc_info())


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    state_data = []
    next_state_data = []
    for i in range(1000):
        state = env.reset()
        while True:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            state_data.append(state)
            next_state_data.append(next_state)
            state = next_state
            if done:
                break
    state_data = np.array(state_data)
    next_state_data = np.array(next_state_data)

    X_train, X_valid, y_train, y_vaild = train_test_split(state_data, next_state_data, test_size=0.33, random_state=42)
    config = {}
    config["lr"] = 0.003
    config["batch_size"] = 32
    state_size = X_train.shape[1]
    model = SkillDynamicModel(state_size, config)
    model.test(X_train, X_valid, y_train, y_vaild)
