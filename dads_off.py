# Copyright 2019 Google LLC
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import pickle as pkl
import os
import io
from absl import flags, logging, app
import functools
import json
import sys
sys.path.append(os.path.abspath('./'))

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import dads_agent

from envs import skill_wrapper
from envs import video_wrapper
from envs.gym_mujoco import humanoid


#from lib import py_tf_policy
from sac import SAC

from replay_buffer import ReplayBuffer


from utils import get_environment, collect_experience

FLAGS = flags.FLAGS
# general hyperparameters
flags.DEFINE_string('logdir', '~/tmp/dads', 'Directory for saving experiment data')

# environment hyperparameters
flags.DEFINE_string('environment', 'point_mass', 'Name of the environment')
flags.DEFINE_integer('max_env_steps', 200,
                     'Maximum number of steps in one episode')
flags.DEFINE_integer('reduced_observation', 0,
                     'Predict dynamics in a reduced observation space')
flags.DEFINE_integer(
    'min_steps_before_resample', 50,
    'Minimum number of steps to execute before resampling skill')
flags.DEFINE_float('resample_prob', 0.,
                   'Creates stochasticity timesteps before resampling skill')

# need to set save_model and save_freq
flags.DEFINE_string(
    'save_model', None,
    'Name to save the model with, None implies the models are not saved.')
flags.DEFINE_integer('save_freq', 100, 'Saving frequency for checkpoints')
flags.DEFINE_string(
    'vid_name', None,
    'Base name for videos being saved, None implies videos are not recorded')
flags.DEFINE_integer('record_freq', 100,
                     'Video recording frequency within the training loop')

# final evaluation after training is done
flags.DEFINE_integer('run_eval', 0, 'Evaluate learnt skills')

# evaluation type
flags.DEFINE_integer('num_evals', 0, 'Number of skills to evaluate')
flags.DEFINE_integer('deterministic_eval', 0,
                  'Evaluate all skills, only works for discrete skills')

# training
flags.DEFINE_integer('run_train', 0, 'Train the agent')
flags.DEFINE_integer('num_epochs', 500, 'Number of training epochs')

# skill latent space
flags.DEFINE_integer('num_skills', 2, 'Number of skills to learn')
flags.DEFINE_string('skill_type', 'cont_uniform',
                    'Type of skill and the prior over it')
# network size hyperparameter
flags.DEFINE_integer(
    'hidden_layer_size', 512,
    'Hidden layer size, shared by actors, critics and dynamics')

# reward structure
flags.DEFINE_integer(
    'random_skills', 0,
    'Number of skills to sample randomly for approximating mutual information')

# optimization hyperparameters
flags.DEFINE_integer('replay_buffer_capacity', int(1e6),
                     'Capacity of the replay buffer')
flags.DEFINE_integer(
    'clear_buffer_every_iter', 0,
    'Clear replay buffer every iteration to simulate on-policy training, use larger collect steps and train-steps'
)
flags.DEFINE_integer(
    'initial_collect_steps', 2000,
    'Steps collected initially before training to populate the buffer')
flags.DEFINE_integer('collect_steps', 200, 'Steps collected per agent update')

# relabelling
flags.DEFINE_string('agent_relabel_type', None,
                    'Type of skill relabelling used for agent')
flags.DEFINE_integer(
    'train_skill_dynamics_on_policy', 0,
    'Train skill-dynamics on policy data, while agent train off-policy')
flags.DEFINE_string('skill_dynamics_relabel_type', None,
                    'Type of skill relabelling used for skill-dynamics')
flags.DEFINE_integer(
    'num_samples_for_relabelling', 100,
    'Number of samples from prior for relabelling the current skill when using policy relabelling'
)
flags.DEFINE_float(
    'is_clip_eps', 0.,
    'PPO style clipping epsilon to constrain importance sampling weights to (1-eps, 1+eps)'
)
flags.DEFINE_float(
    'action_clipping', 1.,
    'Clip actions to (-eps, eps) per dimension to avoid difficulties with tanh')
flags.DEFINE_integer('debug_skill_relabelling', 0,
                     'analysis of skill relabelling')

# skill dynamics optimization hyperparamaters
flags.DEFINE_integer('skill_dyn_train_steps', 8,
                     'Number of discriminator train steps on a batch of data')
flags.DEFINE_float('skill_dynamics_lr', 3e-4,
                   'Learning rate for increasing the log-likelihood')
flags.DEFINE_integer('skill_dyn_batch_size', 256,
                     'Batch size for discriminator updates')
# agent optimization hyperparameters
flags.DEFINE_integer('agent_batch_size', 256, 'Batch size for agent updates')
flags.DEFINE_integer('agent_train_steps', 128,
                     'Number of update steps per iteration')
flags.DEFINE_float('agent_lr', 3e-4, 'Learning rate for the agent')

# SAC hyperparameters
flags.DEFINE_float('agent_entropy', 0.1, 'Entropy regularization coefficient')
flags.DEFINE_float('agent_gamma', 0.99, 'Reward discount factor')
flags.DEFINE_string(
    'collect_policy', 'default',
    'Can use the OUNoisePolicy to collect experience for better exploration')

# skill-dynamics hyperparameters
flags.DEFINE_string(
    'graph_type', 'default',
    'process skill input separately for more representational power')
flags.DEFINE_integer('num_components', 4,
                     'Number of components for Mixture of Gaussians')
flags.DEFINE_integer('fix_variance', 1,
                     'Fix the variance of output distribution')
flags.DEFINE_integer('normalize_data', 1, 'Maintain running averages')

# debug
flags.DEFINE_integer('debug', 0, 'Creates extra summaries')

# DKitty
flags.DEFINE_integer('expose_last_action', 1, 'Add the last action to the observation')
flags.DEFINE_integer('expose_upright', 1, 'Add the upright angle to the observation')
flags.DEFINE_float('upright_threshold', 0.9, 'Threshold before which the DKitty episode is terminated')
flags.DEFINE_float('robot_noise_ratio', 0.05, 'Noise ratio for robot joints')
flags.DEFINE_float('root_noise_ratio', 0.002, 'Noise ratio for root position')
flags.DEFINE_float('scale_root_position', 1, 'Multiply the root coordinates the magnify the change')
flags.DEFINE_integer('run_on_hardware', 0, 'Flag for hardware runs')
flags.DEFINE_float('randomize_hfield', 0.0, 'Randomize terrain for better DKitty transfer')
flags.DEFINE_integer('observation_omission_size', 2, 'Dimensions to be omitted from policy input')

# Manipulation Environments
flags.DEFINE_integer('randomized_initial_distribution', 1, 'Fix the initial distribution or not')
flags.DEFINE_float('horizontal_wrist_constraint', 1.0, 'Action space constraint to restrict horizontal motion of the wrist')
flags.DEFINE_float('vertical_wrist_constraint', 1.0, 'Action space constraint to restrict vertical motion of the wrist')

# MPC hyperparameters
flags.DEFINE_integer('planning_horizon', 1, 'Number of primitives to plan in the future')
flags.DEFINE_integer('primitive_horizon', 1, 'Horizon for every primitive')
flags.DEFINE_integer('num_candidate_sequences', 50, 'Number of candidates sequence sampled from the proposal distribution')
flags.DEFINE_integer('refine_steps', 10, 'Number of optimization steps')
flags.DEFINE_float('mppi_gamma', 10.0, 'MPPI weighting hyperparameter')
flags.DEFINE_string('prior_type', 'normal', 'Uniform or Gaussian prior for candidate skill(s)')
flags.DEFINE_float('smoothing_beta', 0.9, 'Smooth candidate skill sequences used')
flags.DEFINE_integer('top_primitives', 5, 'Optimization parameter when using uniform prior (CEM style)')




def main(argv):
    # setting up
    print("start main")
    with open ("param.json", "r") as f:
        param = json.load(f)

    start_time = time.time()
    logging.set_verbosity(logging.INFO)
    global observation_omit_size, goal_coord, sample_count, iter_count, episode_size_buffer, episode_return_buffer
    root_dir = os.path.abspath(os.path.expanduser(FLAGS.logdir))
    log_dir = os.path.join(root_dir, FLAGS.environment)
    save_dir = os.path.join(log_dir, 'models')
    print('directory for recording experiment data:', log_dir)
    # in case training is paused and resumed, so can be restored
    try:
        sample_count = np.load(os.path.join(log_dir, 'sample_count.npy')).tolist()
        iter_count = np.load(os.path.join(log_dir, 'iter_count.npy')).tolist()
        episode_size_buffer = np.load(os.path.join(log_dir, 'episode_size_buffer.npy')).tolist()
        episode_return_buffer = np.load(os.path.join(log_dir, 'episode_return_buffer.npy')).tolist()
    except:
        sample_count = 0
        iter_count = 0
        episode_size_buffer = []
        episode_return_buffer = []
    print("off 1113 sample count ", sample_count)
    print("off 1114 log dir ", log_dir)
    env = get_environment(env_name=FLAGS.environment)
    print("obs space ", env.observation_space)
    obs_space = env.observation_space.shape[0] + 5
    agent = SAC(obs_space, env.action_space, param)
    print(agent.policy)
    env = skill_wrapper.SkillWrapper(
            env,
            num_latent_skills=FLAGS.num_skills,
            skill_type=FLAGS.skill_type,
            preset_skill=None,
            min_steps_before_resample=FLAGS.min_steps_before_resample,
            resample_prob=FLAGS.resample_prob)
    state = env.reset()
    print(state)
    print(state.shape)
    save_dir = ""
    skill_dynamics_observation_size = 0
    total_steps = 0
    d_agent = dads_agent.DADSAgent(
        # DADS parameters
        save_dir,
        skill_dynamics_observation_size,
        agent,
        latent_size=FLAGS.num_skills,
        latent_prior=FLAGS.skill_type,
        prior_samples=FLAGS.random_skills,
        fc_layer_params=(FLAGS.hidden_layer_size,) * 2,
        num_mixture_components=FLAGS.num_components,
        skill_dynamics_learning_rate=FLAGS.skill_dynamics_lr)
    start_time = time.time()
    obs_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    print(action_space)
    device = torch.device("cuda" if int(param["cuda"]) else "cpu")
    capacity = 10000
    replay_buffer = ReplayBuffer((obs_space, ), (action_space,), capacity, device)
    time_step, collect_info = collect_experience(
            env,
            state,
            agent,
            replay_buffer,
            num_steps=FLAGS.initial_collect_steps)
    print("replay buffer size ", replay_buffer.idx)





if __name__ == '__main__':
    app.run(main)
