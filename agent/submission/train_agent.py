from .agent import DQNAgent
from grid2op.Agent import DoNothingAgent
from grid2op import make
from lightsim2grid import LightSimBackend
from .evaluate import evaluate
import os

env = make("l2rpn_neurips_2020_track2_small", backend=LightSimBackend())

def create_agent(actions_max, curr_dir):
  return DQNAgent(env, actions_max, curr_dir)


def train_agent(dqn_agent, nb_episodes, curr_dir):
  network_dir = os.path.join(curr_dir, "DQN_NN")

  # Train the agent
  dqn_agent.learn(env, network_dir, nb_episodes=nb_episodes)
  dqn_agent.dqn.save_nn(network_dir)
  

def load_agent_model(dqn_agent, curr_dir):
  network_dir = os.path.join(curr_dir, "DQN_NN")
  dqn_agent.dqn.load_nn(network_dir)
  dqn_agent.dqn.load_parameters(network_dir)


def evaluate_agent(dqn_agent, name="DQN"):
  nothing_agent = DoNothingAgent(env.action_space)
  evaluate(nothing_agent, "NOTHING")

  evaluate(dqn_agent, name)