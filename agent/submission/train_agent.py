from .agent import DQNAgent, RandomAgent
from grid2op.Agent import DoNothingAgent
from grid2op import make
from lightsim2grid import LightSimBackend
from .evaluate import evaluate
import os

def train_agent(actions_max, nb_episodes, curr_dir):

  env = make("l2rpn_neurips_2020_track2_small", backend=LightSimBackend())

  network_dir = os.path.join(curr_dir, "DQN_NN")

  dqn_agent = DQNAgent(env, actions_max, curr_dir)
  nothing_agent = DoNothingAgent(env.action_space)

  dqn_agent.learn(env, network_dir, nb_episodes=nb_episodes)
  dqn_agent.dqn.save_nn(network_dir)

  dqn_agent.dqn.load_nn(network_dir)
  evaluate(nothing_agent, "NOTHING")
  evaluate(dqn_agent, "DQN")
  