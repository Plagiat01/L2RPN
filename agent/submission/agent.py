from grid2op.Agent import BaseAgent
from .DenseNN import DenseNN
from .DQN import DQN
import numpy as np
import os


class DQNAgent(BaseAgent):
  """
  The template to be used to create an agent: any controller of the power grid is expected to be a subclass of this
  grid2op.Agent.BaseAgent.
  """
  def __init__(self, env, max_actions, curr_dir):
    """Initialize a new agent."""
    BaseAgent.__init__(self, action_space=env.action_space)

    # Load the 1000 best actions
    self.all_actions = [env.action_space({})]
    actions = np.load(os.path.join(curr_dir, "top1000_actions.npz"), allow_pickle=True)["actions"]
    for action in actions:
      self.all_actions.append(env.action_space.from_vect(action))
    # Keep only 'max_actions' actions
    self.all_actions = np.asarray(self.all_actions[:max_actions])

    create_network = lambda : DenseNN(self.convert_obs(env.current_obs).shape[0], max_actions)
    self.dqn = DQN(self, env.reward_range[0], create_network)

  def act(self, observation, reward, done):
    """The action that your agent will choose depending on the observation, the reward, and whether the state is terminal"""
    # do nothing for example (with the empty dictionary) :
    
    return self.dqn.select_action(observation)
  
  def learn(self, env, path, nb_episodes):
    self.dqn.replay_exp(env, path, nb_episodes=nb_episodes)

  @staticmethod
  def convert_obs(obs):
    return obs.to_vect()

def make_agent(env, this_directory_path):
  my_agent = DQNAgent(env, 50, this_directory_path)
  my_agent.dqn.load_nn(os.path.join(this_directory_path, "DQN_NN"))
  return my_agent