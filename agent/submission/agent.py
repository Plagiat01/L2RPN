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


    self.all_actions = [env.action_space({})]
    actions = np.load(os.path.join(curr_dir, "top1000_actions.npz"), allow_pickle=True)["actions"]
    for action in actions:
      self.all_actions.append(env.action_space.from_vect(action))
    # Keep only 'max_actions' actions
    self.all_actions = np.asarray(self.all_actions[:max_actions])


    self.dqn = DQN(self, (lambda : DenseNN(4116, max_actions)))

  def act(self, observation, reward, done):
    """The action that your agent will choose depending on the observation, the reward, and whether the state is terminal"""
    # do nothing for example (with the empty dictionary) :
    
    return self.dqn.select_action(observation)

  @staticmethod
  def convert_obs(obs):
    obs_vec = obs.to_vect()
    res = np.zeros(4116)
    res[:obs_vec.shape[0]] = obs_vec
    return res
    

class RandomAgent(BaseAgent):
  """
  The template to be used to create an agent: any controller of the power grid is expected to be a subclass of this
  grid2op.Agent.BaseAgent.
  """
  def __init__(self, env, curr_dir):
    """Initialize a new agent."""
    BaseAgent.__init__(self, action_space=env.action_space)

    self.all_actions = [env.action_space({})]
    actions = np.load(os.path.join(curr_dir, "top1000_actions.npz"), allow_pickle=True)["actions"]
    for action in actions:
      self.all_actions.append(env.action_space.from_vect(action))
    self.all_actions = np.asarray(self.all_actions)

  def act(self, observation, reward, done):
    """The action that your agent will choose depending on the observation, the reward, and whether the state is terminal"""
    # do nothing for example (with the empty dictionary) :
    
    return np.random.choice(self.all_actions)

def make_agent(env, this_directory_path):
  my_agent = DQNAgent(env, 50, this_directory_path)
  my_agent.dqn.load_nn(os.path.join(this_directory_path, "DQN_NN"))
  return my_agent