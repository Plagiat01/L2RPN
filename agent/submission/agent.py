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


    self.dqn = DQN(self, (lambda : DenseNN(2238, max_actions)))

  def act(self, observation, reward, done):
    """The action that your agent will choose depending on the observation, the reward, and whether the state is terminal"""
    # do nothing for example (with the empty dictionary) :
    
    return self.dqn.select_action(observation)

  @staticmethod
  def convert_obs(observation):
    tmp_list_vect = ['prod_p','load_p','p_or','a_or','p_ex','a_ex','rho','topo_vect','line_status',
                      'timestep_overflow','time_before_cooldown_line','time_before_cooldown_sub']

    li_vect=  []
    for el in tmp_list_vect:
        if el in observation.attr_list_vect:
            v = observation._get_array_from_attr_name(el).astype(np.float32)
            v_fix = np.nan_to_num(v)
            li_vect.append(v_fix)
    return np.concatenate(li_vect)[:2238]

  def train(self, env, nb_episode):
    self.dqn.replay_exp(env, nb_episode=nb_episode)
    

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
  my_agent = DQNAgent(env, 100, this_directory_path)
  my_agent.dqn.load_nn(os.path.join(this_directory_path, "DQN_NN"))
  return my_agent