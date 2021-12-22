
from grid2op.Agent import BaseAgent
from grid2op.Agent import DoNothingAgent
from grid2op import make
from grid2op.PlotGrid import PlotMatplot
from DenseNN import DenseNN
from DQN import DQN
import numpy as np
from evaluate import evaluate
import os

env = make("l2rpn_neurips_2020_track2_small")
plot_helper = PlotMatplot(env.observation_space)

class DQNAgent(BaseAgent):
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
      self.all_actions = np.asarray(self.all_actions[:50])
      print(f"\nAction space: {self.all_actions.shape[0]} actions")

      self.dqn = DQN(self, (lambda : DenseNN(env.current_obs.to_vect().shape[0], self.all_actions.shape[0])))

    def act(self, observation, reward, done):
      """The action that your agent will choose depending on the observation, the reward, and whether the state is terminal"""
      # do nothing for example (with the empty dictionary) :
      
      return self.dqn.select_action(observation)

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


dqn_agent = DQNAgent(env, ".")
random_agent = RandomAgent(env, ".")
nothing_agent = DoNothingAgent(env.action_space)

dqn_agent.train(env, 1000)
dqn_agent.dqn.save_nn("DQN_NN")

dqn_agent.dqn.load_nn("DQN_NN")
evaluate(random_agent, "RANDOM")
evaluate(nothing_agent, "NOTHING")
evaluate(dqn_agent, "DQN")