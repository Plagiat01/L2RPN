import numpy as np
import random
from collections import deque
import os

class DQN:
  def __init__(self, agent, create_model):
    self.agent = agent
    self.main_model = create_model()
    self.target_model = create_model()
    self.main_model.copy_weights(self.target_model)


  def train(self, replay_memory):
    batch_size = 128
    if len(replay_memory) < 1000: return
    mini_batch = random.sample(replay_memory, batch_size)

    lr = 0.7
    discount = 0.618
    
    current_states     = np.asarray([step[0] for step in mini_batch])
    current_qs_list    = self.main_model.predict(current_states)
    next_states        = np.asarray([step[2] for step in mini_batch])
    next_qs_list       = self.target_model.predict(next_states)

    X = np.zeros((batch_size, current_states[0].shape[0]))
    y = np.zeros((batch_size, current_qs_list[0].shape[0]))

    for i, (obs, action, _new_obs, reward, done) in enumerate(mini_batch):
      if not done:
        future_q = reward + discount * np.max(next_qs_list[i])
      if done:
        future_q = reward
      
      # Classic Q-Learning Bellman formula
      current_qs = current_qs_list[i]
      current_qs[action] = (1 - lr) * current_qs[action] + lr * future_q

      X[i] = obs
      y[i] = current_qs    
    
    self.main_model.fit(X, y, batch_size, shuffle=True)
  

  def replay_exp(self, env, nb_episode=150, max_replay_memory=50_000, main_update_step=4, target_update_step=100):
    epsilon = 1
    max_epsilon = 1
    min_epsilon = 0.01
    decay = 0.01

    replay_memory = deque(maxlen=max_replay_memory)
    steps_update = 0

    for episode in range(nb_episode):
      obs = env.reset()
      done = False

      sum_reward = 0
      total_steps = 0
      while not done:
        total_steps += 1
        steps_update += 1
        if np.random.rand() <= epsilon:
          action = np.random.randint(self.agent.all_actions.shape[0])
        else:
          action_list = self.main_model.predict(np.expand_dims(obs.to_vect(), axis=0))
          action = np.argmax(action_list)
        
        new_obs, reward, done, _ = env.step(self.agent.all_actions[action])
        replay_memory.append((obs.to_vect(), action, new_obs.to_vect(), reward, done))

        if steps_update % main_update_step == 0 or done:
          self.train(replay_memory)

        obs = new_obs
        sum_reward += reward
      
      if steps_update >= target_update_step:
        print("\033[92m"+"Copying main model weights to the target model weights" + "\033[0m")
        self.main_model.copy_weights(self.target_model)
        steps_update = 0

      epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

      print(f"Episode {episode} -> survived steps: {total_steps} total reward: {sum_reward:.2f}")

  
  def save_models(self, path):
    self.main_model.save(os.path.join(path, "main_model.h5"))
    self.target_model.save(os.path.join(path, "target_model.h5"))
  
  def load_models(self, path):
    self.main_model.load(os.path.join(path, "main_model.h5"))
    self.target_model.load(os.path.join(path, "target_model.h5"))


  def select_action(self, obs):
    action_list = self.main_model.predict(np.expand_dims(obs.to_vect(), axis=0))
    action = np.argmax(action_list)
    return self.agent.all_actions[action]