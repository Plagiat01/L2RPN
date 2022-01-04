from submission.train_agent import *

dqn_agent = create_agent(actions_max=50, curr_dir="submission")
train_agent(dqn_agent, nb_episodes=2000, curr_dir="submission")
load_agent_model(dqn_agent, curr_dir="submission")
evaluate_agent(dqn_agent)