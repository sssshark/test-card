import rlcard
from rlcard.agents.mccfr_agent import MCCFRAgent
from rlcard import models
from rlcard.utils.utils import set_global_seed
from rlcard.utils.logger import Logger


env = rlcard.make('mahjong', {'allow_step_back':True})
# env = rlcard.make('mahjong')
eval_env = rlcard.make('mahjong')


# Set the iterations numbers and how frequently we evaluate/save plot
evaluate_every = 100
save_plot_every = 1000
evaluate_num = 10000
episode_num = 100000


# The paths for saving the logs and learning curves
root_path = './experiments/mahjong_cfr_result/'
log_path = root_path + 'log.txt'
csv_path = root_path + 'performance.csv'
figure_path = root_path + 'figures/'



# Set a global seed
set_global_seed(0)

# Initilize CFR Agent
agent = MCCFRAgent(env)
# Init a Logger to plot the learning curve
logger = Logger(root_path)


for episode in range(episode_num+1):
    agent.train()
    print('\rIteration {}'.format(episode), end='')
    if episode% 1000 == 0:
        agent.save()
    # # Evaluate the performance. Play with NFSP agents.
    # if episode % evaluate_every == 0:
    #     reward = 0
    #     for eval_episode in range(evaluate_num):
    #         _, payoffs = eval_env.run(is_training=False)
    #
    #         reward += payoffs[0]
    #
    #     logger.log('\n########## Evaluation ##########')
    #     logger.log('Iteration: {} Average reward is {}'.format(episode, float(reward)/evaluate_num))
    #
    #     # Add point to logger
    #     logger.add_point(x=env.timestep, y=float(reward)/evaluate_num)
    #
    # # Make plot
    # if episode % save_plot_every == 0 and episode > 0:
    #     logger.make_plot(save_path=figure_path+str(episode)+'.png')

# Make the final plot
logger.make_plot(save_path=figure_path+'final_'+str(episode)+'.png')







print('done')