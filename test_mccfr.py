import rlcard
from rlcard.agents.mccfr_agent import MCCFRAgent
from rlcard import models
from rlcard.utils.utils import set_global_seed
from rlcard.utils.logger import Logger

def train():
    env = rlcard.make('mahjong', {'allow_step_back':True})
    # env = rlcard.make('mahjong')

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
        if episode% 5000 == 0:
            agent.save(episode)
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



def eval():
    eval_env = rlcard.make('mahjong')
    agent = MCCFRAgent(eval_env)
    agent.load(50000)

    agent_player = 0


    while True:
        agent.env.init_game()
        agent.env.game.players[1].print_hand()
        while not agent.env.is_over():
            current_player = agent.env.get_player_id()
            if current_player == agent_player:
                print('對手牌')
                agent.env.game.players[0].print_hand()
                agent.eval_step()
            else:
                agent.show_actions()
                print('打一張牌： ')
                action = input()
                # print('user do: ', action)
                agent.do_actions(int(action))

        print(agent.env.get_payoffs())
    # _, payoffs = eval_env.run(is_training=False)

    print('done')

if __name__ == '__main__':
    eval()