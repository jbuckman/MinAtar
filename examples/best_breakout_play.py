################################################################################################################
# Authors:                                                                                                     #
# Kenny Young (kjyoung@ualberta.ca)                                                                            #
# Tian Tian(ttian@ualberta.ca)                                                                                 #
#                                                                                                              #
# python3 random_play.py -g <game>                                                                             #                                                              #
################################################################################################################
import random, numpy, argparse, time
from minatar import Environment

NUM_EPISODES = 32 # 1000

parser = argparse.ArgumentParser()
parser.add_argument("--game", "-g", type=str)
args = parser.parse_args()

env = Environment(args.game)

e = 0
returns = []
transitions = []
num_actions = env.num_actions()

# Run NUM_EPISODES episodes and log all returns
while e < NUM_EPISODES:
    # Initialize the return for every episode
    G = 0.0
    T = 0

    # Initialize the environment
    env.reset(seed=e)
    terminated = False

    while(not terminated):
        # env.display_state()
        s = env.state()
        paddle_pos = s[9, :, 0].argmax()
        ball_pos_x = s[:,:,1].argmax() % 10
        ball_pos_y = s[:,:,1].argmax() // 10
        last_pos_x = s[:,:,2].argmax() % 10
        last_pos_y = s[:,:,2].argmax() // 10

        if paddle_pos == ball_pos_x:
            if last_pos_x < ball_pos_x: # moving right
                if ball_pos_x == 9: action = 0
                else: action = 3 # right
            elif last_pos_x > ball_pos_x: # moving left
                if ball_pos_x == 0: action = 0
                else: action = 1 # left
            else: # very start of game
                if ball_pos_x == 0: action = 3
                else:               action = 1
        elif paddle_pos < ball_pos_x:
            action = 3
        else:
            action = 1

        # Act according to the action and observe the transition and reward
        reward, terminated = env.act(action)

        G += reward
        T += 1

    # Increment the episodes
    e += 1

    # Store the return for each episode
    returns.append(G)
    transitions.append(T)
    # print(G,T)
    # import code; code.interact(local=locals())

print("Avg Return: " + str(numpy.mean(returns))+"+/-"+str(numpy.std(returns)/numpy.sqrt(NUM_EPISODES)))
print("Avg Length: " + str(numpy.mean(transitions))+"+/-"+str(numpy.std(transitions)/numpy.sqrt(NUM_EPISODES)))


