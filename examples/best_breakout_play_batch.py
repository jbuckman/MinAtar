################################################################################################################
# Authors:                                                                                                     #
# Kenny Young (kjyoung@ualberta.ca)                                                                            #
# Tian Tian(ttian@ualberta.ca)                                                                                 #
#                                                                                                              #
# python3 random_play.py -g <game>                                                                             #                                                              #
################################################################################################################
import random, numpy, argparse, time
from minatar import Environment, BatchEnvironment,VectorizedEnvironment

NUM_EPISODES = 1000

parser = argparse.ArgumentParser()
parser.add_argument("--game", "-g", type=str)
args = parser.parse_args()

batch_env = BatchEnvironment(NUM_EPISODES, args.game)
vector_env = VectorizedEnvironment(NUM_EPISODES, args.game)

num_actions = batch_env.num_actions()
# num_actions = vector_env.num_actions()

# Initialize the return for every episode
G = 0.0
T = 0

start_time = time.time()

# Initialize the environment
vector_env.reset(seeds=list(range(NUM_EPISODES)))
batch_env.reset(seeds=list(range(NUM_EPISODES)))
vector_terminated = numpy.array(False)
batch_terminated = numpy.array(False)

# while(not vector_terminated.all()):
while(not batch_terminated.all()):
    batch_s = batch_env.state()
    vector_s = vector_env.state()
    assert numpy.allclose(vector_s, batch_s)

    actions = []
    for i in range(NUM_EPISODES):
        # paddle_pos = vector_s[i, 9, :, 0].argmax()
        # ball_pos_x = vector_s[i, :, :, 1].argmax() % 10
        # ball_pos_y = vector_s[i, :, :, 1].argmax() // 10
        # last_pos_x = vector_s[i, :, :, 2].argmax() % 10
        # last_pos_y = vector_s[i, :, :, 2].argmax() // 10
        paddle_pos = batch_s[i, 9, :, 0].argmax()
        ball_pos_x = batch_s[i, :, :, 1].argmax() % 10
        ball_pos_y = batch_s[i, :, :, 1].argmax() // 10
        last_pos_x = batch_s[i, :, :, 2].argmax() % 10
        last_pos_y = batch_s[i, :, :, 2].argmax() // 10

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
        actions.append(action)
        a = numpy.array(actions)

    batch_reward, batch_terminated = batch_env.act(a)
    vector_reward, vector_terminated = vector_env.act(a)
    assert numpy.allclose(batch_reward, vector_reward)
    assert numpy.allclose(batch_terminated, vector_terminated)

    G += batch_reward
    T += numpy.cast[numpy.int](~batch_terminated)
    # G += vector_reward
    # T += numpy.cast[numpy.int](~vector_terminated)

print("Avg Return: " + str(numpy.mean(G))+"+/-"+str(numpy.std(G)/numpy.sqrt(NUM_EPISODES)))
print("Avg Length: " + str(numpy.mean(T))+"+/-"+str(numpy.std(T)/numpy.sqrt(NUM_EPISODES)))
print(f"Time: {time.time() - start_time}")

