################################################################################################################
# Authors:                                                                                                     #
# Kenny Young (kjyoung@ualberta.ca)                                                                            #
# Tian Tian(ttian@ualberta.ca)                                                                                 #
#                                                                                                              #
# python3 random_play.py -g <game>                                                                             #                                                              #
################################################################################################################
import random, numpy, argparse
from minatar import BatchEnvironment

BATCH_SIZE = 1000

parser = argparse.ArgumentParser()
parser.add_argument("--game", "-g", type=str)
args = parser.parse_args()

env = BatchEnvironment(BATCH_SIZE, args.game)

num_actions = env.num_actions()

terminated = numpy.zeros(BATCH_SIZE)
G = numpy.zeros(BATCH_SIZE)

# Initialize the environment
env.reset()

#Obtain first state, unused by random agent, but inluded for illustration
s = env.state()
while(not terminated.all()):
    # Select an action uniformly at random
    action = numpy.random.randint(num_actions, size=BATCH_SIZE)

    # Act according to the action and observe the transition and reward
    reward, terminated = env.act(action)

    # Obtain s_prime, unused by random agent, but inluded for illustration
    s_prime = env.state()

    G += reward * (1. - terminated)

print("Avg Return: " + str(numpy.mean(G))+"+/-"+str(numpy.std(G)/numpy.sqrt(BATCH_SIZE)))


