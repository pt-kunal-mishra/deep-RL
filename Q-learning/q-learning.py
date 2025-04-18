import random
import numpy as np
import gym
import warnings

# Suppress the deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

env= gym.make('Taxi-v3')

#learning rate
alpha=0.9
#discount factor
gamma=0.9
#randomness (1 refers to always random if 0 then never take random action)
epsilon=1.0
epsilon_delay=0.9995
min_epsilon=0.01
num_episodes=10000
max_steps=100

#initialize q-table 5 x5 grids positions->25 positions * 5 * 4
q_table=np.zeros((env.observation_space.n,env.action_space.n))

def choose_action(state):
    if random.uniform(0,1)< epsilon:
        return  env.action_space.sample()
    else:
        return np.argmax(q_table[state,:])

for episode in range(max_steps):
    state,_=env.reset()

    done=False

    for step in range(max_steps):
        action=choose_action(state)

        next_state,reward,done,truncated,info=env.step(action)


        old_value=q_table[state,action]
        next_max=np.max(q_table[next_state,:])

        q_table[state,action]=(1-alpha)*old_value+alpha*(reward+gamma * next_max)
        state=next_state

        if done or truncated:
            break
    
    epsilon=max(min_epsilon,epsilon*epsilon_delay)


env=gym.make('Taxi-v3',render_mode='human')


for episode in range(5):
    state,_ = env.reset()
    done=False

    print("episode",episode)

    for step in range(max_steps):
        env.render()
        action=np.argmax(q_table[state,:])
        next_state,reward,done,truncated,info=env.step(action)
        state=next_state

        if done or truncated:
            env.render()
            print("finished episode",episode,'with reward',reward)
            break



env.close()







