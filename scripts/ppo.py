from tensorforce.agents import PPOAgent

import os
import itertools

import config
from env import GazeboEnv
from worldModels import VAE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

maze_id = config.maze_id
restore = False

record_dir = 'record'
if not os.path.exists(record_dir):
    os.makedirs(record_dir)

saver_dir = './models/nav{}'.format(maze_id)
if not os.path.exists(saver_dir):
    os.makedirs(saver_dir)

summarizer_dir = './record/PPO/nav{}'.format(maze_id)
if not os.path.exists(summarizer_dir):
    os.makedirs(summarizer_dir)

vae = VAE.VAE()
vae.set_weights(config.vae_weight)

network_spec = [
    dict(type='dense', size=512, activation='relu'),  # 'none'
    # dict(type='tf_layer', layer='batch_normalization', center=True, scale=True),
    # dict(type="nonlinearity", name='relu'),
    dict(type='dense', size=512, activation='relu'),
    # dict(type='tf_layer', layer='batch_normalization', center=True, scale=True),
    # dict(type="nonlinearity", name='relu'),
    dict(type='dense', size=512, activation='relu'),
    # dict(type='tf_layer', layer='batch_normalization', center=True, scale=True),
    # dict(type="nonlinearity", name='relu')
]

memory = dict(
    type='replay',
    include_next_states=False,
    capacity=10000
)

exploration = dict(
    type='epsilon_decay',
    initial_epsilon=0.1,
    final_epsilon=0.01,
    timesteps=100000,
    start_timestep=0
)


optimizer = dict(
    type='adam',
    learning_rate=0.0001
)
env = GazeboEnv(maze_id=maze_id, continuous=True)
agent = PPOAgent(
    states=dict(shape=(69,), type='float'),  # GazeboMaze.states,
    actions=env.actions(),
    network=network_spec,
    memory=memory,
    actions_exploration=exploration,
    saver=dict(directory=saver_dir, basename='PPO_model.ckpt', load=restore, seconds=10800),
    summarizer=dict(directory=summarizer_dir, labels=["graph", "total-loss", "reward", "entropy"], seconds=10800),
    step_optimizer=optimizer
)

#episode means the number of games
episode = 0
total_timestep = 0
max_timesteps = 100
max_episodes = 10000
episode_rewards = []
successes = []

while True:
    agent.reset()
    state = env.reset()
    state['img'] = state['img'] / 255.0
    done = False
    score = 0
    # timestep means actions in one game
    timestep = 0
    episode_reward = 0
    success = False

    while True:
        latent_vector = vae.get_vector(state['img'].reshape(1, 48, 64, 3))
        latent_vector = list(itertools.chain(*latent_vector))  # [[ ]]  ->  [ ]
        relative_pos = env.p
        previous_act = env.vel_cmd
        # print(previous_act)
        state = latent_vector + state['scan'] + relative_pos*2 + previous_act*2
        action = agent.act(state)
        state, reward, done = env.execute(action)
        state['img'] = state['img'] / 255.0  # normalize
        # print(reward)
        # Pass feedback about performance (and termination) to the agent
        agent.observe(terminal=done, reward=reward)
        timestep += 1
        print("timestep: ",timestep)
        episode_reward += reward
        if done or timestep == max_timesteps:
            success = env.success
            break

    episode += 1
    print("episode:", episode,"\tepisode_reward: ",episode_reward)
    total_timestep += timestep
    # avg_reward = float(episode_reward)/timestep
    successes.append(success)
    episode_rewards.append([episode_reward, timestep, success])

    if total_timestep > 100000:
        print('{}th episode reward: {}'.format(episode, episode_reward))

    if episode % 100 == 0:
        f = open(record_dir + '/PPO_nav' + str(maze_id) + '.txt', 'a+')
        for i in episode_rewards:
            f.write(str(i))
            f.write('\n')
        f.close()
        episode_rewards = []
        agent.save_model('./models/')

    if len(successes) > 100:
        if sum(successes[-100:]) > 80:
            env.close()
            agent.save_model('./models/')
            f = open(record_dir + '/PPO_nav' + str(maze_id) + '.txt', 'a+')
            for i in episode_rewards:
                f.write(str(i))
                f.write('\n')
            f.close()
            print("Training End!")
            break