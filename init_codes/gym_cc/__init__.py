from gym.envs.registration import register

register(
    id='cc-v0',
    entry_point='gym_cc.cc:CC',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 20000},
)

register(
    id='cc-quickrun-v0',
    entry_point='gym_cc.cc:CC',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 200},
)
