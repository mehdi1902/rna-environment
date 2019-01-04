from gym.envs.registration import register

register(
    id='rna-v0',
    entry_point='gym_rna.envs:RNAEnv',
)