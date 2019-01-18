


from DataModel.ddqn_globals import DdqnGlobals 


class AgentParameters:
    def __init__(self,
                 state_size,
                 epsilon_start, 
                 epsilon_min, 
                 epsilon_decay_step,
                 delayTraining,
                 update_target_rate,
                 gamma,
                 learning_rate):
        self.state_size = state_size
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_step = epsilon_decay_step
        self.delayTraining = delayTraining
        self.update_target_rate = update_target_rate
        self.gamma = gamma
        self.learning_rate = learning_rate
        

StandardAgentParameters = AgentParameters(
        state_size = DdqnGlobals.STATE_DIMENSIONS,
        epsilon_start = 1.0,
        epsilon_min = 0.05,
        epsilon_decay_step = 2e-6,
        delayTraining = 20000,
        update_target_rate = 10000,
        gamma = 0.99,
        learning_rate = 3e-5
        )

class MiscParameters:
    def __init__(self, createGifEveryXEpisodes, usePrioritizedReplay):
        self.createGifEveryXEpisodes = createGifEveryXEpisodes
        self.usePriorizedReplay = usePrioritizedReplay


ShortEpisodeParameters = MiscParameters(createGifEveryXEpisodes=500, usePrioritizedReplay=False)
LongEpisodeParameters = MiscParameters(createGifEveryXEpisodes=100, usePrioritizedReplay=False)

ShortPrioritizedParameters = MiscParameters(createGifEveryXEpisodes=500, usePrioritizedReplay=True)
LongPrioritizedParameters = MiscParameters(createGifEveryXEpisodes=100, usePrioritizedReplay=True)
