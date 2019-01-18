
import os.path
import numpy as np
import gym

from active_memory import ActiveMemory
from prioritized_memory import PrioritizedMemory
from batch_helper import BatchHelper
from ddqn_agent import DdqnAgent
from gif_saver import GifSaver
from DataModel.ddqn_globals import DdqnGlobals
from DataModel.progress_tracker import ProgressTracker, ProgressTrackerParms
from DataModel.hyper_parameters import StandardAgentParameters
from DataModel.hyper_parameters import ShortEpisodeParameters, LongEpisodeParameters
from DataModel.hyper_parameters import ShortPrioritizedParameters, LongPrioritizedParameters


'''
 Based on agents from rlcode, keon, A.L.Ecoffet, and probably several others
'''

class ImagePreProcessor:
    def to_grayscale(img):
        return np.mean(img, axis=2).astype(np.uint8)
    
    def downsample(img):
        return img[::2, ::2]
    
    def Preprocess(img):
        shrunk = ImagePreProcessor.downsample(img)
        return ImagePreProcessor.to_grayscale(shrunk)

class EpisodeManager:
    def __init__(self, environment, memory, action_size, miscParameters):
        self.environment = environment
        self.memory = memory
        batchHelper = BatchHelper(memory, DdqnGlobals.BATCH_SIZE, action_size)
        self.progressTracker = ProgressTracker(
                ProgressTrackerParms(avgPerXEpisodes=10, longAvgPerXEpisodes=100))

        self.agent = DdqnAgent(StandardAgentParameters, 
                               action_size, 
                               batchHelper, 
                               self.progressTracker)
        self.gifSaver = GifSaver(memory, 
                                  self.agent, 
                                  save_every_x_episodes=miscParameters.createGifEveryXEpisodes)
        
    def ShouldStop(self):
        return os.path.isfile("StopTraining.txt")
        
    def Run(self):
        while(self.ShouldStop() == False):
            self.OnEpisodeStart()
            score, steps = self.RunOneEpisode()
            self.OnEpisodeOver(score, steps)
        self.agent.OnExit()

    def OnEpisodeStart(self):
        self.progressTracker.OnEpisodeStart()
        self.memory.OnEpisodeStart()

    def OnEpisodeOver(self, score, steps):
        self.memory.OnEpisodeOver(score)
        self.agent.OnGameOver(steps)
        self.gifSaver.OnEpisodeOver()
        self.progressTracker.OnEpisodeOver(score, steps)

    def PrepEnvironmentForNextEpisode(self):
        self.environment.reset()
        info = None
        for _ in range(np.random.randint(DdqnGlobals.FRAMES_PER_STATE, DdqnGlobals.MAX_NOOP)):
            frame, _, done, info = self.NextStep(self.agent.GetNoOpAction())
            self.memory.AddFrame(frame)
        return info
            
    def NextStep(self, action):
        rawFrame, reward, done, info = self.environment.step(action)
        processedFrame = ImagePreProcessor.Preprocess(rawFrame)
        return processedFrame, reward, done, info
            
    def RunOneEpisode(self):
        info = self.PrepEnvironmentForNextEpisode()
        done = False
        stepCount = 0
        score = 0
        livesLeft = info['ale.lives']
        while not done:            
            action = self.agent.GetAction()
            frame, reward, done, info = self.NextStep(action)
            score += reward
            if(info['ale.lives'] < livesLeft):
                reward = -1
                livesLeft = info['ale.lives']
            self.memory.AddMemory(frame, action, reward, done)
            self.agent.Replay()
            stepCount += 1
        return score, stepCount
             
class Trainer:
    def Run(self, whichGame, miscParams):
        env = gym.make(whichGame)
        print(env.unwrapped.get_action_meanings())
        if(miscParams.usePriorizedReplay == True):
            memory = PrioritizedMemory()
        else:
            memory = ActiveMemory()
        num_actions = env.action_space.n
        if('Pong' in whichGame):
            num_actions = 4  # Don't need RIGHTFIRE or LEFTFIRE (do we?)
        mgr = EpisodeManager(env, memory, action_size = num_actions, miscParameters=miscParams)
        mgr.Run()

def Main(whichGame, miscParams):
    trainer = Trainer()
    trainer.Run(whichGame, miscParams)

#Main('PongDeterministic-v4', LongEpisodeParameters)
Main('BreakoutDeterministic-v4', ShortPrioritizedParameters)

