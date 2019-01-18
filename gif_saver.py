
import imageio
import os
from skimage.transform import resize
import numpy as np

'''
From:  
    https://medium.com/@fabiograetz/tutorial-double-deep-q-learning-with-dueling-network-architectures-4c1b3fb7f756
'''
class GifSaver:
    def __init__(self, memory, agent, save_every_x_episodes):
        self.episode_counter = 0
        self.save_every_x_episodes = save_every_x_episodes
        self.memory = memory
        self.agent = agent
        self.outputDir = "gifs"
        os.makedirs(self.outputDir, exist_ok=True)

    def OnEpisodeOver(self):
        self.episode_counter += 1
        if (self.episode_counter >= self.save_every_x_episodes):
            self._CreateGif()
            self.episode_counter = 0

    def _CreateGif(self):
        frames = self._CollectFramesForGif()
        self._GenerateGif(frames[0].id, frames)
        
    def _CollectFramesForGif(self):
        frameContents = self.memory.GetFramesForLatestEpisode()
        return frameContents
            
    def _GenerateGif(self, gifName, frames_for_gif):
        for idx, frame_obj in enumerate(frames_for_gif): 
            frames_for_gif[idx] = resize(frame_obj.Contents, (420, 320),
                          mode='constant',
                          preserve_range=True, order=0).astype(np.uint8)
        
        imageio.mimsave(f'{self.outputDir}/{gifName}.gif', 
                        frames_for_gif, duration=1/15)
        
