
import numpy as np
from DataModel.ddqn_globals import DdqnGlobals
from DataModel.frame_helper import Normalize


'''
  Memories for a complete episode 
'''
class SingleEpisode():
    def __init__(self):
        self.Frames = dict()
        self.Memories = []
        self.EpisodeReward = 0
        self.MinFrameId = 0
        self.MaxFrameId = 0

    def AddMemory(self, theMemory):
        self.Memories.append(theMemory)
        
    def AddFrame(self, iFrame):
        self.Frames[iFrame.id] = iFrame
        self.MaxFrameId = iFrame.id
        if(self.MinFrameId == 0):
            self.MinFrameId = iFrame.id

    def ContainsFrame(self, frameId):
        return (frameId >= self.MinFrameId) and (frameId <= self.MaxFrameId)

    def GetFramesForState(self, stateId):
        tempTuple = ()
        for i in range(DdqnGlobals.FRAMES_PER_STATE):
            frameData = self.Frames[stateId+i].Contents
            tempTuple = tempTuple + (frameData.reshape(
                    DdqnGlobals.FRAME_DIMENSIONS[0], 
                    DdqnGlobals.FRAME_DIMENSIONS[1], 
                    1),)
        
        frameStack = np.concatenate(tempTuple, axis=-1)
        return Normalize(frameStack)
