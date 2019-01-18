
from memory_base import MemoryBase
from DataModel.single_memory import SingleMemory
from DataModel.single_frame import SingleFrame
from DataModel.single_episode import SingleEpisode
from DataModel.top_score_list import TopScoreList
from DataModel.ddqn_globals import DdqnGlobals

class PrioritizedMemory(MemoryBase):
    def __init__(self):
        self.TopScoreSize = 10
        self.BatchMemoryBuffer = []
        self.TopScoreEpisodes = TopScoreList(self.TopScoreSize)
        self.ActiveEpisode = None
        self.MaxFrameId = 0

    @property
    def Memories(self):
        return self.BatchMemoryBuffer

    def AddMemory(self, newFrameData, action, reward, gameOver):
        newFrameId = self.AddFrame(newFrameData)
        startFrameId = newFrameId - DdqnGlobals.FRAMES_PER_STATE
        assert (self.ActiveEpisode.ContainsFrame(startFrameId)), "Frame buffer out of sync with memory state."
                
        theMemory = SingleMemory(int(startFrameId), int(action), int(reward), gameOver)
        self.ActiveEpisode.AddMemory(theMemory)

    def GetMemory(self, index):
        return self.BatchMemoryBuffer[index]

    def AddFrame(self, newFrameData):
        self.MaxFrameId += 1
        theFrame = SingleFrame(self.MaxFrameId, newFrameData)
        self.ActiveEpisode.AddFrame(theFrame)
        return theFrame.id
 
    def OnEpisodeStart(self):
        self.ActiveEpisode = SingleEpisode()
        
    def OnEpisodeOver(self, totalReward):
        self.ActiveEpisode.EpisodeReward = totalReward
        removed = self.TopScoreEpisodes.Add(self.ActiveEpisode)
        if(removed == None):
            self.AddMemoriesToBuffer(self.ActiveEpisode)
        elif(removed != self.ActiveEpisode):
            self.RebuildBatchMemoryBuffer()

    def AddMemoriesToBuffer(self, iEpisode):
        self.BatchMemoryBuffer.extend(iEpisode.Memories)

    def RebuildBatchMemoryBuffer(self):
        self.BatchMemoryBuffer = []
        for e in self.TopScoreEpisodes.Contents:
            self.AddMemoriesToBuffer(e)

    def GetCurrentState(self):
        currentStateId = (self.MaxFrameId - DdqnGlobals.FRAMES_PER_STATE) + 1
        assert (currentStateId > 0), "Not enough frames to define a complete state."
        assert (self.ActiveEpisode.ContainsFrame(currentStateId)), "Unable to locate frame for current state."
        return self.ActiveEpisode.GetFramesForState(currentStateId)
    
    def FindEpisodeContainingFrame(self, startFrameId):
        found = self.TopScoreEpisodes.FindEpisodeContainingFrame(startFrameId)
        assert(found != None), "Unable to locate frame for batch."
        return found
    
    def GetFramesForState(self, stateId):
        episode = self.FindEpisodeContainingFrame(stateId)
        return episode.GetFramesForState(stateId)

    def GetFramesForLatestEpisode(self):
        frames = []
        for frameId in range(self.ActiveEpisode.MinFrameId, self.ActiveEpisode.MaxFrameId+1):
            frames.append(self.ActiveEpisode.Frames[frameId])
        return frames

