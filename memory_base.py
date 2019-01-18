

class MemoryBase:
    def AddFrame(self, newFrameData):
        raise NotImplementedError

    def AddMemory(self, newFrameData, action, reward, gameOver):
        raise NotImplementedError
    
    def GetMemory(self, index):
        raise NotImplementedError

    def GetCurrentState(self):
        raise NotImplementedError
    
    def GetFramesForState(self, stateId):
        raise NotImplementedError

    def OnEpisodeStart(self):
        pass
        
    def OnEpisodeOver(self, totalReward):
        pass
    
