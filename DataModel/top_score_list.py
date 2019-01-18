


class TopScoreList:
    def __init__(self, maxSize):
        self.MaxSize = maxSize
        self.Contents = []
    
    def Add(self, newEpisode):
        self.Contents.append(newEpisode)
        removed = None
        if(len(self.Contents) > self.MaxSize):
            removed = self.RemoveLowestEntry()
        return removed
    
    def RemoveLowestEntry(self):
        lowValue = self.Contents[0].EpisodeReward
        lowIndex = 0
        for index in range(1,self.MaxSize):
            if(self.Contents[index].EpisodeReward < lowValue):
                lowValue = self.Contents[index].EpisodeReward
                lowIndex = index
        removed = self.Contents.pop(lowIndex)
        return removed

    def FindEpisodeContainingFrame(self, startFrameId):
        for e in self.Contents:
            if(e.ContainsFrame(startFrameId)):
                return e
        return None
