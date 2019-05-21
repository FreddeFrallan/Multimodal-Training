import time

class PrintStopwatch():

    def __init__(self, startMessage):
        self.startTime = time.time()
        self.lastLapTime = time.time()
        if(startMessage != False):
            self.printStartMeasure(startMessage)

    def printTotalTime(self, message):
        print(message, time.time() - self.startTime)

    def printLapTime(self, message):
        print(message, time.time() - self.lastLapTime)
        self.lastLapTime = time.time()

    def printStartMeasure(self, message):
        self.lastLapTime = time.time()
        print(message)
