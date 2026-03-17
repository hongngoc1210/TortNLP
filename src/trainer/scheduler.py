class TeacherForcingScheduler:
    
    def __init__(self, start=1.0, end=0.0, epochs=10):

        self.start = start
        self.end = end
        self.epochs = epochs

    def get_eta(self, epoch):

        progress = min(epoch / self.epochs, 1.0)

        eta = self.start - progress * (self.start - self.end)

        return eta