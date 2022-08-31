class CompoundOptimizers():

    def __init__(self, optimizers, schedulers=None):
        self.optimizers = optimizers
        if schedulers:
            self.schedulers = schedulers
        else:
            self.schedulers = []

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()
        for sch in self.schedulers:
            sch.step()
