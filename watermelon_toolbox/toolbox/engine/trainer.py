class Trainer:
    def __init__(self, model, dataloader, optimizer):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer

    def fit(self, epochs: int):
        raise NotImplementedError
