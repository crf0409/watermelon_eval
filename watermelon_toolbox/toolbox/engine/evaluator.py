class Evaluator:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader

    def evaluate(self):
        raise NotImplementedError
