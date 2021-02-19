class Optimizer:
    def __init__(self, learning_rate=0.0001):
        self.learning_rate = learning_rate
    
    def apply_grads(self, grads, model):
        
