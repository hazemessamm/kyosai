from jax.experimental import optimizers #type: ignore
from jax import value_and_grad, grad #type: ignore
from jax import jit #type: ignore

class Optimizer:
    '''
    Optimizer base class
    All optimizers should be a subclass from this class
    and all optimizers should follow JAX optimizers rules

    Params:
        - loss_fn: stores the loss function to get the gradients of the loss function with respect to the params
        - model: stores the model to update it's weights every step
        - learning_rate: stores the learning rate (step_size), default: 0.0001
    '''
    def __init__(self, loss_fn, model, learning_rate=0.0001):
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.model = model
        self.step_index = 0
        self.loss_grad = jit(value_and_grad(loss_fn))

    #this function should be implemented by the subclasses

    def apply_gradients(self, grads):
        raise NotImplementedError("This method should be implemented in a subclass and should not be called from the Optimizer base class")
    
    def increment_step_index(self):
        'Increment step index'
        self.step_index += 1 

    def get_gradients(self, x, y):
        'Returns the loss value and the gradients'
        # self.increment_step_index()
        return self.loss_grad(self.model.params, x, y)

class SGD(Optimizer):
    '''
    Optimizer subclass

    Params:
        - loss_fn: stores the loss function to get the gradients of the loss function with respect to the params
        - model: stores the model to update it's weights every step
        - learning_rate: stores the learning rate (step_size), default: 0.0001
    '''
    def __init__(self, loss_fn, model, learning_rate=0.0001):
        super(SGD, self).__init__(loss_fn=loss_fn, model=model, learning_rate=learning_rate)
        '''
        Initializes the Stochastic Gradient Descent
        returns initializer function, update function and get params function
        init function just takes the current model params
        update function takes the step index, gradients and the current optimizer state
        get params takes the optimizer state and returns the params
        '''
        self.init_fn, self.update_fn, self.get_params = optimizers.sgd(step_size=learning_rate)
        #declars optimizer state and takes model current trainable params
        self.optimizer_state = self.init_fn(model.params)
        self.update_fn = jit(self.update_fn)

    def apply_gradients(self, grads):
        'Updates the model params'
        #returns new optimizer state by calling the update function
        self.optimizer_state = self.update_fn(self.step_index, grads, self.optimizer_state)
        #Apply new weights on the model
        self.model.update_weights(self.get_params(self.optimizer_state))

class Adam(Optimizer):
    '''
    Optimizer subclass

    Params:
        - loss_fn: stores the loss function to get the gradients of the loss function with respect to the params
        - model: stores the model to update it's weights every step
        - learning_rate: stores the learning rate (step_size), default: 0.0001
        - beta_1: a positive scalar value for beta_1, the exponential decay rate for the first moment estimates, default: 0.9
        - beta_2: a positive scalar value for beta_2, the exponential decay rate for the second moment estimates, default: 0.999
        - epsilon: a positive scalar value for epsilon, a small constant for numerical stability, default: 1e-8
    '''
    def __init__(self, loss_fn, model, learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08):
        super(Adam, self).__init__(loss_fn=loss_fn, model=model, learning_rate=learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.init_fn, self.update_fn, self.get_params = optimizers.adam(step_size=learning_rate, b1=beta_1, b2=beta_2, eps=epsilon)
        self.optimizer_state = self.init_fn(model.params)
        self.update_fn = jit(self.update_fn)

    def apply_gradients(self, grads):
        self.optimizer_state = self.update_fn(self.step_index, grads, self.optimizer_state)
        self.model.update_weights(self.get_params(self.optimizer_state))
        

class Adagrad(Optimizer):
    '''
    Optimizer subclass

    Params:
        - loss_fn: stores the loss function to get the gradients of the loss function with respect to the params
        - model: stores the model to update it's weights every step
        - learning_rate: stores the learning rate (step_size), default: 0.0001
        - momentum: a positive scalar value for momentum
    '''
    def __init__(self, loss_fn, model, learning_rate=0.001, momentum=0.9):
        super(Adagrad, self).__init__(loss_fn=loss_fn, model=model, learning_rate=learning_rate)
        self.momentum = momentum
        self.init_fn, self.update_fn, self.get_params = optimizers.adagrad(step_size=learning_rate, momentum=momentum)
        self.optimizer_state = self.init_fn(model.params)
        self.update_fn = jit(self.update_fn)
    
    def apply_gradients(self, grads, step=0):
        self.optimizer_state = self.update_fn(self.step_index, grads, self.optimizer_state)
        self.model.update_weights(self.get_params(self.optimizer_state))


supported_optimizers = {
    'sgd': SGD,
    'adam': Adam,
    'adagrad': Adagrad
}

def get(identifier):
    if identifier is None:
        return None
    elif isinstance(identifier, Optimizer):
        return identifier
    else:
        optimizer = supported_optimizers.get(identifier, None)
        if optimizer is None:
            raise Exception('Cannot find the specified optimizer')
        return optimizer
