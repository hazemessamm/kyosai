from jax import vmap
from jax.experimental import stax #type: ignore
from jax.random import PRNGKey #type: ignore
from .core import Layer
from jax import numpy as jnp

class Conv2D(Layer):
    '''
    Convolutional Layer, (Layer Subclass)
    Params:
        - filters: Stores number of filters, accepts int
        - kernel_size: stores size of each filter, accepts int or tuple
        - strides: stores size of the strides, default (1,1), accepts int or tuple
        - padding: padding for the input, accepts "valid" or "same"
        - activation: stores the activation function, accepts activation as string or callable
        - kernel_initializer: stores the kernel initializer, default "glorot_uniform"
        - bias_initializer: stores the bias initializer, default "zeros"
        - key: stores Pseudo Random Generator Key, default PRNGKey(1)
        - input_dim_order: stores the order of the dimensions, default NHWC
            where N=Batch size, H=Height, W=Width and C=Number of channels
        - kernel_dim_order: stores the order of the dimensions, default HWIO
            where H=Height, W=Width, I=Input Size which is the number of channels of the input and O=Output size which is the number of the filters
        - output_dim_order: stores the order of the dimensions, default NHWC

    '''
    def __init__(self, filters, kernel_size, 
                 strides=(1,1), padding='valid', activation=None, 
                 kernel_initializer='glorot_uniform', bias_initializer='normal', 
                 key=PRNGKey(100), input_dim_order="NHWC", 
                 kernel_dim_order="HWIO", output_dim_order="NHWC", 
                 trainable=True, 
                 name=None, **kwargs):
        super(Conv2D, self).__init__(key=key, trainable=trainable, name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = self.get_activation(activation)
        self.kernel_initializer = self.get_initializer(kernel_initializer)
        self.bias_initializer = self.get_initializer(bias_initializer)
        self.built = False
        self.dimension_numbers = (input_dim_order, kernel_dim_order, output_dim_order)
        self._validate_init()
        #self.apply_fn = jit(self.apply_fn)
        shape = kwargs.pop('shape', False) or kwargs.pop('input_shape', False)
        if shape:
            self.build(shape)
    
    @property
    def kernel_shape(self):
        'Returns the kernel dimensions'
        return self._kernel_shape

    @property
    def bias_shape(self):
        'Returns the bias dimensions'
        return self._bias_shape

    def _validate_init(self):
        if isinstance(self.strides, int):
            self.strides = (self.strides, self.strides)
        elif isinstance(self.strides, tuple) and len(self.strides) == 1:
            self.strides += self.strides

        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)
        elif isinstance(self.kernel_size, tuple) and len(self.kernel_size) == 1:
            self.kernel_size += self.kernel_size

    def build(self, input_shape):
        'Initializes the Kernel and stores the Conv2D weights'
        if len(input_shape) == 3:
            input_shape = (None,*input_shape)
        elif len(input_shape) < 3:
            raise Exception(f'Expected input shape to be 3 dimensions (Height, Width, Channels), found {len(input_shape)} dimensions')
        elif len(input_shape) > 3:
            input_shape = (None, *input_shape[1:])
        
        init_fn, self.apply_fn = stax.GeneralConv(dimension_numbers=self.dimension_numbers, 
        filter_shape=self.kernel_size, padding=self.padding, out_chan=self.filters, W_init=self.kernel_initializer, b_init=self.bias_initializer)
        self._check_jit()
        #initializes the conv layer
        self.shape, self._params = init_fn(input_shape=input_shape, rng=self.key)
        self.input_shape = input_shape
        self._kernel_shape = self._params[0].shape
        self._bias_shape = self._params[1].shape
        self.built = True

    def call(self, inputs):
        'Used during training to pass the parameters while getting the gradients'
        self.output = self.apply_fn(self._params, inputs)
        if self.activation:
            self.output = self.activation(self.output)
        return self.output

    def call_with_external_weights(self, params, inputs):
        self.output =  self.apply_fn(params=params, inputs=inputs)
        if self.activation:
            self.output = self.activation(self.output)
        return self.output

    
    def __call__(self, inputs, **kwargs):
        if not hasattr(inputs, 'shape'):
            raise Exception("Inputs should be tensors, or use Input layer for configuration")
        #here it takes the previous layer to build the current layer
        if self.in_construction_mode(inputs):
            self.build(inputs.shape)
            #General function used to connect with the previous layer
            self.connect(inputs)
            #returns the current layer
            return self
        else:
            #if the inputs are tensors and the function it will be built and continue apply conv to it
            if not self.built:
                self.build(inputs.shape)

            if self.input_shape[-3:] != inputs.shape[-3:] and len(self.input_shape) == 4:
                raise Exception(f"Not expected shape, input dims should be {self.input_shape} found {inputs.shape}")
            else:
                return self.call(inputs)

    def __repr__(self):
        if self.built:
            return f"<Convolutional Layer with input shape {self.input_shape} and output shape {self.shape}>"
        else:
            return "<Convolutional Layer>"

class MaxPool2D(Layer):
    '''
    MaxPool Layer, (Layer subclass)
    Params:
        - pool_size: takes the pooling size, default (2,2), accepts int or tuple
        - strides: stores size of the strides, default (1,1), accepts int or tuple
        - padding: padding for the input, accepts "valid" or "same"
        - spec: store the layer specs
        - key: stores Pseudo Random Generator Key, default PRNGKey(1)
    '''

    def __init__(self, pool_size=(2,2), strides=None, padding='valid', spec=None, key=PRNGKey(1)):
        super(MaxPool2D, self).__init__(key=key)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.spec = spec
        self._validate_init()
        
    def _validate_init(self):
        if isinstance(self.pool_size, int):
            self.pool_size = (self.pool_size, self.pool_size)
        elif isinstance(self.pool_size, tuple) and len(self.pool_size) == 1:
            self.pool_size += self.pool_size

        if isinstance(self.strides, int):
            self.strides = (self.strides, self.strides)
        elif isinstance(self.strides, tuple) and len(self.strides) == 1:
            self.strides += self.strides
    
    def build(self, input_shape):
        #initializing maxpool
        init_fn, self.apply_fn = stax.MaxPool(window_shape=self.pool_size, padding=self.padding, strides=self.strides, spec=self.spec)
        #returns output shape, and the params
        self.shape, self._params = init_fn(input_shape=(1, *input_shape[1:]), rng=self.key)
        self.input_shape = input_shape
        self.shape = (None, *self.shape[1:])
        self.built = True
        self._check_jit()

    def call(self, inputs):
        self.output = self.apply_fn(self._params, inputs)
        return self.output

    def call_with_external_weights(self, params, inputs):
        self.output =  self.apply_fn(params=params, inputs=inputs)
        return self.output

    def __call__(self, inputs, **kwargs):
        if not hasattr(inputs, 'shape'):
            raise Exception("Inputs should be tensors, or use Input layer for configuration")
        if self.in_construction_mode(inputs):
            self.build(inputs.shape)
            self.connect(inputs)
            return self
        else:
            self.build(inputs.shape)
            if self.input_shape != inputs.shape:
                raise Exception(f"Not expected shape, input dims should be {self.input_shape} found {inputs.shape}")
            else:
                return self.apply_fn(inputs=inputs, params=self._params)
    def __repr__(self):
        if self.built:
            return f"<MaxPool Layer with input shape {self.input_shape} and output shape {self.shape}>"
        else:
            return "<MaxPool Layer>"

class AvgPool2D(Layer):
    pass

class Conv2DTranspose(Layer):
    pass

