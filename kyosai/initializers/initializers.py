from jax.nn import initializers as jax_initializers  # type: ignore


class InitializerNotFoundException(Exception):
    def __repr__(self):
        return "InitializerNotFoundException"


class Initializer:
    def __call__(self, key, shape, dtype=None, **kwargs):
        raise NotImplementedError

    def get_config(self):
        return {}


class Zeros(Initializer):
    def __call__(self, key, shape, dtype=None):
        if dtype is None:
            dtype = "float32"
        return jax_initializers.zeros(key=key, shape=shape, dtype=dtype)


class Ones(Initializer):
    def __call__(self, key, shape, dtype=None):
        if dtype is None:
            dtype = "float32"

        return jax_initializers.ones(key=key, shape=shape, dtype=dtype)


class GlorotUniform(Initializer):
    def __call__(self, key, shape, dtype=None):
        if dtype is None:
            dtype = "float32"
        initializer_fn = jax_initializers.glorot_uniform()
        return initializer_fn(key, shape, dtype)


class GlorotNormal(Initializer):
    def __call__(self, key, shape, dtype=None):
        if dtype is None:
            dtype = "float32"
        initializer_fn = jax_initializers.glorot_normal()
        return initializer_fn(key, shape, dtype)


class HeNormal(Initializer):
    def __call__(self, key, shape, dtype=None):
        if dtype is None:
            dtype = "float32"
        initializer_fn = jax_initializers.he_normal()
        return initializer_fn(key, shape, dtype)


class HeUniform(Initializer):
    def __call__(self, key, shape, dtype=None):
        if dtype is None:
            dtype = "float32"
        initializer_fn = jax_initializers.he_uniform()
        return initializer_fn(key, shape, dtype)


class KaimingNormal(Initializer):
    def __call__(self, key, shape, dtype=None):
        if dtype is None:
            dtype = "float32"
        initializer_fn = jax_initializers.kaiming_normal()
        return initializer_fn(key, shape, dtype)


class KaimingUniform(Initializer):
    def __call__(self, key, shape, dtype=None):
        if dtype is None:
            dtype = "float32"
        initializer_fn = jax_initializers.kaiming_uniform()
        return initializer_fn(key, shape, dtype)


class LecunNormal(Initializer):
    def __call__(self, key, shape, dtype=None):
        if dtype is None:
            dtype = "float32"
        initializer_fn = jax_initializers.lecun_normal()
        return initializer_fn(key, shape, dtype)


class LecunUniform(Initializer):
    def __call__(self, key, shape, dtype=None):
        if dtype is None:
            dtype = "float32"
        initializer_fn = jax_initializers.lecun_uniform()
        return initializer_fn(key, shape, dtype)


class Normal(Initializer):
    def __call__(self, key, shape, dtype=None):
        if dtype is None:
            dtype = "float32"
        initializer_fn = jax_initializers.normal()
        return initializer_fn(key, shape, dtype)


class Uniform(Initializer):
    def __call__(self, key, shape, dtype=None):
        if dtype is None:
            dtype = "float32"
        initializer_fn = jax_initializers.uniform()
        return initializer_fn(key, shape, dtype)


supported_initializations = {
    "zeros": jax_initializers.zeros,
    "ones": jax_initializers.ones,
    "glorot_uniform": jax_initializers.glorot_uniform(),
    "glorot_normal": jax_initializers.glorot_normal(),
    "he_normal": jax_initializers.he_normal(),
    "he_uniform": jax_initializers.he_uniform(),
    "kaiming_normal": jax_initializers.kaiming_normal(),
    "kaiming_uniform": jax_initializers.kaiming_uniform(),
    "lecun_normal": jax_initializers.lecun_normal(),
    "lecun_uniform": jax_initializers.lecun_uniform,
    "normal": jax_initializers.normal(),
    "uniform": jax_initializers.uniform(),
}


def get(identifier):
    global supported_initializations

    if identifier is None:
        return None
    if callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        result = supported_initializations.get(identifier, None)
        if not result:
            raise InitializerNotFoundException("Identifier is not found")
        return result
    else:
        raise Exception("Identifier should be string")
