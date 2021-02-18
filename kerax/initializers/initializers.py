from __future__ import absolute_import
import set_path
from jax.nn import initializers


class InitializerNotFoundException(Exception):
    def __repr__(self):
        return 'InitializerNotFoundException'




inits = {'zeros': initializers.zeros,'ones': initializers.ones,
                             'glorot_uniform': initializers.glorot_uniform(), 
                             'glorot_normal': initializers.glorot_normal(), 
                             'he_normal': initializers.he_normal(), 
                             'he_uniform': initializers.he_uniform(), 
                            'kaiming_normal': initializers.kaiming_normal(),
                            'kaiming_uniform': initializers.kaiming_uniform(),
                            'lecun_normal': initializers.lecun_normal(),
                            'lecun_uniform': initializers.lecun_uniform(),
                            'normal': initializers.normal()
                            }

def get(identifier):
    if identifier is None:
        return None
    if callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        result = inits.get(identifier, None)
        if not result:
            raise InitializerNotFoundException('Identifier is not found')
        return result
    else:
        raise Exception('Identifier should be string')


def normal(shape, stddev, key):
    return initializers.normal(stddev)(key=key, shape=shape)