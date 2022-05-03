from setuptools import setup, find_packages


setup(
    name="kerax",
    version=0.1,
    description="Deep Learning library based on JAX and inspired from Keras",
    long_description="Simple Deep Learning library based on JAX and inspired from Keras, the main idea behind it to have the simplicity of Keras while using JAX",
    author="Hazem Essam",
    author_email="hazemessam199@gmail.com",
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'numpy',
        'jax==0.3.4',
        'jaxlib==0.3.2',
        'optax',
        'tqdm'
    ],
    url='https://github.com/hazemessamm/kerax',
)