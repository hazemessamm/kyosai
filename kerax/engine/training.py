from kerax.engine.tracker import Trackable


class Model(Trackable):
    def __init__(self, name: str = None):
        super().__init__(name=name)
