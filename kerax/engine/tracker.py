class Trackable:
    __layers = {}
    __name_uids = {}
    __models = {}
    __depth = 0

    def __init__(self, name):
        base_class_name = self.__class__.__base__.__name__
        self.name = Trackable.get_uid(name)
        if base_class_name == "Layer":
            Trackable.track_layer(self.name, self)
            self.depth = Trackable.__depth
        elif name == "Model" or name == "Sequential":
            Trackable.track_model(self.name, self)

    @staticmethod
    def get_uid(string):
        Trackable.__name_uids[string] = Trackable.__name_uids.get(string, 0) + 1
        return f"{string}_{Trackable.__name_uids[string]}"

    @staticmethod
    def track_model(name, model_instance):
        Trackable.__models[name] = model_instance

    @staticmethod
    def track_layer(layer_name, layer_instance):
        Trackable.__layers[layer_name] = layer_instance
        Trackable.__depth += 1

    @staticmethod
    def reset():
        Trackable.__layers.clear()
        Trackable.__name_uids.clear()
        Trackable.__models.clear()
        Trackable.__depth = 0
