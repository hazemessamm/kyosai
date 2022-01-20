


class Trackable:
    layers = {}
    name_uids = {}
    models = {}
    depth = 0

    def __init__(self, name):
        base_class_name = self.__class__.__base__.__name__
        self.name = Trackable.get_uid(name)
        if base_class_name == 'Layer':
            Trackable.track_layer(self.name, self)
            self.depth = Trackable.depth
        elif name == 'Model' or name == 'Sequential':
            Trackable.track_model(self.name, self)

    @staticmethod
    def get_uid(string):
        Trackable.name_uids[string] = Trackable.name_uids.get(string, 0) + 1
        return f'{string}_{Trackable.name_uids[string]}'

    @staticmethod
    def track_model(name, model_instance):
        Trackable.models[name] = model_instance

    @staticmethod
    def track_layer(layer_name, layer_instance):
        Trackable.layers[layer_name] = layer_instance
        Trackable.depth += 1

    @staticmethod
    def reset():
        Trackable.layers.clear()
        Trackable.name_uids.clear()
        Trackable.models.clear()
        Trackable.depth = 0