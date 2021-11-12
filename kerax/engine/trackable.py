

class Trackable:
    layers = {}
    name_uids = {}
    models = {}
    depth = 0

    @staticmethod
    def get_uid(string):
        if string in Trackable.name_uids:
            Trackable.name_uids[string] += 1
        else:
            Trackable.name_uids[string] = 1
        return string+'_'+str(Trackable.name_uids[string])

    @staticmethod
    def track_model(model_instance):
        Trackable.num_models += 1
        Trackable.models[f'{model_instance.__class__.__name__}_{Trackable.num_models}'] = model_instance
        Trackable.reset()

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