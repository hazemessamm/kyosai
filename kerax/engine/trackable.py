

class Tracker:
    num_layers = 0
    num_models = 0
    layers = []
    models = {}

    @staticmethod
    def reset():
        Tracker.num_layers = 0
        Tracker.num_models = 0
        Tracker.layers = []

    @staticmethod
    def track_model(model_class):
        def wrapper(*args, **kwargs):
            Tracker.num_models += 1
            model_instance = model_class(*args, **kwargs)
            Tracker.models[f'model_{Tracker.num_models}'] = model_instance
            return model_instance
        return wrapper


