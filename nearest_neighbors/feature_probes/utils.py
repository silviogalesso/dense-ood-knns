import torch

class FeatureProbe:
    def __init__(self, model, **kwargs):
        # hooks, pre-post fns, get features
        self.activation_dict = self._register_hooks(model, **kwargs)

    def get_features(self):
        return self._features_post_process(self.activation_dict)

    @staticmethod
    def _register_input_activation_hook(module, feature_extraction_fn, output_dict, key):
        def activation_hook(self, x, y):
            output_dict[key] = feature_extraction_fn(x)
        module.register_forward_hook(activation_hook)
    
    @staticmethod
    def _register_output_activation_hook(module, feature_extraction_fn, output_dict, key):
        def activation_hook(self, x, y):
            output_dict[key] = feature_extraction_fn(y)
        module.register_forward_hook(activation_hook)

    @staticmethod
    def _register_hooks(model, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _features_post_process(ft_dict):
        raise NotImplementedError

