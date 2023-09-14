from .utils import FeatureProbe


class FeatureProbeConvNeXt(FeatureProbe):
    @staticmethod
    def _register_hooks(model, stages=(2,)):
        feat_dict = dict()
        for stage in stages:
            FeatureProbe._register_output_activation_hook(model.module.backbone.stages[stage],
                                                   lambda ft: ft[0].permute(1, 2, 0),
                                                   feat_dict,
                                                   f"stage{stage}")
        return feat_dict

    @staticmethod
    def _features_post_process(ft_dict):
        return ft_dict
