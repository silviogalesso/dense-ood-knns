import torch
from .utils import FeatureProbe


class FeatureProbeViT(FeatureProbe):
    @staticmethod
    def _register_hooks(model, layer_index=-1):
        # model.module.backbone.layers[layer_index].attn.attn = MultiHeadAttentionExplicitQKV(
        #     model.module.backbone.layers[layer_index].attn.attn)

        selected_layer = model.module.backbone.layers[layer_index]
        assert selected_layer.attn.attn.in_proj_weight is not None and selected_layer.attn.attn.q_proj_weight is None
        num_extra_tokens = model.module.backbone.num_extra_tokens if hasattr(model.module.backbone, "num_extra_tokens") else 1

        def _extract_keys(x):
            # that's how mmseg vit computes attention...
            q, k, v = torch.nn.functional._in_projection_packed(x[0], x[0], x[0], selected_layer.attn.attn.in_proj_weight, selected_layer.attn.attn.in_proj_bias)
            return k[:,num_extra_tokens:].permute(1, 0, 2)

        feat_dict = dict()
        FeatureProbe._register_output_activation_hook(model.module.backbone.patch_embed,
                                                      lambda x: x[1],  # x[0] is the fwd pass tensor, we only need the shape
                                                      feat_dict,
                                                      "hw_shape")
        
        FeatureProbe._register_input_activation_hook(selected_layer.attn,
                                                     _extract_keys,
                                                     feat_dict,
                                                     "k")
        return feat_dict

    @staticmethod
    def _features_post_process(ft_dict):
        for k, ft in zip("qkv", ft_dict.pop("qkv", [])):
            ft_dict[k] = ft
        hw_shape = ft_dict.pop("hw_shape")
        return {k: ft.reshape(*hw_shape, -1) for k, ft in ft_dict.items()}


class FeatureProbeViTSETR(FeatureProbe):
    @staticmethod
    def _register_hooks(model, layer_index=-1, eval_all_features=False):
        assert not eval_all_features

        feat_dict = dict()

        num_extra_tokens = 0            
        
        FeatureProbe._register_output_activation_hook(model.module.backbone.patch_embed,
                                               lambda x: x.shape,
                                               feat_dict,
                                               "input_shape")
        FeatureProbe._register_output_activation_hook(model.module.backbone.blocks[layer_index].attn.qkv,
                                               lambda x: x, 
                                               feat_dict,
                                               "qkv")
        return feat_dict
    
    @staticmethod
    def _features_post_process(ft_dict):
        B, C, H, W = ft_dict.pop("input_shape")
        qkv = ft_dict["qkv"] # B, N, 3*C
        k = qkv.view(H*W+1, 3, C)[1:,1].view(H, W, C)
        return {"k": k}