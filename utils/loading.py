import torch

def load_checkpoint(model, path):
    ckpt = torch.load(path)
    model_state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    if "backbone.dist_token" in model_state_dict:
        # Distilled ViT (DEIT), discard distillation token
        del model_state_dict["backbone.dist_token"]
        if model_state_dict["backbone.pos_embed"].shape[1] == model.backbone.pos_embed.shape[1]+1:
            # discard positional embedding for distillation token
            model_state_dict["backbone.pos_embed"] = torch.cat([model_state_dict["backbone.pos_embed"][:,:1], 
                                                                model_state_dict["backbone.pos_embed"][:,2:]], dim=1)
    model.load_state_dict(model_state_dict)