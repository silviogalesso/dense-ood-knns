import mmcv
import warnings
import torch


def fwd_pass_mmseg(data_sample, model):
    image = data_sample["img"]
    if isinstance(image, mmcv.parallel.data_container.DataContainer):
        image = image.data
    assert isinstance(image, list) and len(image) == 1, "Should be single image in a list... " + str(type(image)) + str(
        len(image))
    image = image[0]
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    imgh, imgw = image.shape[2:]

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        logits = model.whole_inference(image.cuda(), data_sample["img_metas"], rescale=False).cpu()

    return logits


def upsample_scores(scores, reference):
    return torch.nn.functional.interpolate(scores[None, None, ...], size=reference.shape).squeeze()


def combine_scores(max_logit, nn_dist, min_ml, max_ml, min_nn, max_nn):
    ml_ = ((max_logit - min_ml) / (max_ml - min_ml))
    nn_ = ((nn_dist-min_nn)/(max_nn-min_nn))
    return ml_+nn_

