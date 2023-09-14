# import faiss
import os
import argparse
import random
import torch
import numpy as np
from tqdm import tqdm
import warnings
from pprint import pprint

import mmcv
from mmcv import DictAction
from mmcv.cnn.utils import revert_sync_batchnorm
from mmseg.models import build_segmentor
from mmseg.datasets import build_dataset
from mmseg.utils import build_dp
import models

# from nearest_neighbors import get_reference_features, compute_nn_scores, subsample_features, compute_min_max_nndist
from nearest_neighbors import get_reference_features, subsample_features, kNNEngineTorch  #, kNNEngineFAISS
from nearest_neighbors.utils import upsample_scores, combine_scores
from nearest_neighbors.feature_probes import FeatureProbeConvNeXt, FeatureProbeViT, FeatureProbeViTSETR

from utils import StreamingEval, ood_score_functions, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(
        description='Eval a model for out-of-distribution segmentation')

    parser.add_argument('config', help='test config file path, for the model and for the reference data if --reference-data-config is not set')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('test_data_config', help="config file for the test dataset")

    parser.add_argument('--reference-data-config', default=None, help='Config file for the reference dataset. If not provided, the main/model config is used.')
    parser.add_argument('--test-data-cfg-split', type=str, default='test')
    parser.add_argument('--nn-k', type=int, default=3, help='number of nearest neighbors')
    parser.add_argument('--nn-reference-samples', type=int, default=500, help='number of NN reference samples')
    parser.add_argument('--nn-subsample-mode', type=str, default="random", choices=["random"])
    parser.add_argument('--nn-subsample-number', type=int, default=100000)
    parser.add_argument('--nn-dist-type', default="euclidean")
    parser.add_argument('--parametric-score-fn', default="logsumexp")
    # parser.add_argument('--faiss-index', default="Flat", type=str, help="the string defining the faiss index type")
    parser.add_argument('--random-seed', default=0, type=int)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override config setting, see mmcv/mmseg')
    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    return args


def main():
    args = parse_args()

    print("checkpoint:", args.checkpoint)
    print("config:", args.config)

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    model = build_segmentor(cfg.model).eval().cuda()
    load_checkpoint(model, args.checkpoint)
    model = revert_sync_batchnorm(model)
    model = build_dp(model, 'cuda')

    ## DATASETS
    # test data
    test_data_config = mmcv.Config.fromfile(args.test_data_config)
    ood_idx = test_data_config.ood_index
    test_data_config = getattr(test_data_config.data, args.test_data_cfg_split)
    test_dataset = build_dataset(test_data_config)
    test_dataset.ood_idx = ood_idx

    # reference data
    reference_data_config = cfg.data.train if not args.reference_data_config else mmcv.Config.fromfile(args.reference_data_config).data.train
    if isinstance(reference_data_config, list): reference_data_config = reference_data_config[0]
    reference_data_config.pipeline = test_data_config.pipeline
    reference_dataset = build_dataset(reference_data_config)

    ## NNs
    if cfg.model.backbone.type.startswith("ResNet"):
        raise NotImplementedError
    elif "ConvNeXt" in cfg.model.backbone.type:
        feature_probe = FeatureProbeConvNeXt(model)
    elif cfg.model.backbone.type in ["MixVisionTransformer"]:
        raise NotImplementedError
    elif cfg.model.backbone.type in ["VisionTransformer", "DistilledVisionTransformer", "SETR"]:
        feature_probe = FeatureProbeViT(model)
    elif cfg.model.backbone.type == "VisionTransformerSETR":
        feature_probe = FeatureProbeViTSETR(model)
    else:
        raise ValueError(cfg.model.backbone.type)
    print('Registered activation hook(s)')

    # gather reference features dataset
    print(f"Getting reference features")
    reference_features, reference_labels, (min_score, max_score) = get_reference_features(
        model, reference_dataset, feature_probe,
        num_samples=args.nn_reference_samples,
        ood_scoring_fn_for_scaling=ood_score_functions[args.parametric_score_fn]
    )
    print(f"Extracted features from {args.nn_reference_samples} images")
    
    # subsample features
    if args.nn_subsample_number > 0:
        subsampled_features, subsampled_labels = subsample_features(
            reference_features, reference_labels,
            args.nn_subsample_number,
            args.nn_subsample_mode,
        )
    else:
        subsampled_features = reference_features
    [print(f"Subsampled '{k}' features to {len(ft)}") for k, ft in subsampled_features.items()]

    # create kNN engines, get nn dist stats
    knnengines = {k: kNNEngineTorch(ref_ft, args.nn_k, args.nn_dist_type, reduction="mean") for k, ref_ft in subsampled_features.items()}
    min_max_nndist = {k: knnengines[k].compute_min_max_nndist(ft) for k, ft in reference_features.items()}

    for k, eng in knnengines.items():
        print(f"features '{k}', engine: {eng}")

    eval_ood(model, test_dataset, feature_probe,
             knn_engines=knnengines,
             score_fn=args.parametric_score_fn,
             scores_min_max=(min_score, max_score),
             nn_min_max=min_max_nndist,
             )


def eval_ood(model, test_dataset, feature_probe, knn_engines, score_fn,
             scores_min_max, nn_min_max):
    model.eval()

    ood_eval_param = StreamingEval(test_dataset.ood_idx, ignore_ids=255)
    ood_eval_dnp = {k: StreamingEval(test_dataset.ood_idx, ignore_ids=255) for k in knn_engines.keys()}
    ood_eval_cdnp = {k: StreamingEval(test_dataset.ood_idx, ignore_ids=255) for k in knn_engines.keys()}

    progress = tqdm(total=len(test_dataset))
    for sample_index, sample in enumerate(test_dataset):
        segm = test_dataset.get_gt_seg_map_by_idx(sample_index)
        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            logits = model.module.whole_inference(sample["img"][0].unsqueeze(0).cuda(), sample["img_metas"],
                                                  rescale=False).cpu()
        param_scores = {k: ood_score_functions[k](logits) for k in ood_score_functions}
        ood_eval_param.add(param_scores[score_fn], segm)
        
        # DNP, cDNP
        test_features = feature_probe.get_features()
        for ft_k, ft in test_features.items():
            nn_scores = knn_engines[ft_k].compute_nn_scores(ft)
            nn_scores = upsample_scores(nn_scores, segm)
            ood_eval_dnp[ft_k].add(nn_scores, segm)

            comb_scores = combine_scores(param_scores[score_fn], nn_scores, *scores_min_max, 0, nn_min_max[ft_k][1])  # min dist is 0
            ood_eval_cdnp[ft_k].add(comb_scores, segm)

        progress.update()
    progress.close()

    print(f"Max GPU memory allocated: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    for k, e in knn_engines.items():
        if e.search_runtimes:
            print(f"Avg search time {k} = {np.mean(e.search_runtimes):03f}")

    print("Results:")
    print("parametric:  AP = {:.02f}, FPR95 = {:.02f}".format(*ood_eval_param.get_results()[1:]))
    for k in ood_eval_dnp.keys():
        print(f"ft {k}")
        print("       DNP:  AP = {:.02f}, FPR95 = {:.02f}".format(*ood_eval_dnp[k].get_results()[1:]))
        print("      cDNP:  AP = {:.02f}, FPR95 = {:.02f}".format(*ood_eval_cdnp[k].get_results()[1:]))


if __name__ == '__main__':
    main()
