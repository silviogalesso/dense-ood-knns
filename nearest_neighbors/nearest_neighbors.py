import os
from time import time
from tqdm import tqdm
import numpy as np
import random
import logging
from mmseg.datasets import build_dataloader
# from .subsampling import greedy_coreset_subsampling, per_class_greedy_coreset_subsampling
from .utils import *
try:
    import faiss
except ModuleNotFoundError:
    pass
import torch


def get_reference_features(model, dataset, feature_probe, num_samples, ood_scoring_fn_for_scaling):
    ft_list = []
    avg_valids = []
    coordinates = []
    labels_list = []
    max_param_score = -np.inf; min_param_score = np.inf

    indices = random.sample(range(len(dataset)), num_samples)
    subset = torch.utils.data.Subset(dataset, indices)
    loader_cfg = dict(
        dist=False,
        shuffle=False,
        samples_per_gpu=1,
        workers_per_gpu=1
    )
    loader = build_dataloader(subset, **loader_cfg)
    model.eval()
    for smpl_i, (index, sample) in tqdm(enumerate(zip(indices, loader))):
        smpl_ft_list = []
        smpl_coords = []
        smpl_labels = []
        logits = fwd_pass_mmseg(sample, model)
        max_param_score = max(max_param_score, ood_scoring_fn_for_scaling(logits).max().item())
        min_param_score = min(min_param_score, ood_scoring_fn_for_scaling(logits).min().item())
        proc_act_dict = feature_probe.get_features()
        features = [proc_act_dict[k] for k in proc_act_dict]
        segm = torch.tensor(dataset.get_gt_seg_map_by_idx(index))
        for ft in features:
            h, w, c = ft.shape
            downsampled_segm = torch.nn.functional.interpolate(segm.view((1, 1, *segm.shape)).byte(), size=(h, w))
            valid = (downsampled_segm.squeeze() != 255).flatten()
            smpl_ft_list.append(ft.view(h * w, c).cpu()[valid])
            smpl_coords.append(np.stack(np.meshgrid(range(w), range(h))+[(np.ones((h, w))*smpl_i)], -1).reshape(h*w, 3)[valid])
            smpl_labels.append(downsampled_segm.flatten()[valid])
        ft_list.append(smpl_ft_list)
        coordinates.append(smpl_coords)
        avg_valids.append(valid.float().mean())
        labels_list.append(smpl_labels)

    ft_lists = zip(*ft_list)  # from list of feature tuples, to lists of features (n_ft, n_imgs, samples_per_img)
    lb_lists = zip(*labels_list)  # from list of label tuples, to lists of labels (n_ft, n_imgs, samples_per_img)
    ft_dict = {k: torch.cat(fl, dim=0) for k, fl in zip(proc_act_dict.keys(), ft_lists)}  # concatenate for each ft type
    lb_dict = {k: torch.cat(ll, dim=0) for k, ll in zip(proc_act_dict.keys(), lb_lists)}  # concatenate for each ft type

    return ft_dict, lb_dict, ((min_param_score), (max_param_score))


def subsample_features(features_dict, labels_dict, subsample_number, mode="random"):
    output_features = dict()
    output_labels = dict()
    for k, ft in features_dict.items():
        assert len(ft) == len(labels_dict[k])
        assert subsample_number <= len(ft), f"Subsample number ({subsample_number}) is larger than the number of features ({len(ft)})"
        if mode == 'random':
            ids = random.sample(range(len(ft)), subsample_number)
            output_features[k] = ft[ids]
            output_labels[k] = labels_dict[k][ids]
        else:
            raise NotImplementedError("Mode: "+mode)

    return output_features, output_labels



class kNNEngine:
    def __init__(self, reference_features, num_neighbors, dist_type, reduction='mean'):
        self.num_neighbors = num_neighbors
        self.dist_type = dist_type
        self.reduction = reduction
        assert len(reference_features.shape) == 2, "Features should be in (N, c) shape"
        self.reference_features = reference_features
        self.search_runtimes = []
    
    def compute_nn_scores(self, test_features):
        raise NotImplementedError

    def compute_min_max_nndist(self, reference_features, max_samples=30000):
        raise NotImplementedError
    
    def reduce(self, nn_dists):
        if self.reduction == "max":
            return nn_dists.max(-1)[0]
        elif self.reduction == "mean":
            return nn_dists.mean(-1)
        else:
            raise ValueError

    def __repr__(self):
        return f"features: {self.reference_features.shape[0]}x{self.reference_features.shape[1]}, k={self.num_neighbors}, dist={self.dist_type}"


class kNNEngineTorch(kNNEngine):
    def __init__(self, reference_features, num_neighbors, dist_type, reduction='mean'):
        super().__init__(reference_features, num_neighbors, dist_type, reduction)
        self.reference_features = self.reference_features.cuda()

    def _compute_dists(self, ft1, ft2):
        assert len(ft1.shape) == len(ft2.shape) == 2, "Features should be in (N, c) shape"
        if self.dist_type in ["euclidean", "l2"]:
            dists = torch.cdist(ft1.unsqueeze(0), ft2.unsqueeze(0))[0]
        elif self.dist_type == "l1":
            dists = torch.cdist(ft1.unsqueeze(0), ft2.unsqueeze(0), p=1)[0]
        return dists

    def compute_min_max_nndist(self, reference_features, max_samples=30000):
        if len(reference_features) > max_samples:
            reference_features = reference_features[random.sample(range(len(reference_features)), max_samples)]
        reference_features = reference_features.cuda()
        dists = self._compute_dists(reference_features, reference_features)
        nn_dists = torch.topk(dists, self.num_neighbors + 1, dim=-1, largest=False, sorted=True).values[:,1:]
        nn_dists = self.reduce(nn_dists)
        return nn_dists.min().item(), nn_dists.max().item()

    def compute_nn_scores(self, test_features):
        assert len(test_features.shape) == 3, "Test features should be in (h, w, c) shape"
        t0 = time()
        dists = self._compute_dists(test_features.flatten(0, 1).cuda(), self.reference_features)
        nn_dists = torch.topk(dists, self.num_neighbors, dim=-1, largest=False).values
        nn_dists = self.reduce(nn_dists).cpu()
        torch.cuda.synchronize()
        self.search_runtimes.append((time() - t0))
        return nn_dists.view(test_features.shape[:2])

    def __repr__(self):
        return "kNNEngineNaive "+super().__repr__()
    

class kNNEngineFAISS(kNNEngine):
    def __init__(self, reference_features, num_neighbors, dist_type, index_string, reduction='mean'):
        super().__init__(reference_features, num_neighbors, dist_type, reduction)

        res = faiss.StandardGpuResources()
        d = self.reference_features.shape[-1]

        # extract nprobe if in string
        if "/" in index_string:
            nprobe = int(index_string.split("/")[-1])
            index_string = index_string.split("/")[0]
        else:
            nprobe = None

        if dist_type in ["l2", "euclidean"]:
            self.index = faiss.index_factory(d, index_string)
        elif dist_type == "l1":
            assert index_string == "Flat"
            self.index = faiss.IndexFlat(d, faiss.METRIC_L1)
        else:
            raise ValueError
    
        self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        if index_string != "Flat":
            self.index.train(self.reference_features.numpy())
        self.index.add(self.reference_features.contiguous().numpy())
        assert self.index.is_trained
        if nprobe:
            self.index.nprobe = nprobe

    def compute_min_max_nndist(self, reference_features, max_samples=30000):
        if len(reference_features) > max_samples:
            reference_features = reference_features[random.sample(range(len(reference_features)), max_samples)]
        min_max_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss.IndexFlatL2(reference_features.shape[-1]))
        min_max_index.add(reference_features.numpy())
        nn_dists = np.sqrt(min_max_index.search(reference_features.numpy(), self.num_neighbors+1)[0][:,1:])
        nn_dists = self.reduce(nn_dists)
        return nn_dists.min().item(), nn_dists.max().item()
    
    def compute_nn_scores(self, test_features):
        assert len(test_features.shape) == 3, "Test features should be in (h, w, c) shape"
        test_features = test_features
        t0 = time()
        nn_dists, _ = self.index.search(test_features.flatten(0, 1).cpu().numpy(), self.num_neighbors)
        self.search_runtimes.append((time() - t0))
        nn_dists = self.reduce(torch.tensor(np.sqrt(nn_dists)))
        return nn_dists.view(test_features.shape[:2])
    
    def __repr__(self):
        return "kNNEngineFAISS "+super().__repr__()
