from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
import torch
import numpy as np


def fpr_at_tpr(scores, ood_gt, tpr=0.95):
    fprs, tprs, _ = roc_curve(ood_gt, scores)
    idx = np.argmin(np.abs(np.array(tprs) - tpr))
    return fprs[idx]


def f_max_score(scores, ood_gt):
    precision, recall, thresholds = precision_recall_curve(ood_gt, scores)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-6)
    f_max = np.max(f1_scores)
    return f_max


def ap(scores, ood_gt):
    return average_precision_score(ood_gt, scores)


def auroc(scores, target):
    return roc_auc_score(target, scores)


class StreamingEval:
    def __init__(self, ood_id, ignore_ids):
        self.ood_id = ood_id
        self.collected_scores = []
        self.collected_gts = []
        if isinstance(ignore_ids, int):
            ignore_ids = [ignore_ids]
        self.ignore_ids = ignore_ids

    def add(self, scores, segm_gt):
        valid = np.logical_not(np.in1d(segm_gt, self.ignore_ids))
        ood_gt = (segm_gt == self.ood_id)
        if not isinstance(scores, torch.Tensor):
            scores = torch.tensor(scores)
        if not isinstance(ood_gt, torch.Tensor):
            ood_gt = torch.tensor(ood_gt)
        self.collected_scores.append(scores.cpu().flatten()[valid])
        self.collected_gts.append(ood_gt.cpu().flatten()[valid])

    def get_scores_and_labels(self):
        all_scores = torch.cat(self.collected_scores, 0)
        all_gts = torch.cat(self.collected_gts, 0)
        return all_scores, all_gts

    def get_results(self):
        scores, labels = self.get_scores_and_labels()
        return auroc(scores, labels.int()) * 100, ap(scores, labels.int()) * 100, fpr_at_tpr(scores, labels.int()) * 100


def prediction_entropy(probs, eps=1e-6):
    assert len(probs.shape) == 4, "Required: probs (b, c, h, w), got shape {} instead".format(probs.shape)
    h = -(probs * (probs + eps).log()).sum(1)
    return h


ood_score_functions = {
    "max_softmax": lambda l: 1 - l.softmax(dim=1).max(dim=1)[0].detach(),
    "entropy": lambda l: prediction_entropy(l.softmax(1)).detach(),
    "logsumexp": lambda l: 1-l.logsumexp(dim=1).detach(),
    "max_logit": lambda l: - l.max(dim=1)[0].detach()
}
