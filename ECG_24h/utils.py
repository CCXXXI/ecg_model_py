# -*- coding: utf-8 -*-
"""
@time: 2019/9/12 15:16

@ author: javis
"""
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from torch import nn


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


# 计算F1score
def calc_f1(y_true, y_pre, threshold=0.5):
    y_true = (
        y_true.view(-1).cpu().detach().numpy().astype(np.int32)
    )  # 将矩阵拉直，实际相当于整体计算F1，即微平均
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return f1_score(y_true, y_pre)


def calc_acc(y_true, y_pre, threshold=0.5):
    y_true = (
        y_true.view(-1).cpu().detach().numpy().astype(np.int32)
    )  # 将矩阵拉直，实际相当于整体计算ACC，即微平均
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return accuracy_score(y_true, y_pre)


def calc_sensitivity(y_true, y_pre, threshold=0.5):
    y_true = (
        y_true.view(-1).cpu().detach().numpy().astype(np.int32)
    )  # 将矩阵拉直，实际相当于整体计算敏感度，即微平均
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return recall_score(y_true, y_pre)


def calc_precision(y_true, y_pre, threshold=0.5):
    y_true = (
        y_true.view(-1).cpu().detach().numpy().astype(np.int32)
    )  # 将矩阵拉直，实际相当于整体计算精确率，即微平均
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return precision_score(y_true, y_pre)


def calc_f1_npy(y_true, y_pre, threshold=0.5):
    y_true = y_true.astype(np.int32)  # 将矩阵拉直，实际相当于整体计算F1，即微平均
    y_pre = y_pre > threshold
    return f1_score(y_true, y_pre)


def calc_acc_npy(y_true, y_pre, threshold=0.5):
    y_true = y_true.astype(np.int32)  # 将矩阵拉直，实际相当于整体计算acc，即微平均
    y_pre = y_pre > threshold
    return accuracy_score(y_true, y_pre)


def calc_sensitivity_npy(y_true, y_pre, threshold=0.5):
    y_true = y_true.astype(np.int32)  # 将矩阵拉直，实际相当于整体计算灵敏度，即微平均
    y_pre = y_pre > threshold
    return recall_score(y_true, y_pre)


def calc_precision_npy(y_true, y_pre, threshold=0.5):
    y_true = y_true.astype(np.int32)  # 将矩阵拉直，实际相当于整体计算精确度，即微平均
    y_pre = y_pre > threshold
    return precision_score(y_true, y_pre)


# 打印时间
def print_time_cost(since):
    time_elapsed = time.time() - since
    return "{:.0f}m{:.0f}s\n".format(time_elapsed // 60, time_elapsed % 60)


# 调整学习率
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


# 多标签使用类别权重
class WeightedMultilabel(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super(WeightedMultilabel, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction="none")
        self.weights = weights

    def forward(self, outputs, targets):
        loss = self.cerition(outputs, targets)
        return (loss * self.weights).mean()


# focal loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        # def __init__(self, alpha=0.25, gamma=1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.sigmoid = nn.Sigmoid()

    def forward(self, y_pred, y_true):
        # if 'DEBUG' in dir(config) and config.DEBUG:
        #     np.savetxt('output_debug.csv', y_pred.cpu().detach().numpy(), delimiter=',')
        y_pred = self.sigmoid(y_pred)  # 之前有sigmoid的话记得注释掉这一句

        # if 'DEBUG' in dir(config) and config.DEBUG:
        #     np.savetxt('y_pred_debug.csv', y_pred.cpu().detach().numpy(), delimiter=',')

        # 防止梯度爆炸
        y_pred = y_pred.clamp(min=0.0001, max=0.9999)

        # if 'DEBUG' in dir(config) and config.DEBUG:
        #     np.savetxt('y_pred_debug.csv', y_pred.cpu().detach().numpy(), delimiter=',')

        fl = -self.alpha * y_true * torch.log(y_pred) * (
            (1.0 - y_pred) ** self.gamma
        ) - (1.0 - self.alpha) * (1.0 - y_true) * torch.log(1.0 - y_pred) * (
            y_pred**self.gamma
        )
        fl_sum = fl.sum()
        return fl_sum


class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        # def __init__(self, alpha=0.25, gamma=1):
        super(MultiClassFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        # important to add reduction='none' to keep per-batch-item loss
        # ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
        ce_loss = F.cross_entropy(y_pred, y_true, reduction="none")
        pt = torch.exp(-ce_loss)
        # mean over the batch
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


# 标签平滑后使用类别权重的交叉熵
class WeightedMultilabelWithLabelSmooth(nn.Module):
    def __init__(self, weights: torch.Tensor, epsilon=0.1):
        super(WeightedMultilabelWithLabelSmooth, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction="none")
        self.weights = weights
        self.epsilon = epsilon

    def forward(self, outputs, targets):
        new_tars = targets * (1 - self.epsilon) + self.epsilon / targets.shape[-1]

        loss = self.cerition(outputs, new_tars)
        return (loss * self.weights).mean()


# class WeightedMultiLabelSoftMarginLoss(nn.Module):
#     def __init__(self, weights: torch.Tensor):
#         super(WeightedMultiLabelSoftMarginLoss, self).__init__()
#         self.cerition = nn.MultiLabelSoftMarginLoss(reduction='none')
#         self.weights = weights
#
#     def forward(self, outputs, targets):
#         loss = self.cerition(outputs, targets)
#         return (loss * self.weights).mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(
        self, device, temperature=0.07, contrast_mode="all", base_temperature=0.07
    ):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)  # labels [bsz, 1]
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = (
                torch.eq(labels, labels.T).float().to(self.device)
            )  # [bsz, bsz] mask[i][j] = 0 or 1, means weather record[i] and record[j] belong to the same label class
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(
            torch.unbind(features, dim=1), dim=0
        )  # [bsz+bsz, 128] i and i+bsz are two features from the same source record
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )  # [bsz*2, bsz*2] anchor_dot_contrast[i][j] means z[i] inner product z[j] / temperature
        # anchor_dot_contrast[i][j] = z[i]*z[j]/temperature
        # only for contrast_mode == 'all'
        # for numerical stability
        logits_max, _ = torch.max(
            anchor_dot_contrast, dim=1, keepdim=True
        )  # logits_max : [bsz+bsz, 1] logits_max[i][0] = max(anchor_dot_contrast[i][0..bsz*2-1]) # only for contrast_mode == 'all'
        logits = (
            anchor_dot_contrast - logits_max.detach()
        )  # logits : anchor_dot_contrast[i][j] - max(anchor_dot_contrast[i])

        # tile mask
        mask = mask.repeat(
            anchor_count, contrast_count
        )  # [bsz*2, bsz*2] now mask[i][j] (0, 1) refer to weather the labels of anchor_dot_contrast are equal # only for contrast_mode == 'all'
        # mask-out self-contrast cases
        logits_mask = torch.scatter(  # [bsz*2, bsz*2] eye matrix
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count)
            .view(-1, 1)
            .to(self.device),  # [bsz*2, 1] [i][0] = i
            0,
        )
        mask = (
            mask * logits_mask
        )  # [bsz*2, bsz*2] this operatation makes mask[i][i] = 0

        # compute log_prob
        exp_logits = (
            torch.exp(logits) * logits_mask
        )  # [bsz*2, bsz*2] exp_logits = exp(logits) and this operatation makes exp_logits[i][i] = 0
        # if logits[i] logits[j] label is the same and i!= j, exp_logits[i][j] = exp(logits[i][j]) else 0
        log_prob = logits - torch.log(
            exp_logits.sum(1, keepdim=True)
        )  # [bsz*2, bsz*2] log_prob[i][j] = logits[i][j] - sum(exp_logits(i))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # [bsz*2, bsz*2]

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class MLB_SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, device, temperature=0.07, base_temperature=0.07):
        super(MLB_SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, features, labels, error_rate, weight_mode, formula_mode):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        batch_size = features.shape[0]
        # if labels is not None:
        labels = labels.contiguous().view(-1, 1)  # labels [bsz, 1]

        if labels.shape[0] != batch_size:
            raise ValueError("Num of labels does not match num of features")
        mask = (
            torch.eq(labels, labels.T).float().to(self.device)
        )  # [bsz, bsz] mask[i][j] = 0 or 1, means weather record[i] and record[j] belong to the same label class

        # calculate weight er_weight
        er_w = None
        if weight_mode == "oneplus":
            er_w = 1 + error_rate
        elif weight_mode == "onesub":
            er_w = 1 - error_rate

        er_w = er_w.contiguous().view(-1, 1)  # er_w [bsz, 1]
        if formula_mode == "direct_one":
            er_w_matrix = torch.matmul(er_w, er_w.T).to(
                self.device
            )  # [bsz, bsz] er_w_matrix[i][j] = er_w[i]*er_w[j]

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(
            torch.unbind(features, dim=1), dim=0
        )  # [bsz+bsz, 128] i and i+bsz are two features from the same source record

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )  # [bsz*2, bsz*2] anchor_dot_contrast[i][j] means z[i] inner product z[j] / temperature
        # anchor_dot_contrast[i][j] = z[i]*z[j]/temperature
        # only for contrast_mode == 'all'

        # add new formula
        if formula_mode == "direct_dot":
            # tail er_w_matrix
            er_w_matrix = er_w_matrix.repeat(anchor_count, contrast_count)
            anchor_dot_contrast = anchor_dot_contrast * er_w_matrix

        # for numerical stability
        logits_max, _ = torch.max(
            anchor_dot_contrast, dim=1, keepdim=True
        )  # logits_max : [bsz+bsz, 1] logits_max[i][0] = max(anchor_dot_contrast[i][0..bsz*2-1]) # only for contrast_mode == 'all'
        logits = (
            anchor_dot_contrast - logits_max.detach()
        )  # logits : anchor_dot_contrast[i][j] - max(anchor_dot_contrast[i])

        # tile mask
        mask = mask.repeat(
            anchor_count, contrast_count
        )  # [bsz*2, bsz*2] now mask[i][j] (0, 1) refer to weather the labels of anchor_dot_contrast are equal # only for contrast_mode == 'all'
        # mask-out self-contrast cases
        logits_mask = torch.scatter(  # [bsz*2, bsz*2] eye matrix
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count)
            .view(-1, 1)
            .to(self.device),  # [bsz*2, 1] [i][0] = i
            0,
        )
        mask = (
            mask * logits_mask
        )  # [bsz*2, bsz*2] this operatation makes mask[i][i] = 0

        # compute log_prob
        exp_logits = (
            torch.exp(logits) * logits_mask
        )  # [bsz*2, bsz*2] exp_logits = exp(logits) and this operatation makes exp_logits[i][i] = 0
        # if logits[i] logits[j] label is the same and i!= j, exp_logits[i][j] = exp(logits[i][j]) else 0
        log_prob = logits - torch.log(
            exp_logits.sum(1, keepdim=True)
        )  # [bsz*2, bsz*2] log_prob[i][j] = logits[i][j] - sum(exp_logits(i))

        # compute mean of log-likelihood over positive

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # [bsz*2]

        if formula_mode == "dot_eachi":
            er_w = er_w.view(-1).repeat(contrast_count).to(self.device)
            mean_log_prob_pos = er_w * mean_log_prob_pos

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class WeightSinglelabel(nn.Module):
    def __init__(self):
        super(WeightSinglelabel, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, outputs, targets, weight):
        loss = self.cerition(outputs, targets)
        return (loss * weight).mean()


class MultilabelSoftmaxCrossEntropy(nn.Module):
    """多标签分类的交叉熵
    说明：targets和outputs的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
         不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
         本文。
    来源：修改自下文的Keras版本
    苏剑林. (Apr. 25, 2020). 《将“softmax+交叉熵”推广到多标签分类问题 》[Blog post]. Retrieved from https://kexue.fm/archives/7359
    """

    def __init__(self):
        super(MultilabelSoftmaxCrossEntropy, self).__init__()

    def forward(self, outputs, targets):
        outputs = (1 - 2 * targets) * outputs
        outputs_neg = outputs - targets * 1e12
        outputs_pos = outputs - (1 - targets) * 1e12
        zeros = torch.zeros_like(outputs[:, :1])
        outputs_neg = torch.cat([outputs_neg, zeros], dim=-1)
        outputs_pos = torch.cat([outputs_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(outputs_neg, dim=-1)
        pos_loss = torch.logsumexp(outputs_pos, dim=-1)
        loss = neg_loss + pos_loss
        return loss.mean()


class WeightedBCERDropLoss(nn.Module):
    def __init__(self, weights: torch.Tensor, alpha=5):
        super(WeightedBCERDropLoss, self).__init__()
        self.weightedBCELoss = WeightedMultilabel(weights)
        self.alpha = alpha

    def forward(self, outputs1, targets, outputs2=None, train=False):
        ce_loss = self.weightedBCELoss(outputs1, targets)
        kl_loss = 0

        if train:
            ce_loss = 0.5 * (ce_loss + self.weightedBCELoss(outputs2, targets))
            kl_loss1 = F.kl_div(
                F.logsigmoid(outputs1), torch.sigmoid(outputs2), reduction="batchmean"
            )
            kl_loss2 = F.kl_div(
                F.logsigmoid(outputs2), torch.sigmoid(outputs1), reduction="batchmean"
            )
            kl_loss = 0.5 * (kl_loss1 + kl_loss2)

        # 注意，这里ce_loss和kl_loss的量纲不对等，ce_loss的reduction是mean，kl_loss的reduction是batchmean，也就是说相较而言kl_loss被放大了num_classes倍
        # 这是一个实现错误，但暂时不修正了
        loss = ce_loss + self.alpha * kl_loss
        return loss


class MultiLabelSCERDropLoss(nn.Module):
    def __init__(self, alpha=5):
        super(MultiLabelSCERDropLoss, self).__init__()
        self.MultilabelSoftmaxCrossEntropyLoss = MultilabelSoftmaxCrossEntropy()
        self.alpha = alpha

    def forward(self, outputs1, targets, outputs2=None, train=False):
        ce_loss = self.MultilabelSoftmaxCrossEntropyLoss(outputs1, targets)
        kl_loss = 0

        if train:
            ce_loss = 0.5 * (
                ce_loss + self.MultilabelSoftmaxCrossEntropyLoss(outputs2, targets)
            )
            kl_loss1 = F.kl_div(
                torch.log_softmax(outputs1, dim=-1),
                torch.softmax(outputs2, dim=-1),
                reduction="batchmean",
            )
            kl_loss2 = F.kl_div(
                torch.log_softmax(outputs2, dim=-1),
                torch.softmax(outputs1, dim=-1),
                reduction="batchmean",
            )
            kl_loss = 0.5 * (kl_loss1 + kl_loss2)

        # 此处ce_loss和kl_loss的量纲对等，ce_loss的mean是对batch而言的，kl_loss的reduction是batchmean
        loss = ce_loss + self.alpha * kl_loss
        return loss


class HSICWeightedBCELoss(nn.Module):
    def __init__(self, weights: torch.Tensor, w_hsic):
        super(HSICWeightedBCELoss, self).__init__()
        self.weightedBCELoss = WeightedMultilabel(weights)
        self.w_hsic = w_hsic
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, outputs, targets, global_feature, local_feature):
        ce_loss = self.weightedBCELoss(outputs, targets)

        bsz = outputs.shape[0]
        R = torch.eye(bsz).to(self.device) - (1 / bsz) * torch.ones(bsz, bsz).to(
            self.device
        )
        K1 = torch.mm(global_feature, global_feature.t())
        K2 = torch.mm(local_feature, local_feature.t())
        RK1 = torch.mm(R, K1)
        RK2 = torch.mm(R, K2)
        HSIC_loss = torch.trace(torch.mm(RK1, RK2)) / ((bsz - 1) * (bsz - 1))

        # print('HSIC_loss = {}'.format(HSIC_loss.item()))
        # print('ce_loss = {}---'.format(ce_loss.item()))

        loss = ce_loss + self.w_hsic * HSIC_loss
        return loss


class HSICWeightedBCERDropLoss(nn.Module):
    def __init__(self, weights: torch.Tensor, w_hsic, alpha=0.3):
        super(HSICWeightedBCERDropLoss, self).__init__()
        self.HSICWeightedBCELoss = HSICWeightedBCELoss(weights, w_hsic)
        self.alpha = alpha

    def forward(
        self,
        outputs1,
        targets,
        global_feature1,
        local_feature1,
        outputs2=None,
        global_feature2=None,
        local_feature2=None,
        train=False,
    ):
        hsic_ce_loss = self.HSICWeightedBCELoss(
            outputs1, targets, global_feature1, local_feature1
        )
        kl_loss = 0

        if train:
            ce_loss = 0.5 * (
                hsic_ce_loss
                + self.HSICWeightedBCELoss(
                    outputs2, targets, global_feature2, local_feature2
                )
            )
            kl_loss1 = F.kl_div(
                F.logsigmoid(outputs1), torch.sigmoid(outputs2), reduction="batchmean"
            )
            kl_loss2 = F.kl_div(
                F.logsigmoid(outputs2), torch.sigmoid(outputs1), reduction="batchmean"
            )
            kl_loss = 0.5 * (kl_loss1 + kl_loss2)

        # 注意，这里hsic_ce_loss中的ce_loss部分和kl_loss的量纲不对等，ce_loss的reduction是mean，kl_loss的reduction是batchmean，也就是说相较而言kl_loss被放大了num_classes倍
        # 这是一个实现错误，但暂时不修正了
        loss = hsic_ce_loss + self.alpha * kl_loss
        return loss
