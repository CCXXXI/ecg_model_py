import torch.nn as nn
from torch.nn import Parameter
import torch
import math
from tqdm import tqdm
import numpy as np
import os
from models.resnet_cbam import resnet34_cbam

__all__ = ["GCNResnet", "MLGCN_resnet34_cbam"]


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            # self.bias = Parameter(torch.Tensor(1, 1, out_features))
            self.bias = Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GCNResnet(nn.Module):
    def __init__(
        self,
        model,
        num_classes,
        adj: np.ndarray,
        word_embedding: np.ndarray,
        device,
        in_channel=300,
        t=0.0,
        gen_A_method="default",
    ):
        super(GCNResnet, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.device = device
        self.num_classes = num_classes
        self.pooling = nn.MaxPool1d(
            14, 14
        )  # (batchsize, 512, 64) -> (batchsize, 512, 4)

        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        # self.gc1 = GraphConvolution(in_channel, 1024, bias=True)
        # self.gc2 = GraphConvolution(1024, 2048, bias=True)
        self.relu = nn.LeakyReLU(0.2)
        # self.relu = nn.LeakyReLU(0.9)
        # self.relu = nn.ReLU()

        # self.bn = nn.BatchNorm1d(num_features=num_classes) # add

        _adj = self.gen_A(num_classes, t, adj, gen_A_method)  # shape=(n, n)
        self.A = Parameter(torch.from_numpy(_adj).float())
        self.inp = torch.tensor(word_embedding, dtype=torch.float32)
        # # image normalization
        # self.image_normalization_mean = [0.485, 0.456, 0.406]
        # self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature):  # , inp):
        feature = self.features(feature)
        feature = self.pooling(feature)  # (batchsize, 512, 64) -> (batchsize, 512, 4)
        feature = feature.view(
            feature.size(0), -1
        )  # (batchsize, 512, 4) -> (batchsize, 2048)
        # feature = self.bn(feature)

        adj = self.gen_adj(self.A).detach()
        inp = self.inp.detach()
        adj = adj.to(self.device)
        inp = inp.to(self.device)
        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        x = x.transpose(0, 1)

        x = torch.matmul(feature, x)

        # x = self.bn(x)

        return x

    def get_config_optim(self, lr, lrp):
        return [
            {"params": self.features.parameters(), "lr": lr * lrp},
            {"params": self.gc1.parameters(), "lr": lr},
            {"params": self.gc2.parameters(), "lr": lr},
        ]

    def gen_adj(self, A):
        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A, D).t(), D)
        # adj = torch.matmul(torch.matmul(D, A), D)
        return adj

    def gen_A(self, num_classes, t, adj, method="t_threshold"):
        _adj = adj.copy()
        if method == "t_threshold":
            _adj[_adj < t] = 0
            _adj[_adj >= t] = 1
            _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
            _adj = _adj + np.identity(num_classes, np.int32)
        elif method == "default":
            for i in range(self.num_classes):
                _adj[i][i] = 1.0
        return _adj


def MLGCN_resnet34_cbam(
    num_classes,
    adj: np.ndarray,
    word_embedding: np.ndarray,
    t: float,
    device,
    gen_A_method="default",
):
    rs34_cbam = resnet34_cbam(num_classes=num_classes)
    model = GCNResnet(
        rs34_cbam,
        adj=adj,
        word_embedding=word_embedding,
        num_classes=30,
        t=t,
        gen_A_method=gen_A_method,
        device=device,
    )
    return model


if __name__ == "__main__":
    model = GraphConvolution(300, 1024)
    print(model)
    train_labelrelationship_matrix = os.path.join(
        "E:/Work/ECG_AI/上海数创医疗/data/20210401 筛选数据90258条",
        "train_v7_labelrelationship_matrix.csv",
    )
    P = np.loadtxt(train_labelrelationship_matrix, delimiter=",")
    for i in range(P.shape[0]):
        P[i][i] = 0.0
    inp = torch.randn(30, 300)
    model = GCNResnet(
        resnet34_cbam(num_classes=30),
        adj=P,
        word_embedding=inp.numpy(),
        num_classes=30,
        t=0.4,
    )
    feature = torch.randn(1, 8, 2048)

    y = model(feature)
    print(y.shape)
