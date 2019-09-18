import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def pairwise_loss(outputs1, outputs2, label1, label2, sigmoid_param=1.0, l_threshold=15.0, class_num=1.0):
    # 传入的outputs1是 [36×48] outputs2是 [36×48]
    
    # s_ij = 1 (similar) or s_ij = 0 (dissimilar),      矩阵a和b矩阵相乘 注意label有一个转置
    similarity = Variable(torch.mm(label1.data.float(), label2.data.float().t()) > 0).float()
    dot_product = sigmoid_param * torch.mm(outputs1, outputs2.t())  #dot_product [36×36] α <h_i,h_j>
    exp_product = torch.exp(dot_product)    # 返回一个新张量，包含输入input张量每个元素的指数   exp (α <h_i,h_j>))
    mask_dot = dot_product.data > l_threshold
    mask_exp = dot_product.data <= l_threshold
    mask_positive = similarity.data > 0
    mask_negative = similarity.data <= 0
    mask_dp = mask_dot & mask_positive  # w_ij
    mask_dn = mask_dot & mask_negative
    mask_ep = mask_exp & mask_positive
    mask_en = mask_exp & mask_negative

    dot_loss = dot_product * (1-similarity)
    exp_loss = (torch.log(1+exp_product) - similarity * dot_product)

    # w_ij*( log(1 + exp (α <h_i,h_j>)) − α s_ij <h_i,h_j>),
    # w_ij*( log(1 + exp (dot_product)) − s_ij dot_product),
    # w_ij*( log(1 + exp_product − s_ij dot_product),

    loss = (torch.sum(torch.masked_select(exp_loss, Variable(mask_ep))) + torch.sum(torch.masked_select(dot_loss, Variable(mask_dp)))) * class_num + torch.sum(torch.masked_select(exp_loss, Variable(mask_en))) + torch.sum(torch.masked_select(dot_loss, Variable(mask_dn)))
    # https://blog.csdn.net/q511951451/article/details/81611903

    return loss / (torch.sum(mask_positive.float()) * class_num + torch.sum(mask_negative.float()))
