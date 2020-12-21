# Implementation of Proxy-based deep Graph Metric Learning (ProxyGML) approach
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import numpy as np

class ProxyGML(nn.Module):
    def __init__(self, opt, dim=512):
        super(ProxyGML, self).__init__()
        self.opt=opt
        dim=self.opt.dim
        self.C = opt.C
        self.N = opt.N
        self.Proxies = Parameter(torch.Tensor(dim, opt.C*opt.N))
        self.instance_label = torch.tensor(np.repeat(np.arange(opt.C), opt.N)).to(self.opt.device)
        self.y_instacne_onehot = self.to_one_hot(self.instance_label, n_dims=self.C).to(self.opt.device)
        self.class_label = torch.tensor(np.repeat(np.arange(opt.C), 1)).to(self.opt.device)
        init.kaiming_uniform_(self.Proxies, a=math.sqrt(5))
        self.index = 0
        print("#########")
        print("GraphLoss trained on dataset {}, |weight_lambda is {}, N is {}, r is {}, |center lr is {}, rate is {}, epoch_to_decay is {}|".format(opt.dataset,opt.weight_lambda,opt.N,opt.r,opt.centerlr,opt.rate,opt.new_epoch_to_decay))
        return

    def to_one_hot(self, y, n_dims=None):
        ''' Take integer tensor y with n dims and convert it to 1-hot representation with n+1 dims. '''
        y_tensor = y.type(torch.LongTensor).view(-1, 1)
        n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
        y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
        y_one_hot = y_one_hot.view(*y.shape, -1)
        return y_one_hot

    def scale_mask_softmax(self,tensor,mask,softmax_dim,scale=1.0):
        #scale = 1.0 if self.opt.dataset != "online_products" else 20.0
        scale_mask_exp_tensor = torch.exp(tensor* scale) * mask.detach()
        scale_mask_softmax_tensor = scale_mask_exp_tensor / (1e-8 + torch.sum(scale_mask_exp_tensor, dim=softmax_dim)).unsqueeze(softmax_dim)
        return scale_mask_softmax_tensor

    def forward(self, input, target):
        self.index += 1
        centers = F.normalize(self.Proxies, p=2, dim=0)
        #constructing directed similarity graph
        similarity= input.matmul(centers)
        #relation-guided sub-graph construction
        positive_mask=torch.eq(target.view(-1,1).to(self.opt.device)-self.instance_label.view(1,-1),0.0).float().to(self.opt.device) #obtain positive mask
        topk = math.ceil(self.opt.r * self.C * self.N)
        _, indices = torch.topk(similarity + 1000 * positive_mask, topk, dim=1) # "1000*" aims to rank faster
        mask = torch.zeros_like(similarity)
        mask = mask.scatter(1, indices, 1)
        prob_a =mask*similarity
        #revere label propagation (including classification process)
        logits=torch.matmul(prob_a , self.y_instacne_onehot)
        y_target_onehot = self.to_one_hot(target, n_dims=self.C).to(self.opt.device)
        logits_mask=1-torch.eq(logits,0.0).float().to(self.opt.device)
        predict=self.scale_mask_softmax(logits, logits_mask,1).to(self.opt.device)
        # classification loss
        lossClassify=torch.mean(torch.sum(-y_target_onehot* torch.log(predict + 1e-20),dim=1))
        #regularization on proxies
        if self.opt.weight_lambda  > 0:
            simCenter = centers.t().matmul(centers)
            centers_logits = torch.matmul(simCenter , self.y_instacne_onehot)
            reg=F.cross_entropy(centers_logits, self.instance_label)
            return lossClassify+self.opt.weight_lambda*reg, lossClassify
        else:
            return lossClassify,torch.tensor(0.0).to(self.opt.device)
