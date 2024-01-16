import torch
import torch.nn.functional as F
from torch import nn


class SOFIN(nn.Module):
    def __init__(self, n_rules, input_dim,  output_dim=1, distance_metric="L1", order=1):
        super(SOFIN, self).__init__()
        self.input_dim=input_dim
        self.output_dim = output_dim
        self.distance_metric = distance_metric
        self.order=order
        self._commitment_cost = 0.25
        self.rule_dropout=nn.Dropout(0.0)
        self.n_rules=n_rules
        # at begining we don't have data, just random create one rule
        self.softmax = nn.Softmax(dim=-1)
        self.rules = nn.ModuleList([HTSK_Fuzzy_rule(self.input_dim,output_dim,distance_metric,order) for i in range(n_rules)])
    def get_all_fs(self,x):
        z_outs=[]
        with torch.no_grad():
            for rule in self.rules:
                z = rule.get_FS(x)
                z_outs.append(z)
            z_outs = torch.stack(z_outs,dim=-1)
            all_fs=F.softmax(z_outs,dim=-1)
        return all_fs
    def update_rules(self, x):
        # print("x",x.shape)
        # print("rule number before drop", len(self.rules))
        self.drop_rules(x)
        # print("rule number after drop",len(self.rules))
        self.add_rules(x)
        self.n_rules = len(self.rules)
        # print("rule number after add",len(self.rules))
    def drop_rules(self,x,threshold=0.8):
        while len(self.rules)>=2:
            all_fs=self.get_all_fs(x)
            mask = torch.all(all_fs > threshold, dim=0)
            # Get the indices where the condition is not met
            # print("mask",mask.shape,mask)
            mask=[not elem for elem in mask]
            if any(mask):
                if False in mask:
                    self.rules = nn.ModuleList([module for module, include in zip(self.rules, mask) if include])
                else:
                    break
            else:
                break
    def add_rules(self, x):
        ood_samples=x
        while len(ood_samples)>2 and len(self.rules) <=20:
            threshold=1.1/len(self.rules)
            if len(self.rules)<=1:
                self.rules.append(HTSK_Fuzzy_rule(self.input_dim,self.output_dim,self.distance_metric,self.order))
                ood_samples = ood_samples[1:]
                pass
            all_fs=self.get_all_fs(ood_samples)
            max_probabilities, predicted_classes = torch.max(all_fs, dim=1)
            confused_mask = max_probabilities < threshold
            # print("max_probabilities",max_probabilities)
            ood_indices = torch.nonzero(confused_mask).squeeze()
            if ood_indices.dim() == 0:
                break
            ood_samples=ood_samples[ood_indices]
            if len(ood_samples)>2:
                self.rules.append(HTSK_Fuzzy_rule(self.input_dim,self.output_dim,self.distance_metric,self.order, center=ood_samples[0]).to(x.device))
                ood_samples=ood_samples[1:]
    def forward(self, x):
        if len(x.shape)>2:
            x=x.view(x.shape[0],-1)
        cq_outs = []
        z_outs = []
        for rule in self.rules:
            z, cq = rule(x)
            cq_outs.append(cq)
            z_outs.append(z)
        z_outs = torch.stack(z_outs,dim=-1)
        fs_outs=F.softmax(z_outs,dim=-1)
        fs_outs=self.rule_dropout(fs_outs)
        cq_outs = torch.stack(cq_outs,dim=-2)
        FNN_outs=cq_outs * fs_outs.unsqueeze(-1)
        FNN_outs = FNN_outs.sum(-2)
        return F.sigmoid(FNN_outs)

class HTSK_Fuzzy_rule(nn.Module):
    """ one rule"""
    def __init__(self,input_dim,output_dim=1,distance_metric="L1", order=0, center=None,cq_width=64):
        super().__init__()
        if center is None:
            self.center=nn.Parameter(torch.rand(input_dim))
        else:
            self.center = nn.Parameter(center)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.widths = nn.Parameter(torch.ones(input_dim)*0.25, requires_grad=True)
        self.distance_metric=distance_metric
        self.order=order
        if self.order==0:
            self.consequent = nn.Parameter(torch.rand(output_dim))
        elif self.order==1:
            self.consequent = nn.Linear(self.input_dim, output_dim, bias=True)
            self.consequent = nn.Sequential(nn.Linear(self.input_dim,output_dim,bias=True),nn.Sigmoid())
        elif self.order>=2:
            layers = [nn.Linear(self.input_dim, cq_width)]
            for l in range (self.order-1):
                layers.append(nn.Linear(cq_width, cq_width))
            layers.append(nn.Linear(cq_width, output_dim))
            # layers.append(nn.Sigmoid)
            self.consequent = nn.Sequential(*layers)
    def get_dist(self,x):
        if len(x.shape)==1:
            x=x.unsqueeze(0)
        # first normalize the input
        x=F.sigmoid(x)
        if self.distance_metric=="L1":
            torch_dist = torch.abs(x - self.center)
        elif self.distance_metric=="L2":
            torch_dist = torch.square(x - self.center)
        return torch_dist
    def get_z_values(self, x):
        # print("fuzzy input",x.shape)
        aligned_x = x
        dist=self.get_dist(aligned_x)
        # dist is already sum/prod, not on all dimensions
        aligned_w=self.widths
        prot = torch.div(dist,aligned_w)
        # HTSK dived D
        root=-torch.square(prot)*0.5
        z_values =root
        # print("gmv",x.shape,dist.shape,prot.shape,root.shape,membership_values.shape)
        return z_values
    def get_FS(self, x):
        # mvs=self.get_Membership_values(x)
        #
        # fs=mvs.prod(-1)
        # HTSK dived D
        mvs = self.get_z_values(x)
        fs = mvs.mean(-1)
        # print("gfs", x.shape, mvs.shape)
        return fs
    def forward(self,x):
        fs=self.get_FS(x)
        # conq = self.consequent
        # conq=torch.Tensor(conq).to('cuda')
        # out=fs @ conq
        if self.order==0:
            return fs, self.consequent
        else:
            return fs, self.consequent(x)

