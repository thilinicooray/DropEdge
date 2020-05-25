import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from torch.nn.parameter import Parameter

from numpy.linalg import matrix_power

device = torch.device("cuda:0")


class GCNModel1(nn.Module):
    """
       The model for the single kind of deepgcn blocks.

       The model architecture likes:
       inputlayer(nfeat)--block(nbaselayer, nhid)--...--outputlayer(nclass)--softmax(nclass)
                           |------  nhidlayer  ----|
       The total layer is nhidlayer*nbaselayer + 2.
       All options are configurable.
    """

    def __init__(self,
                 nfeat,
                 nhid,
                 nclass,
                 nhidlayer,
                 dropout,
                 baseblock="mutigcn",
                 inputlayer="gcn",
                 outputlayer="gcn",
                 nbaselayer=0,
                 activation=lambda x: x,
                 withbn=True,
                 withloop=True,
                 aggrmethod="add",
                 mixmode=False):
        """
        Initial function.
        :param nfeat: the input feature dimension.
        :param nhid:  the hidden feature dimension.
        :param nclass: the output feature dimension.
        :param nhidlayer: the number of hidden blocks.
        :param dropout:  the dropout ratio.
        :param baseblock: the baseblock type, can be "mutigcn", "resgcn", "densegcn" and "inceptiongcn".
        :param inputlayer: the input layer type, can be "gcn", "dense", "none".
        :param outputlayer: the input layer type, can be "gcn", "dense".
        :param nbaselayer: the number of layers in one hidden block.
        :param activation: the activation function, default is ReLu.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                           is "add", for others the default is "concat".
        :param mixmode: enable cpu-gpu mix mode. If true, put the inputlayer to cpu.
        """
        super(GCNModel1, self).__init__()
        self.mixmode = mixmode
        self.dropout = dropout

        if baseblock == "resgcn":
            self.BASEBLOCK = ResGCNBlock
        elif baseblock == "densegcn":
            self.BASEBLOCK = DenseGCNBlock
        elif baseblock == "mutigcn":
            self.BASEBLOCK = MultiLayerGCNBlock
        elif baseblock == "inceptiongcn":
            self.BASEBLOCK = InecptionGCNBlock
        else:
            raise NotImplementedError("Current baseblock %s is not supported." % (baseblock))
        if inputlayer == "gcn":
            # input gc
            self.ingc = GraphConvolutionBS(nfeat, nhid, activation, withbn, withloop)
            baseblockinput = nhid
        elif inputlayer == "none":
            self.ingc = lambda x: x
            baseblockinput = nfeat
        else:
            self.ingc = Dense(nfeat, nhid, activation)
            baseblockinput = nhid

        outactivation = lambda x: x
        if outputlayer == "gcn":
            self.outgc = GraphConvolutionBS(baseblockinput, nclass, outactivation, withbn, withloop)
        # elif outputlayer ==  "none": #here can not be none
        #    self.outgc = lambda x: x 
        else:
            self.outgc = Dense(nhid, nclass, activation)

        # hidden layer
        self.midlayer = nn.ModuleList()
        # Dense is not supported now.
        # for i in xrange(nhidlayer):
        for i in range(nhidlayer):
            gcb = self.BASEBLOCK(in_features=baseblockinput,
                                 out_features=nhid,
                                 nbaselayer=nbaselayer,
                                 withbn=withbn,
                                 withloop=withloop,
                                 activation=activation,
                                 dropout=dropout,
                                 dense=False,
                                 aggrmethod=aggrmethod)
            self.midlayer.append(gcb)
            baseblockinput = gcb.get_outdim()
        # output gc
        outactivation = lambda x: x  # we donot need nonlinear activation here.
        self.outgc = GraphConvolutionBS(baseblockinput, nclass, outactivation, withbn, withloop)

        self.reset_parameters()
        if mixmode:
            self.midlayer = self.midlayer.to(device)
            self.outgc = self.outgc.to(device)

    def reset_parameters(self):
        pass

    def forward(self, fea, adj):
        # input
        if self.mixmode:
            x = self.ingc(fea, adj.cpu())
        else:
            x = self.ingc(fea, adj)

        x = F.dropout(x, self.dropout, training=self.training)
        if self.mixmode:
            x = x.to(device)

        # mid block connections
        # for i in xrange(len(self.midlayer)):
        for i in range(len(self.midlayer)):
            midgc = self.midlayer[i]
            x = midgc(x, adj)
        # output, no relu and dropput here.
        x = self.outgc(x, adj)
        x = F.log_softmax(x, dim=1)
        return x

class GCNModel(nn.Module):
    """
       The model for the single kind of deepgcn blocks.

       The model architecture likes:
       inputlayer(nfeat)--block(nbaselayer, nhid)--...--outputlayer(nclass)--softmax(nclass)
                           |------  nhidlayer  ----|
       The total layer is nhidlayer*nbaselayer + 2.
       All options are configurable.
    """

    def __init__(self,
                 nfeat,
                 nhid,
                 nclass,
                 nhidlayer,
                 dropout,
                 baseblock="mutigcn",
                 inputlayer="gcn",
                 outputlayer="gcn",
                 nbaselayer=0,
                 activation=lambda x: x,
                 withbn=True,
                 withloop=True,
                 aggrmethod="add",
                 mixmode=False):
        """
        Initial function.
        :param nfeat: the input feature dimension.
        :param nhid:  the hidden feature dimension.
        :param nclass: the output feature dimension.
        :param nhidlayer: the number of hidden blocks.
        :param dropout:  the dropout ratio.
        :param baseblock: the baseblock type, can be "mutigcn", "resgcn", "densegcn" and "inceptiongcn".
        :param inputlayer: the input layer type, can be "gcn", "dense", "none".
        :param outputlayer: the input layer type, can be "gcn", "dense".
        :param nbaselayer: the number of layers in one hidden block.
        :param activation: the activation function, default is ReLu.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                           is "add", for others the default is "concat".
        :param mixmode: enable cpu-gpu mix mode. If true, put the inputlayer to cpu.
        """
        super(GCNModel, self).__init__()

        self.dropout = dropout
        self.key_proj = nn.Linear(nhid+nfeat,nhid)
        self.query_proj = nn.Linear(nhid,nhid)

        self.ingc = GraphConvolutionBS(nfeat, nhid, activation, withbn, withloop)
        self.midlayer = nn.ModuleList()
        self.keylayer = nn.ModuleList()
        self.querylayer = nn.ModuleList()
        for i in range(nhidlayer):
            gcb = GraphConvolutionBS(nhid, nhid, activation, withbn, withloop)
            self.midlayer.append(gcb)
            key = nn.Linear(nhid+nfeat,nhid)
            self.keylayer.append(key)
            query = nn.Linear(nhid,nhid)
            self.querylayer.append(query)

        outactivation = lambda x: x  # we donot need nonlinear activation here.
        self.outgc = GraphConvolutionBS(nhid, nclass, outactivation, withbn, withloop)
        self.norm = PairNorm()

        self.mu = GraphConvolutionBS(nhid, nhid, activation, withbn, withloop)
        self.logvar = GraphConvolutionBS(nhid, nhid, activation, withbn, withloop)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

        self.node_regen = GraphConvolutionBS(nhid, nfeat, activation, withbn, withloop)

    def reset_parameters(self):
        pass

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask > 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = F.dropout(p_attn, dropout, training=self.training)

        return torch.matmul(p_attn, value)

    def forward(self, fea, adj):
        flag_adj = adj.masked_fill(adj > 0, 1)
        x = self.ingc(fea, adj)

        x = F.dropout(x, self.dropout, training=self.training)
        #adj_con = torch.zeros_like(adj)
        mu = self.mu(x, adj)
        logvar = self.logvar(x, adj)
        z = self.reparameterize(mu, logvar)
        adj1 = self.dc(z)


        #get masked new adj
        zero_vec = -9e15*torch.ones_like(adj1)
        masked_adj = torch.where(adj > 0, adj1, zero_vec)
        adj_con = F.softmax(masked_adj, dim=1)

        key = self.key_proj(torch.cat([x,fea],-1))

        val = self.attention(key, self.query_proj(x), key, adj)


        val_in = val + x


        mask = flag_adj



        # mid block connections
        # for i in xrange(len(self.midlayer)):
        for i in range(len(self.midlayer)):
            mask = mask + torch.mm(mask, flag_adj)
            midgc = self.midlayer[i]
            midkey = self.keylayer[i]
            midquery = self.querylayer[i]

            #x = midgc(torch.cat([x, fea],-1), adj)
            x = midgc(val_in, adj)
            #x = self.norm(x)
            x = F.dropout(x, self.dropout, training=self.training)
            #vae
            mu = self.mu(x, adj)
            logvar = self.logvar(x, adj)
            z = self.reparameterize(mu, logvar)
            adj1 = self.dc(z)


            #get masked new adj
            zero_vec = -9e15*torch.ones_like(adj1)
            masked_adj = torch.where(adj > 0, adj1, zero_vec)
            adj_con =  F.softmax(adj_con +masked_adj, dim=1)

            key = midkey(torch.cat([x,fea],-1))
            query = midquery(x)
            val = val + self.attention(key, query, key, mask)
            val = F.dropout(val, 0.2, training=self.training)
            mfb_sign_sqrt = torch.sqrt(F.relu(val)) - torch.sqrt(F.relu(-(val)))

            val = F.normalize(mfb_sign_sqrt)
            val_in = val + x


        # output, no relu and dropput here.
        x = self.outgc(val_in, adj)
        x = F.log_softmax(x, dim=1)
        return adj_con//(len(self.midlayer) +  1) , mu, logvar, x


class GCNModel_org(nn.Module):
    """
       The model for the single kind of deepgcn blocks.

       The model architecture likes:
       inputlayer(nfeat)--block(nbaselayer, nhid)--...--outputlayer(nclass)--softmax(nclass)
                           |------  nhidlayer  ----|
       The total layer is nhidlayer*nbaselayer + 2.
       All options are configurable.
    """

    def __init__(self,
                 nfeat,
                 nhid,
                 nclass,
                 nhidlayer,
                 dropout,
                 baseblock="mutigcn",
                 inputlayer="gcn",
                 outputlayer="gcn",
                 nbaselayer=0,
                 activation=lambda x: x,
                 withbn=True,
                 withloop=True,
                 aggrmethod="add",
                 mixmode=False):
        """
        Initial function.
        :param nfeat: the input feature dimension.
        :param nhid:  the hidden feature dimension.
        :param nclass: the output feature dimension.
        :param nhidlayer: the number of hidden blocks.
        :param dropout:  the dropout ratio.
        :param baseblock: the baseblock type, can be "mutigcn", "resgcn", "densegcn" and "inceptiongcn".
        :param inputlayer: the input layer type, can be "gcn", "dense", "none".
        :param outputlayer: the input layer type, can be "gcn", "dense".
        :param nbaselayer: the number of layers in one hidden block.
        :param activation: the activation function, default is ReLu.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                           is "add", for others the default is "concat".
        :param mixmode: enable cpu-gpu mix mode. If true, put the inputlayer to cpu.
        """
        super(GCNModel_org, self).__init__()

        self.dropout = dropout

        self.key_proj = nn.Linear(nhid+nfeat,nhid)
        self.query_proj = nn.Linear(nhid,nhid)
        self.value_proj = nn.Linear(nhid+nfeat,nhid)
        self.proj = nn.Linear(nhid+nfeat,nhid)


        self.ingc = GraphConvolutionBS(nfeat, nhid, activation, withbn, withloop)
        self.midlayer = nn.ModuleList()
        self.midlayer_org = nn.ModuleList()
        self.keylayer = nn.ModuleList()
        self.querylayer = nn.ModuleList()
        for i in range(nhidlayer):
            gcb = GraphConvolutionBS(nhid*2, nhid, activation, withbn, withloop)
            self.midlayer.append(gcb)
            ingc = GraphConvolutionBS(nfeat, nhid, activation, withbn, withloop)
            self.midlayer_org.append(ingc)
            key = nn.Linear(nhid+nfeat,nhid)
            self.keylayer.append(key)
            query = nn.Linear(nhid,nhid)
            self.querylayer.append(query)

        outactivation = lambda x: x  # we donot need nonlinear activation here.
        self.outgc = GraphConvolutionBS(nhid*2, nclass, outactivation, withbn, withloop)
        self.norm = PairNorm()

        self.mu = GraphConvolutionBS(nhid, nhid, activation, withbn, withloop)
        self.logvar = GraphConvolutionBS(nhid, nhid, activation, withbn, withloop)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask > 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = F.dropout(p_attn, dropout, training=self.training)

        return torch.matmul(p_attn, value)

    def reset_parameters(self):
        pass

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, fea, adj):

        flag_adj = adj.masked_fill(adj > 0, 1)

        x = self.ingc(fea, adj)

        x = F.dropout(x, self.dropout, training=self.training)
        #adj_con = torch.zeros_like(adj)
        key = self.key_proj(torch.cat([x,fea],-1))

        val = self.attention(key, self.query_proj(x), key, adj)

        #print('val first', val [:5,:10], x[:5,:10])

        '''mfb_sign_sqrt = torch.sqrt(F.relu(val+x)) - torch.sqrt(F.relu(-(val+x)))
        val = F.normalize(mfb_sign_sqrt)'''
        val_in = val + x

        mask = flag_adj
        orgx = x


        # mid block connections
        # for i in xrange(len(self.midlayer)):
        for i in range(len(self.midlayer)):
            current_layer_adj = torch.mm(mask, flag_adj)
            mask = mask + current_layer_adj

            midgc = self.midlayer[i]
            midgc_org = self.midlayer_org[i]
            midkey = self.keylayer[i]
            midquery = self.querylayer[i]
            x = midgc(torch.cat([orgx, val_in],-1), adj)
            #x = midgc(val_in, adj)
            x = F.dropout(x, self.dropout, training=self.training)

            orgx = midgc_org(fea, current_layer_adj)
            orgx = F.dropout(orgx, self.dropout, training=self.training)

            key = midkey(torch.cat([x,fea],-1))
            query = midquery(x)
            val = val + self.attention(key, query, key, mask)
            #val = F.dropout(val, 0.2, training=self.training)
            mfb_sign_sqrt = torch.sqrt(F.relu(val)) - torch.sqrt(F.relu(-(val)))

            val = F.normalize(mfb_sign_sqrt)
            #gate to decide which amount should come from global and neighbours

            val_in = val + x

        #print('val, x', x[:5,:5], val[:5,:5])
        x = self.outgc(torch.cat([orgx, val_in],-1), adj)
        #x = self.outgc(val_in, adj)
        x = F.log_softmax(x, dim=1)
        return x

# Modified GCN
class GCNFlatRes(nn.Module):
    """
    (Legacy)
    """
    def __init__(self, nfeat, nhid, nclass, withbn, nreslayer, dropout, mixmode=False):
        super(GCNFlatRes, self).__init__()

        self.nreslayer = nreslayer
        self.dropout = dropout
        self.ingc = GraphConvolution(nfeat, nhid, F.relu)
        self.reslayer = GCFlatResBlock(nhid, nclass, nhid, nreslayer, dropout)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.attention.size(1))
        # self.attention.data.uniform_(-stdv, stdv)
        # print(self.attention)
        pass

    def forward(self, input, adj):
        x = self.ingc(input, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.reslayer(x, adj)
        # x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

class GATModel_org(nn.Module):
    """
       The model for the single kind of deepgcn blocks.

       The model architecture likes:
       inputlayer(nfeat)--block(nbaselayer, nhid)--...--outputlayer(nclass)--softmax(nclass)
                           |------  nhidlayer  ----|
       The total layer is nhidlayer*nbaselayer + 2.
       All options are configurable.
    """

    def __init__(self,
                 nfeat,
                 nhid,
                 nclass,
                 nhidlayer,
                 dropout,
                 baseblock="mutigcn",
                 inputlayer="gcn",
                 outputlayer="gcn",
                 nbaselayer=0,
                 activation=lambda x: x,
                 withbn=True,
                 withloop=True,
                 aggrmethod="add",
                 mixmode=False):
        """
        Initial function.
        :param nfeat: the input feature dimension.
        :param nhid:  the hidden feature dimension.
        :param nclass: the output feature dimension.
        :param nhidlayer: the number of hidden blocks.
        :param dropout:  the dropout ratio.
        :param baseblock: the baseblock type, can be "mutigcn", "resgcn", "densegcn" and "inceptiongcn".
        :param inputlayer: the input layer type, can be "gcn", "dense", "none".
        :param outputlayer: the input layer type, can be "gcn", "dense".
        :param nbaselayer: the number of layers in one hidden block.
        :param activation: the activation function, default is ReLu.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                           is "add", for others the default is "concat".
        :param mixmode: enable cpu-gpu mix mode. If true, put the inputlayer to cpu.
        """
        super(GATModel_org, self).__init__()

        self.dropout = dropout

        self.key_proj = nn.Linear(nhid+nfeat,nhid)
        self.query_proj = nn.Linear(nhid,nhid)
        self.value_proj = nn.Linear(nhid+nfeat,nhid)
        self.proj = nn.Linear(nhid+nfeat,nhid)


        #self.ingc = GraphAttentionLayer(nfeat, nhid, activation, withbn, withloop)
        self.ingc = nn.Linear(nfeat, nhid)
        self.midlayer = nn.ModuleList()
        for i in range(nhidlayer):
            gcb = GraphAttentionLayer(nhid+nfeat, nhid, activation, withbn, withloop)
            self.midlayer.append(gcb)

        outactivation = lambda x: x  # we donot need nonlinear activation here.
        self.outgc = GraphAttentionLayer(nhid+nfeat, nclass, outactivation, withbn, withloop)
        self.norm = PairNorm()


    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask > 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = F.dropout(p_attn, dropout, training=self.training)

        return torch.matmul(p_attn, value)

    def reset_parameters(self):
        pass

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, fea, adj):


        x = self.ingc(fea, adj)

        x = F.dropout(x, self.dropout, training=self.training)


        # mid block connections
        # for i in xrange(len(self.midlayer)):
        for i in range(len(self.midlayer)):


            midgc = self.midlayer[i]
            #x = midgc(torch.cat([fea, val_in],-1), adj)
            x = midgc(torch.cat([fea, x],-1), adj)
            x = F.dropout(x, self.dropout, training=self.training)

        #print('val, x', x[:5,:5], val[:5,:5])
        #x = self.outgc(torch.cat([fea, val_in],-1), adj)
        x = self.outgc(torch.cat([fea, x],-1), adj)
        x = F.log_softmax(x, dim=1)
        return x


