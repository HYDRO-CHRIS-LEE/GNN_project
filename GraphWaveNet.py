import torch
import torch.nn as nn
import torch.nn.functional as F

# Graph WaveNet Model
# Matrix multiplication
class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order # nubmer of hops

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class gwnet(nn.Module):
    def __init__(self, device, num_nodes, in_dim, out_dim, blocks, layers, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None,
                 residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512,
                 kernel_size=2):
        super(gwnet, self).__init__()
        #------------------------------------------------------#
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj  # Adaptive Adjacency Matrix
        self.supports = supports    # Initial Adjacency Matrix Information
        #------------------------------------------------------#
        ## TGCN
        self.filter_convs = nn.ModuleList() # dilated convolution filter
        self.gate_convs = nn.ModuleList()   # dilated convolution gate
        #------------------------------------------------------#
        ## GCN
        self.gconv = nn.ModuleList()
        #------------------------------------------------------#
        ## etc
        self.residual_convs = nn.ModuleList() # residual convolution (1x1)
        self.skip_convs = nn.ModuleList() # skip connection convolution (1x1)
        self.bn = nn.ModuleList() # batch normalization
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1)) # initial convolution layer (1x1)

        #------------------------------------------------------#
        ## Build the model's layers
        receptive_field = 1
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        ## Create the initial node embeddings
        if gcn_bool and addaptadj:
            # Random Initialization
            if aptinit is None: 
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len += 1

            # Pretrained Initialization
            else: 
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, input, supports):
        in_len = input.size(3)

        # (1) zero-padding
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input

        # (2) initial convolution layer (1x1)
        x = self.start_conv(x)
        skip = 0

        # (3) Calculate the appropriate "adaptive adjacency matrix" at every iteration
        ## (3-1) self.support : initial adjacency matrix
        ## (3-2) adp : updated adjacency matrix
        ## (3-3) new_supports : updated adjacency matrix list
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # (4) WaveNet
        
            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

        
        for i in range(self.blocks * self.layers):
            # (4-1) TCN (Dilated Convolution)
            residual = x

            # TCN-a
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            # TCN-b
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            # TCN-a & TCN-b
            x = filter * gate

            # (4-2) Parameterized Skip Connection
            s = x
            s = self.skip_convs[i](s) # 1x1 convolution for dimension matching
            if isinstance(skip, int):
                skip = s
            else:
                skip = skip[:, :, :, -s.size(3):]
                skip = s + skip
            
            # (4-3) GCN (Calculate adaptive adjacency matrix)
            if self.gcn_bool and self.supports is not None:
                # case1: use adaptive adjacency matrix
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                # case2: use initial adjacency matrix
                else:
                    x = self.gconv[i](x, self.supports)
            # case3: not use GCN
            else:
                x = self.residual_convs[i](x)

            # (4-4) Residual Connection
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        # (4-5) Truncate to the original temporal length
        x = x[:, :, :, :in_len]

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        if x.size(-1) == 1:
            x = x.squeeze(-1)

        return x

    def get_adaptive_adj(self):
        """
        Returns the current adaptive adjacency matrix, if addaptadj == True.
        Otherwise, returns None.
        """
        if self.gcn_bool and self.addaptadj:
            # This is exactly how we compute 'adp' in forward(...)
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            return adp
        else:
            return None