from model import common
import torch.nn as nn
import torch

def make_model(args, parent=False):
    return SEAN(args)

class LFF(nn.Module):
    def __init__(self, args, conv=common.default_conv, n_feats=64):
        super(LFF, self).__init__()

        kernel_size = 3
        n_layes = 5
        scale = args.scale[0]
        act = nn.ReLU(True)

        m_head = [conv(3, n_feats, kernel_size)]

        m_body = [
            conv(
                n_feats, n_feats, kernel_size
            ) for _ in range(n_layes)
        ]

        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(
                n_feats, args.n_colors, kernel_size,
                padding=(kernel_size//2)
            )
        ]

        self.LLF_head = nn.Sequential(*m_head)
        self.LLF_body = nn.Sequential(*m_body)
        self.LLF_tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.LLF_head(x)
        x = self.LLF_body(x)
        x = self.LLF_tail(x)
        return x 


class MSRB(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(MSRB, self).__init__()

        n_feats = 64
        kernel_size_1 = 3
        kernel_size_2 = 5

        self.conv_3_1 = conv(n_feats, n_feats, kernel_size_1)
        self.conv_3_2 = conv(n_feats * 2, n_feats * 2, kernel_size_1)
        self.conv_5_1 = conv(n_feats, n_feats, kernel_size_2)
        self.conv_5_2 = conv(n_feats * 2, n_feats * 2, kernel_size_2)
        self.confusion = nn.Conv2d(n_feats * 4, n_feats, 1, padding=0, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input_1 = x
        output_3_1 = self.relu(self.conv_3_1(input_1))
        output_5_1 = self.relu(self.conv_5_1(input_1))
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.relu(self.conv_5_2(input_2))
        input_3 = torch.cat([output_3_2, output_5_2], 1)
        output = self.confusion(input_3)
        output += x
        return output

class Edge_Net(nn.Module):
    def __init__(self, args, conv=common.default_conv, n_feats=64):
        super(Edge_Net, self).__init__()
        
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        n_blocks = 5
        self.n_blocks = n_blocks
        
        modules_head = [conv(3, n_feats, kernel_size)]

        modules_body = nn.ModuleList()
        for i in range(n_blocks):
            modules_body.append(
                MSRB())

        modules_tail = [
            nn.Conv2d(n_feats * (self.n_blocks + 1), n_feats, 1, padding=0, stride=1),
            conv(n_feats, n_feats, kernel_size),
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.Edge_Net_head = nn.Sequential(*modules_head)
        self.Edge_Net_body = nn.Sequential(*modules_body)
        self.Edge_Net_tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.Edge_Net_head(x)
        res = x

        MSRB_out = []
        for i in range(self.n_blocks):
            x = self.Edge_Net_body[i](x)
            MSRB_out.append(x)
        MSRB_out.append(res)

        res = torch.cat(MSRB_out,1)
        x = self.Edge_Net_tail(res)
        return x 


class Net(nn.Module):
    def __init__(self, args, conv=common.default_conv, n_feats=64):
        super(Net, self).__init__()

        n_resblock = 40
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)
        
        m_head = [conv(n_feats, n_feats, kernel_size)]

        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblock)
        ]

        m_tail = [conv(n_feats, 3, kernel_size)]

        self.Net_head = nn.Sequential(*m_head)
        self.Net_body = nn.Sequential(*m_body)
        self.Net_tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.Net_head(x)
        res = self.Net_body(x)
        res += x
        x = self.Net_tail(res)
        return x


class SEAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SEAN, self).__init__()

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # define head module
        m_LFF = [LFF(args, n_feats=n_feats)]

        # define body module
        m_Edge = [Edge_Net(args, n_feats=n_feats)]

        m_Fushion= [conv(6, n_feats, kernel_size=1)]

        # define tail module
        m_Net = [Net(args, n_feats=n_feats)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.lff = nn.Sequential(*m_LFF)
        self.edge = nn.Sequential(*m_Edge)
        self.fushion = nn.Sequential(*m_Fushion)
        self.net = nn.Sequential(*m_Net)

    def forward(self, x):
        x = self.sub_mean(x)
        low = self.lff(x)
        high = self.edge(x)
        out = torch.cat([low, high], 1)
        out = self.fushion(out)
        out = self.net(out)
        x = self.add_mean(out)
        return high, x
