import torch
import torch.nn as nn
from math import sqrt
from model import common
from torch.nn import init

def make_model(args, parent=False):
    return SGNDN3(args)




class Dot(nn.Module):
    def __init__(self):
        super(Dot, self).__init__()

    def forward(self, x, y):
        return x * y

class Concate(nn.Module):
    def __init__(self):
        super(Concate, self).__init__()

    def forward(self, x, y):
        return torch.cat((x,y),1)





class Basic_Block(nn.Module):
    def __init__(self, conv, in_feat, out_feat, kernel_size,bias=True, bn=False, act=nn.ReLU(True)):
            super(Basic_Block, self).__init__()
            m = []
            m.append(conv(in_feat,out_feat,kernel_size,bias=bias))
            if bn: m.append(nn.BatchNorm2d(out_feat))
            if act is not None: m.append(act)
            self.body = nn.Sequential(*m)

        
    def forward(self, x):
        	return self.body(x)



class Upsampling(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False,  bias=True):

        m = []
        m.append(conv(n_feat, scale*scale*3, 3, bias))
        m.append(nn.PixelShuffle(scale))
        super(SimpleUpsampler, self).__init__(*m)

class SGNDN3(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SGNDN3, self).__init__()


        g_blocks = args.g_blocks
        m_blocks = args.m_blocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale
        if args.act =='lrelu':
            self.act = nn.LeakyReLU(0.2, inplace=False)
        else:
            self.act = nn.ReLU(True)


        inputnumber = args.n_colors
		
        bn  = args.bn
        #if args.fuse_operation in ['Dot']:
        #    self.fusion = Dot()
        #elif args.fuse_operation in ['Concate']:
        self.fusion = Concate() 


        self.upsampling = nn.PixelShuffle(2)

        m_lrhead1 = [conv(inputnumber*4, n_feats*2, 3)]
        m_lrbody1 = [Basic_Block(conv, n_feats*2, n_feats*2, kernel_size, bn=bn, act=self.act) for _ in range(g_blocks)]
        m_lrbody11 = [conv(n_feats*2, n_feats*2, 3)]

        m_lrhead2 = [conv(inputnumber*16, n_feats*4, 3)]
        m_lrbody2 = [Basic_Block(conv, n_feats*4, n_feats*4, kernel_size, bn=bn, act=self.act) for _ in range(g_blocks)]
        m_lrbody21 = [conv(n_feats*4, n_feats*4, 3)]


        m_lrhead3 = [conv(inputnumber*64, n_feats*8, 3)]
        m_lrbody3 = [Basic_Block(conv, n_feats*8, n_feats*8, kernel_size, bn=bn, act=self.act) for _ in range(g_blocks)]
        m_lrbody31 = [conv(n_feats*8, n_feats*8, 3)]

  
       
        # define head module
        m_head = [conv(args.n_colors, n_feats, 3)]

        # define body module

        m_lrtail1 = [Basic_Block(conv, n_feats*2, n_feats*2, kernel_size, bn=bn, act=self.act)]
        m_lrtail2 = [Basic_Block(conv, n_feats*4, n_feats*4, kernel_size, bn=bn, act=self.act)]
        m_lrtail3 = [Basic_Block(conv, n_feats*8, n_feats*8, kernel_size, bn=bn, act=self.act)]


        m_lrhead1_0 = [Basic_Block(conv, 3*n_feats, n_feats*2, kernel_size, bn=bn, act=self.act)]
        m_lrhead2_0 = [Basic_Block(conv, 6*n_feats, n_feats*4, kernel_size, bn=bn, act=self.act)]
        

        m_body0 = [Basic_Block(conv, int(1.5*n_feats),n_feats, kernel_size, bn=bn, act=self.act)  ]
        m_body1 = [Basic_Block(conv, n_feats, n_feats,  kernel_size, bn=bn, act=self.act) for _ in range(m_blocks)  ]

        m_tail = [conv(n_feats, args.n_colors, kernel_size)]

        self.lrhead1 = nn.Sequential(*m_lrhead1)
        self.lrbody1 = nn.Sequential(*m_lrbody1)
        self.lrtail1 = nn.Sequential(*m_lrtail1)
        self.lrhead2 = nn.Sequential(*m_lrhead2)
        self.lrbody2 = nn.Sequential(*m_lrbody2)
        self.lrtail2 = nn.Sequential(*m_lrtail2)
        self.lrhead3 = nn.Sequential(*m_lrhead3)
        self.lrbody3 = nn.Sequential(*m_lrbody3)
        self.lrtail3 = nn.Sequential(*m_lrtail3)	
	

        self.lrbody11 = nn.Sequential(*m_lrbody11)
        self.lrbody21 = nn.Sequential(*m_lrbody21)
        self.lrbody31 = nn.Sequential(*m_lrbody31)




        self.lrhead1_0 = nn.Sequential(*m_lrhead1_0)	
        self.lrhead2_0 = nn.Sequential(*m_lrhead2_0)	


        self.head = nn.Sequential(*m_head)
        self.body0 = nn.Sequential(*m_body0)
        self.body1 = nn.Sequential(*m_body1)
        self.tail = nn.Sequential(*m_tail)
        self.upsampling = nn.PixelShuffle(2)
        self.reset_params()

        m_upsampler = [common.SimpleUpsampler(conv, scale, n_feats)]
        self.upsampler = nn.Sequential(*m_upsampler)

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, x):
        g1 = common.DownSamplingShuffle(x,2)
        g2 = common.DownSamplingShuffle(g1,2)
        g3 = common.DownSamplingShuffle(g2,2)




        g3 = self.act(self.lrhead3(g3))
        g3 = self.lrbody31(self.lrbody3(g3))+g3
        g3 = self.lrtail3(g3)
        g3 = self.upsampling(g3)



        g2 = self.act(self.lrhead2(g2))
        g2 = self.lrhead2_0(self.fusion(g2,g3))
        g2 = self.lrbody21(self.lrbody2(g2))+g2
        g2 = self.lrtail2(g2)
        g2 = self.upsampling(g2)

        g1 = self.act(self.lrhead1(g1))
        g1 = self.lrhead1_0(self.fusion(g1,g2))
        g1 = self.lrbody11(self.lrbody1(g1))+g1
        g1 = self.lrtail1(g1)		
        g1 = self.upsampling(g1)
		


        residual = self.head(x)
        residual = self.fusion(g1,self.act(residual))
        residual = self.body0(residual)
        residual = self.body1(residual)


        

        return self.tail(residual)+x







