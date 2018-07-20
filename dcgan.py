from torch import nn

class NetG(nn.Module):
    '''
    定义生成器网络
    '''

    def __init__(self,opt):
        super(NetG, self).__init__()
        ngf = opt.ngf #生成器的feature map数

        self.out = nn.Sequential(
            #对输入的nz维度的噪声进行解卷积操作，将其视为nz*1*1的feature map
            nn.ConvTranspose2d(opt.nz,ngf*8,4,1,0,bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            #输出形状为(ngf*8)*4*4

            nn.ConvTranspose2d(ngf*8,ngf*4,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # 输出形状为(ngf*4)*8*8

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 上一步的输出形状： (ngf*2) x 16 x 16

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 上一步的输出形状：(ngf) x 32 x 32

            nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False),
            nn.Tanh()  # 输出范围 -1~1 故而采用Tanh
            # 输出形状：3 x 96 x 96
        )
        self._init_weight()

    def _init_weight(self):
        '''
        对生成器网络可进行初始化，采用(0,0.02)的高斯分布随机数
        :return:
        '''
        for layer in self.out:
            if isinstance(layer,nn.ConvTranspose2d):
                layer.weight.data.normal_(.0,0.02)
            if isinstance(layer,nn.BatchNorm2d):
                layer.weight.data.normal_(1.,0.02)
                layer.bias.data.fill_(0)

    def forward(self, input):
        return self.out(input)

class NetD(nn.Module):
    '''
    定义判别器网络
    '''

    def __init__(self,opt):
        super(NetD, self).__init__()
        ndf = opt.ndf
        self.main = nn.Sequential(
            #输入图片3*96*96
            nn.Conv2d(3,ndf,5,3,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            #输出ndf*32*32

            nn.Conv2d(ndf,ndf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出(ndf*2)*16*16

            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出(ndf*4)*8*8

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出(ndf*8)*4*4

            nn.Conv2d(ndf*8,1,4,1,0,bias=False),
            nn.Sigmoid() #因为需要输出0-1的值，所以采用sigmoid
        )
        self._init_weight()

    def _init_weight(self):
        '''
        对生成器网络可进行初始化，采用(0,0.02)的高斯分布随机数
        :return:
        '''
        for layer in self.main:
            if isinstance(layer,nn.Conv2d):
                layer.weight.data.normal_(.0,0.02)
            if isinstance(layer,nn.BatchNorm2d):
                layer.weight.data.normal_(1.,0.02)
                layer.bias.data.fill_(0)

    def forward(self, input):
        return self.main(input).view(-1)