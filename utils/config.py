
class Config(object):
    '''
    保存网络的配置信息
    '''
    data_path = 'data/'
    num_workers = 4
    image_size = 96 #图片尺寸
    batch_size = 128
    max_epoch = 200
    lrG = 2e-4 #生成器的学习率
    lrD = 2e-4 #鉴别器的学习率
    gpu = True
    nz = 100 #噪声维度
    ngf = 64 #生成器feature map数
    ndf = 64 #判别器feature map数

    save_path = 'imgs/'

    vis = True
    env = 'GAN'
    plot_time = 20 #间隔20，visdom画图一次

    d_every = 1 #每一个batch训练一次判别器
    g_every = 5 #每5个训练一次生成器
    decay_every = 5 #每10个epoch保存一次模型
    netd_path = None
    netg_path = None

    # 只测试不训练
    gen_img = 'result.png'
    # 从512张生成的图片中保存最好的64张
    gen_num = 64
    gen_search_num = 512
    gen_mean = 0  # 噪声的均值
    gen_std = 1  # 噪声的方差

def parse(self,kwargs):
    '''
    根据字典跟新参数
    :param kwargs:
    :return:
    '''
    for k,v in kwargs.items():
        if not hasattr(self,k):
            warnings.warn('warning: opt has no attribute %s' %k)
        setattr(self,k,v)

    print('user config:')
    for k,v in self.__class__.__dict__.items():
        if not k.startswith('_'):
            print(k,getattr(self,k))

Config.parse = parse
opt = Config()
