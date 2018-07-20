import os
import torch as t
import torchvision as tv
import tqdm
from model import NetG,NetD
from torchnet.meter import AverageValueMeter
from utils import opt,Visualizer

def train(**kwargs):
    '''
    训练函数
    :param kwargs: fire传进来的训练参数
    :return:
    '''
    opt.parse(kwargs)
    for k_,v_ in kwargs.items():
        setattr(opt,k_,v_)
    if opt.vis:
        vis = Visualizer(opt.env)

    #step1：数据预处理
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    dataset = tv.datasets.ImageFolder(opt.data_path,transform=transforms)
    dataloader = t.utils.data.DataLoader(dataset,
                                        batch_size=opt.batch_size,
                                        shuffle=True,
                                        num_workers=opt.num_workers,
                                        drop_last=True)

    #step2: 定义网络
    netg,netd = NetG(opt),NetD(opt)
    map_location = lambda storage,loc:storage
    if opt.netd_path:
        netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    if opt.netg_path:
        netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))

    #定义优化器和损失函数
    optimizer_g = t.optim.Adam(netg.parameters(), opt.lrG, betas=(0.5, 0.999))
    optimizer_d = t.optim.Adam(netd.parameters(), opt.lrD, betas=(0.5, 0.999))
    criterion = t.nn.BCELoss()

    #真图片label为1，加图片label为0
    #noise为网络的输入
    true_labels = t.ones(opt.batch_size)
    fake_labels = t.zeros(opt.batch_size)
    fix_noises = t.randn(opt.batch_size,opt.nz,1,1)
    noises = t.randn(opt.batch_size,opt.nz,1,1)

    errord_meter = AverageValueMeter()
    errorg_meter = AverageValueMeter()

    if opt.gpu:
        device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
        netd.to(device)
        netg.to(device)
        criterion.to(device)
        true_labels,fake_labels = true_labels.to(device),fake_labels.to(device)
        fix_noises,noises = fix_noises.to(device),noises.to(device)

    epochs = range(140)
    for epoch in iter(epochs):
        for ii,(img,_) in tqdm.tqdm(enumerate(dataloader),total=len(dataloader)):
            if opt.gpu:
                real_img = img.to(device)
            if ii%opt.d_every == 0: #每个batch训练一次鉴别器
                optimizer_d.zero_grad()
                output = netd(real_img) #判断真图片(使其尽可能大)
                error_d_real = criterion(output,true_labels)
                error_d_real.backward()

                ##尽可能把假图片判断为错误
                noises.data.copy_(t.randn(opt.batch_size,opt.nz,1,1))
                fake_img = netg(noises).detach() #根据噪声生成假图
                output = netd(fake_img)
                error_d_fake = criterion(output,fake_labels)
                error_d_fake.backward()

                optimizer_d.step()
                error_d = error_d_fake + error_d_real
                errord_meter.add(error_d.item())

            if ii%opt.g_every == 0: #每5个batch更新一次生成器
                #训练生成器
                optimizer_g.zero_grad()
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = netg(noises)
                output = netd(fake_img)
                error_g = criterion(output,true_labels)
                error_g.backward()
                optimizer_g.step()
                errord_meter.add(error_g.item())

            if opt.vis and ii%opt.plot_time == opt.plot_time - 1:
                ##可视化
                fix_fake_img = netg(fix_noises) #使用噪声生成图片
                vis.images(fix_fake_img.data.cpu().numpy()[:64]*0.5+0.5,win = 'fixfake')
                # vis.images(real_img.data.cpu().numpy()[:64]*0.5+0.5,win = 'real')
                vis.plot('errord', errord_meter.value()[0])
                vis.plot('errorg', errorg_meter.value()[0])

        if epoch%opt.decay_every == opt.decay_every-1:
            #保存模型，图片
            tv.utils.save_image(fix_fake_img.data[:64],'%s/new%s.png'%(opt.save_path,epoch),
                                normalize=True,range=(-1,1))
            t.save(netd.state_dict(), 'checkpoints/new_netd_%s.pth' % epoch)
            t.save(netg.state_dict(), 'checkpoints/new_netg_%s.pth' % epoch)
            errord_meter.reset()
            errorg_meter.reset()
            optimizer_g = t.optim.Adam(netg.parameters(), opt.lrG, betas=(0.5, 0.999))
            optimizer_d = t.optim.Adam(netd.parameters(), opt.lrD, betas=(0.5, 0.999))


def generate(**kwargs):
    '''
    随机生成动漫头像，并根据netd的分数选择较好的结果
    :param kwargs:
    :return:
    '''
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    netg,netd = NetG(opt).eval(),NetD(opt).eval()
    noises = t.randn(opt.gen_search_num, opt.nz, 1, 1).normal_(opt.gen_mean, opt.gen_std)
    map_location = lambda storage, loc: storage
    netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))
    if opt.gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        netd.to(device)
        netg.to(device)
        noises.to(device)

    #生成图片
    fake_img = netg(noises)
    scores = netd(fake_img).data

    #挑选最好的某几张
    indexs = scores.topk(opt.gen_num)[1]
    result = []
    for i in indexs:
        result.append(fake_img.data[ii])
    #保存图片
    tv.utils.save_image(t.stack(result),opt.gen_img,normalize=True,range=(-1,1))

if __name__ == '__main__':
    import fire

    fire.Fire()
    # if t.cuda.is_available():
    #     print(1)
