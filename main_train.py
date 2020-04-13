import os
from pathlib import Path

import numpy as np

import click

import torch
from torch import optim, nn
from torchvision.datasets import Cityscapes
from torchvision import transforms

from model import Encoder, Decoder, Discriminator
from vgg import Vgg19
from deepfashion import Deepfashion

#import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

def split_class(x, sem, nc):
    return torch.cat([(sem==i).float()*x for i in range(nc)], dim=1)

def extract_class(x, sem, c):
    return np.where(sem == c, x, 0)


def extract_other_class(x, sem, c):
    return np.where(sem != c, x, 0)


class ToTensor(object):
    def __call__(self, target):
        target = torch.as_tensor(np.asarray(target), dtype=torch.int64)
        target = torch.unsqueeze(target, dim=0)
        return target

def update_lr_default(epoch):
    if epoch < 100:
        return 1.0
    elif 100 <= epoch < 200:
        return (200 - epoch)/100
    return 0

def update_lr_deepfashion(epoch):
    if epoch < 60:
        return 1.0
    elif 60 <= epoch < 100:
        return (100 - epoch)/40
    return 0

@click.command()
@click.option('--save_path', default='checkpoint/test', type=Path)
@click.option('--checkpoint', default='checkpoint/test/latest.pth', type=Path)
@click.option('--data_root', default='~/data/cityscapes/', type=Path)
@click.option('--batch_size', default=8, type=int)
@click.option('--dataset', default='cityscapes', type=click.Choice(['cityscapes', 'deepfashion']))
def train(save_path, checkpoint, data_root, batch_size, dataset):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([transforms.Resize((128, 128)),
                                    transforms.ToTensor()])
    target_transform = transforms.Compose([transforms.Resize((128, 128)),
                                           ToTensor()])
    if dataset == 'cityscapes':
        train_data = Cityscapes(str(data_root), split='train', mode='fine', target_type='semantic', transform=transform, target_transform=transform)
        eG = 35
        dG = [35, 35, 20, 14, 10, 4, 1]
        eC = 8
        dC = 280
        n_classes = len(Cityscapes.classes)
        update_lr = update_lr_default
        epoch = 200
    else:
        train_data = Deepfashion(str(data_root), split='train', transform=transform, target_transform=transform)
        n_classes = len(Deepfashion.eclasses)
        eG = 8
        eC = 64
        dG = [8, 8, 4, 4, 2, 2, 1]
        dC = 160
        update_lr = update_lr_deepfashion
        epoch = 100
    data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=1)

    os.makedirs(save_path, exist_ok=True)
    

    n_channels = 3
    encoder = Encoder(n_classes*n_channels, C=eC, G=eG)
    decoder = Decoder(8*eG, n_channels, n_classes, C=dC, Gs=dG)
    discriminator = Discriminator(n_classes + n_channels)
    vgg = Vgg19().eval()

    encoder = torch.nn.DataParallel(encoder)
    decoder = torch.nn.DataParallel(decoder)
    discriminator = torch.nn.DataParallel(discriminator)
    vgg = torch.nn.DataParallel(vgg)

    gen_opt = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.0001, betas=(0, 0.9))
    dis_opt = optim.Adam(discriminator.parameters(), lr=0.0004, betas=(0, 0.9))
    gen_scheduler = optim.lr_scheduler.LambdaLR(gen_opt, update_lr)
    dis_scheduler = optim.lr_scheduler.LambdaLR(gen_opt, update_lr)
    params = ['encoder', 'decoder', 'discriminator', 'gen_opt', 'dis_opt', 'gen_scheduler', 'dis_scheduler']

    if os.path.exists(checkpoint):
        cp = torch.load(checkpoint)
        print(f'Load checkpoint: {checkpoint}')
        for param in params:
            eval(param).load_state_dict(cp[param])
        # encoder.load_state_dict(cp['encoder'])
        # decoder.load_state_dict(cp['decoder'])
        # discriminator.load_state_dict(cp['discriminator'])
        # gen_opt.load_state_dict(cp['gen_opt'])
        # dis_opt.load_state_dict(cp['dis_opt'])
        # gen_scheduler.load_state_dict(cp['gen_scheduler'])
        # dis_scheduler.load_state_dict(cp['dis_scheduler'])

    def to_device_optimizer(opt):
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    to_device_optimizer(gen_opt)
    to_device_optimizer(dis_opt)

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    discriminator = discriminator.to(device)
    vgg = vgg.to(device)
    print(len(data_loader))    
    for epoch in range(epoch):
        e_g_loss = []
        e_d_loss = []
        for i, batch in tqdm(enumerate(data_loader)):
            x, sem = batch
            x = x.to(device)
            sem = sem.to(device)
            sem = sem * 255.0
            sem = sem.long()
            s = split_class(x, sem, n_classes)
            sem_target = sem.clone()
            del sem
            sem = torch.zeros(x.size()[0], n_classes, sem_target.size()[2], sem_target.size()[3], device=x.device)
            sem.scatter_(1, sem_target, 1)
            s = s.detach()
            s = s.to(device)
            mu, sigma = encoder(s)
            z = mu + torch.exp(0.5 * sigma) * torch.rand(mu.size(), device=mu.device)
            gen = decoder(z, sem)
            d_fake = discriminator(gen, sem)
            d_real = discriminator(x, sem)
            l1loss = nn.L1Loss()
            gen_opt.zero_grad()
            loss_gen = 0.5 * d_fake[0][-1].mean() + 0.5 * d_fake[1][-1].mean()
            loss_fm = sum([sum([l1loss(f, g) for f, g in zip(fs, rs)]) for fs, rs in zip(d_fake, d_real)]).mean()

            f_fake = vgg(gen)
            f_real = vgg(x)
            # loss_p = 1.0 / 32 * l1loss(f_fake.relu1_2, f_real.relu1_2) + \
            #     1.0 / 16 * l1loss(f_fake.relu2_2, f_real.relu2_2) + \
            #     1.0 / 8 * l1loss(f_fake.relu3_3, f_real.relu3_3) + \
            #     1.0 / 4 * l1loss(f_fake.relu4_3, f_real.relu4_3) + \
            #     l1loss(f_fake.relu5_3, f_real.relu5_3)
            loss_p = 1.0 / 32 * l1loss(f_fake[0], f_real[0]) + \
                1.0 / 16 * l1loss(f_fake[1], f_real[1]) + \
                1.0 / 8 * l1loss(f_fake[2], f_real[2]) + \
                1.0 / 4 * l1loss(f_fake[3], f_real[3]) + \
                l1loss(f_fake[4], f_real[4])            
            loss_kl = -0.5 * torch.sum(1 + sigma - mu*mu - torch.exp(sigma))
            loss = loss_gen + 10.0 * loss_fm + 10.0 * loss_p + 0.05 * loss_kl
            loss.backward(retain_graph=True)
            gen_opt.step()

            dis_opt.zero_grad()
            loss_dis = torch.mean(-torch.mean(torch.min(d_real[0][-1] - 1, torch.zeros_like(d_real[0][-1]))) +
                                  -torch.mean(torch.min(-d_fake[0][-1] - 1, torch.zeros_like(d_fake[0][-1])))) + \
                                  torch.mean(-torch.mean(torch.min(d_real[1][-1] - 1, torch.zeros_like(d_real[1][-1]))) +
                                  -torch.mean(torch.min(-d_fake[1][-1] - 1, torch.zeros_like(d_fake[1][-1]))))
            loss_dis.backward()
            dis_opt.step()

            e_g_loss.append(loss.item())
            e_d_loss.append(loss_dis.item())
            #plt.imshow((gen.detach().cpu().numpy()[0]).transpose(1, 2, 0))
            #plt.pause(.01)
            #print(i, 'g_loss', e_g_loss[-1], 'd_loss', e_d_loss[-1])
            os.makedirs(save_path / str(epoch), exist_ok=True)

            Image.fromarray((gen.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255.0).astype(np.uint8)).save(save_path / str(epoch) / f'{i}.png')
        print('g_loss', np.mean(e_g_loss), 'd_loss', np.mean(e_d_loss))

        # save
        cp = {}
        for param in params:
            cp[param] = eval(param).state_dict()
        torch.save(cp, save_path / 'latest.pth')#{param:eval(param).state_dict() for param in params})
            
if __name__ == '__main__':
    train()
