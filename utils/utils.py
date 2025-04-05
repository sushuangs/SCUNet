import os
import re
import time
import yaml
import logging
import torch
import kornia
import numpy as np
import torchvision
from models.network_scunet import SCUNet
from torch.optim.lr_scheduler import LambdaLR
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def save_image(
    tensor,
    fp,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    range = None,
    scale_each: bool = False,
    pad_value: int = 0,
    format = None,
) -> None:
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = torchvision.utils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, value_range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)



def _normalize(input_tensor):
    min_val, max_val = torch.min(input_tensor), torch.max(input_tensor)
    return (input_tensor-min_val) / (max_val-min_val)



def save_images(cover_img, H_img, noised_img, epoch, current_step, folder, opt, resize_to=None):

    cover_img = cover_img.cpu()
    H_img = H_img.cpu()
    noised_img = noised_img.cpu()

    if opt['datasets']['range_type'] == 0:
        cover_img = (cover_img + 1) / 2
        H_img = (H_img + 1) / 2
        noised_img = (noised_img + 1) / 2

    if resize_to is not None:
        cover_img = F.interpolate(cover_img, size=resize_to)
        H_img = F.interpolate(H_img, size=resize_to)

    stacked_images = torch.cat([cover_img, H_img, noised_img], dim=0)
    filename = os.path.join(folder, 'epoch-{}-step-{}.png'.format(epoch, current_step))
    if opt['train']['saveStacked']:
        save_image(stacked_images, filename, cover_img.shape[0], normalize=False)
    else:
        save_image(H_img, filename, normalize=False)


def psnr_ssim_acc(opt, image, H_img, L_img):
    if opt['datasets']['range_type'] == 0:
        # psnr
        H_psnr = kornia.metrics.psnr(
            ((image + 1) / 2).clamp(0, 1),
            ((H_img.detach() + 1) / 2).clamp(0, 1),
            1,
        )
        L_psnr = kornia.metrics.psnr(
        ((image + 1) / 2).clamp(0, 1),
        ((L_img.detach() + 1) / 2).clamp(0, 1),
        1,
        )
        # ssim
        ssim = kornia.metrics.ssim(
            ((image + 1) / 2).clamp(0, 1),
            ((H_img.detach() + 1) / 2).clamp(0, 1),
            window_size=11,
        ).mean()
    else:
        # psnr
        H_psnr = kornia.metrics.psnr(
            image.clamp(0, 1),
            H_img.detach().clamp(0, 1),
            1,
        )
        L_psnr = kornia.metrics.psnr(
            image.clamp(0, 1),
            L_img.detach().clamp(0, 1),
            1,
        )
        # ssim
        ssim = kornia.metrics.ssim(
            image.clamp(0, 1),
            H_img.detach().clamp(0, 1),
            window_size=11,
        ).mean()
    return H_psnr, L_psnr, ssim


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def log_info(current_epoch, total_epochs, current_step, Lr_current, H_psnr, L_psnr, ssim, loss):
    logging.info('epoch: {}/{}'.format(current_epoch, total_epochs))
    logging.info('step:{}:'.format(current_step))
    logging.info('lr:{}'.format('{:.7f}'.format(Lr_current)))
    logging.info('loss:{}'.format('{:.7f}'.format(loss)))
    logging.info('ssim:{}'.format('{:.7f}'.format(ssim)))
    logging.info('H_Psnr: {}'.format('{:.7f}'.format(H_psnr)))
    logging.info('L_Psnr: {}'.format('{:.7f}'.format(L_psnr)))
    logging.info('---------------------------------------------------------------------------------------------')


def make_and_get_path(opt, time):
    time_now_NewExperiment = time
    folder_root = opt['path']['results_folder'] + '/' + opt['experiment_name'] + '-' + str(time_now_NewExperiment) + '-' + opt['train/test']
    log_folder = folder_root + '/logs'
    img_folder_tra = folder_root  + '/img/train'
    img_folder_val = folder_root  + '/img/val'
    img_folder_test = folder_root + '/img/test'
    path_checkpoint = folder_root  + '/checkpoint'
    path_in = {'log_folder':log_folder, 'img_folder_tra':img_folder_tra, \
                'img_folder_val':img_folder_val,'img_folder_test':img_folder_test,\
                     'path_checkpoint':path_checkpoint}

    for key, value in path_in.items():
        mkdir(value)

    return path_in


def get_datasetloader(opt):
    if opt['datasets']['range_type'] == 0:
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomCrop((opt['datasets']['H'], opt['datasets']['W']), pad_if_needed=True),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'test': transforms.Compose([
                transforms.CenterCrop((opt['datasets']['H'], opt['datasets']['W'])),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        }
    else:
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomCrop((opt['datasets']['H'], opt['datasets']['W']), pad_if_needed=True),
                transforms.ToTensor(),
                # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'test': transforms.Compose([
                transforms.CenterCrop((opt['datasets']['H'], opt['datasets']['W'])),
                transforms.ToTensor(),
                # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        }
    train_images = datasets.ImageFolder(opt['path']['train_folder'], data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(train_images, batch_size=opt['train']['batch_size'], shuffle=True,
                                                num_workers=opt['train']['num_workers'])

    validation_images = datasets.ImageFolder(opt['path']['test_folder'], data_transforms['test'])
    validation_loader = torch.utils.data.DataLoader(validation_images, batch_size=opt['train']['batch_size'],
                                                    shuffle=False, num_workers=opt['train']['num_workers'])
    
    return train_loader, validation_loader

def get_ramdom_L_images(images, min_std, max_std, mean=0, range_type=0):

    assert min_std >=0
    std = np.random.rand() * (max_std - min_std) + min_std
    if range_type == 0:
        L_images = (images + np.random.normal(mean, std, images.shape) /128)
    else:
        L_images = (images + np.random.normal(mean, std, images.shape) /255)
    
    return L_images.float()


def get_model_resume(opt, device):
    assert opt['resume']['checkpoint_path'] != None
    checkpoint = torch.load(opt['resume']['checkpoint_path'])

    network= SCUNet(**opt['resume']['network']).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=opt['lr']['start_lr'])
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda iteration: 0.5 ** (iteration // 40000)
    )

    start_epoch = 0
    start_current = 0
    val_loss = np.inf
    if opt['resume']['only_network']:
        logging.info('Continue trainning only for network')
        network.load_state_dict(checkpoint['network'])
    else:
        logging.info('Continue trainning for all')
        network.load_state_dict(checkpoint['network'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        start_current = checkpoint['current_step']
        val_loss = checkpoint['val_loss']

    return network, optimizer, scheduler, start_epoch, start_current, val_loss



def get_model(opt, device):
    network = SCUNet(**opt['network']).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=opt['lr']['start_lr'])
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda iteration: 0.5 ** (iteration // 200000)
    )
    start_epoch = 0
    start_current = 0
    val_loss = np.inf
    return network, optimizer, scheduler, start_epoch, start_current, val_loss