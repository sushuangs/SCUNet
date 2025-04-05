import os
import sys
import time
import torch
import logging
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from utils import utils
from models.network_scunet import SCUNet
from utils.yml import parse_yml, dict_to_nonedict, dict2str
from utils.early_stopping import EarlyStopping


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

yml_path = './options/opt.yml'
option_yml = parse_yml(yml_path)
opt = dict_to_nonedict(option_yml)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

time_set = time.strftime("%Y-%m-%d-%H:%M", time.localtime())

path_dict = utils.make_and_get_path(opt, time_set)
'''
path_dict{
	folder_root
	log_folder
	img_folder_tra
	img_folder_val
	img_folder_test
	path_checkpoint
}
'''

logging.basicConfig(level=logging.INFO,
				format='%(message)s',
				handlers=[
					logging.FileHandler(os.path.join(path_dict['log_folder'], f'{opt["experiment_name"]}-{time_set}.log')),
					logging.StreamHandler(sys.stdout)
				])

logging.info(dict2str(opt))

if opt['resume']['is_resume']:
	network, optimizer, scheduler, \
		start_epoch, start_current, val_loss = utils.get_model_resume(opt, device)
else:
	network, optimizer, scheduler, \
		start_epoch, start_current, val_loss = utils.get_model(opt, device)

train_loader, validation_loader = utils.get_datasetloader(opt)
early_stopping = EarlyStopping(**opt['early_stopping'], path=path_dict['path_checkpoint'], val_loss=val_loss)
loss_fn = torch.nn.L1Loss()

epoch_number = opt['train']['epoch']
current_step = start_current

logging.info("\n---------------Start training-----------------")
for epoch in range(start_epoch, epoch_number):

	# epoch += train_continue_epoch if train_continue else 0
	start_time = time.time()

    #################
    #     train:    #
    #################
	logging.info('\nStarting epoch {}/{}'.format(epoch, epoch_number))
	for image, _ in train_loader:
		network.train()
		L_image =  utils.get_ramdom_L_images(image, **opt['L_images'])
		with torch.enable_grad():
			optimizer.zero_grad()

			L_image = L_image.to(device)
			image = image.to(device)
			H_image = network(L_image)

			loss = loss_fn(H_image, image)
			loss.backward()

			optimizer.step()
			scheduler.step()

		if current_step % opt['train']['logs_per_step'] == 0:
			H_psnr, L_psnr, ssim = utils.psnr_ssim_acc(opt, image.detach(), H_image.detach(), L_image.detach())
			utils.log_info(epoch, epoch_number, current_step, scheduler.get_last_lr()[0], H_psnr, L_psnr, ssim, loss.item())

		if  current_step % opt["train"]['saveTrainImgs_per_step'] == 0:
			utils.save_images(image.detach(), H_image.detach(), L_image.detach(), epoch, current_step, path_dict['img_folder_tra'], opt, resize_to=None)
		current_step +=1


    #################
    #      val:     #
    #################
	if epoch % opt['val']['val_per_epoch'] == 0:
		val_result = {
			'H_psnr': 0,
			'L_psnr': 0,
			'ssim': 0,
			'loss': 0
		}
		val_step = 0
		logging.info('Running validation for epoch {}/{}'.format(epoch, epoch_number))
		for image, _ in validation_loader:
			network.eval()
			with torch.no_grad():
				L_image =  utils.get_ramdom_L_images(image, **opt['L_images'])
				L_image = L_image.to(device)
				image = image.to(device)

				H_image = network(L_image)

				loss = loss_fn(H_image, image)

				H_psnr, L_psnr, ssim = utils.psnr_ssim_acc(opt, image.detach(), H_image.detach(), L_image.detach())

				if val_step % opt['val']['logs_per_step'] == 0:
					utils.log_info(epoch, epoch_number, current_step, scheduler.get_last_lr()[0], H_psnr, L_psnr, ssim, loss.item())

				if  val_step % opt['val']['saveValImgs_in_step'] == 0:
					utils.save_images(image.detach(), H_image.detach(), L_image.detach(), epoch, current_step, path_dict['img_folder_val'], opt, resize_to=None)
				
				val_result['H_psnr'] += H_psnr
				val_result['L_psnr'] += L_psnr
				val_result['ssim'] += ssim
				val_result['loss'] += loss.item()
			
			val_step +=1
		for key, value in val_result.items():
			val_result[key] = value / val_step
		logging.info(f'\nAverage val loss in epoch {epoch}')
		utils.log_info(epoch, epoch_number, current_step, scheduler.get_last_lr()[0], **val_result)


		early_stopping(val_result['loss'], (network, optimizer, scheduler), epoch, current_step)
		if early_stopping.early_stop:
			print("Early stopping")
			break

