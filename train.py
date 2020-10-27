# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import re
import sys
import glob
import multiprocessing
import time
import argparse
import uuid
import importlib
import logging
import inspect

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
from torch.optim import Adam, lr_scheduler

from utils import *
from datasets import *
from augmentations import *
from sample_network import *
from layer_network import *

FLAGS = None

###############################################################################
# Configuration
###############################################################################

# Number of training epochs. Each epoch is a complete pass over the training images
NUM_EPOCHS = 1000
VALIDATE_AFTER_EACH_X_EPOCHS = 10

# Save training data to a checkpoint file after each x epochs
SAVE_AFTER_NUM_EPOCHS = 100

# Configuration of learning rate 
LEARNING_RATE = 0.0005

# Gradient clamping
GRADIENT_CLAMP_N = 0.001
GRADIENT_CLAMP   = 0.25

###############################################################################
# Utility functions
###############################################################################

def tonemap(f):
	return tonemap_srgb(tonemap_log(f))

def latest_checkpoint(modeldir):
	ckpts = glob.glob(os.path.join(modeldir, "model_*.tar"))
	nums = [int(re.findall('model_\d+', x)[0][6:]) for x in ckpts]
	return ckpts[nums.index(max(nums))]

def get_learning_rate(optimizer):
	lr = 0.0
	for param_group in optimizer.param_groups:
		lr = param_group['lr']
	return lr

def dumpResult(savedir, idx, output, frameData):
	saveImg(os.path.join(savedir, "img%05d_in.png"  % idx), tonemap(frameData.color[0, 0:int(FLAGS.spp),...].cpu().numpy().mean(axis=0)))
	saveImg(os.path.join(savedir, "img%05d_out.png" % idx), tonemap(output.color[0, ...].cpu().numpy()))
	saveImg(os.path.join(savedir, "img%05d_ref.png" % idx), tonemap(frameData.target[0, ...].cpu().numpy()))

###############################################################################
# Dump error metrics
###############################################################################

def computeErrorMetrics(savedir, output, frameData):
	out = output.color
	ref = frameData.target

	relmse_val = relMSE(out, ref).item()
	smape_val  = SMAPE(out,ref).item() 

	outt = torch.clamp(tonemap(out), 0.0, 1.0) 
	reft = torch.clamp(tonemap(ref), 0.0, 1.0)
	psnr_val = PSNR(outt, reft).item()

	print("relMSE: %1.4f - SMAPE: %1.3f - PSNR: %2.2f" % (relmse_val, smape_val, psnr_val))
	return relmse_val, smape_val, psnr_val

###############################################################################
# Network setup
###############################################################################

def createNetwork(FLAGS, dataset, sequenceHeader):
	if FLAGS.network == "SampleSplat":
		return SampleNet(sequenceHeader, tonemap, splat=True, use_sample_info=True, num_samples = FLAGS.spp, kernel_size=FLAGS.kernel_size).cuda()
	elif FLAGS.network == "PixelGather":
		return SampleNet(sequenceHeader, tonemap, splat=False, use_sample_info=False, num_samples = FLAGS.spp, kernel_size=FLAGS.kernel_size).cuda()
	elif FLAGS.network == "PixelSplat":
		return SampleNet(sequenceHeader, tonemap, splat=True, use_sample_info=False, num_samples = FLAGS.spp, kernel_size=FLAGS.kernel_size).cuda()
	elif FLAGS.network == "SampleGather":
		return SampleNet(sequenceHeader, tonemap, splat=False, use_sample_info=True, num_samples = FLAGS.spp, kernel_size=FLAGS.kernel_size).cuda()
	elif FLAGS.network == "Layer":
		return LayerNet(sequenceHeader, tonemap, splat=True, num_samples = FLAGS.spp, kernel_size=FLAGS.kernel_size).cuda()		
	else:
		print("Unsupported network type", FLAGS.network)
		assert False


###############################################################################
# Inference and training
###############################################################################

def inference(data):
	mkdir(FLAGS.savedir)

	dataset = SampleDataset(data[0], data[1], cropSize=None, flags=FLAGS, randomCrop=False)
	loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=FLAGS.num_workers, drop_last=True)

	# Get animation sequence header information
	sequenceHeader = dataset.getHeader()

	# Setup network model
	model = createNetwork(FLAGS, dataset, sequenceHeader)

	ckpt_name = latest_checkpoint(FLAGS.modeldir)
	print("loading checkpoint %s" % ckpt_name)
	checkpoint = torch.load(ckpt_name)
	model.load_state_dict(checkpoint['model.state_dict'])

	with open(os.path.join(FLAGS.savedir, 'metrics.txt'), 'w') as fout:
		fout.write('ID, relMSE, SMAPE, PSNR \n')

		print("Number of images", len(dataset))

		arelmse = np.empty(len(dataset))
		asmape  = np.empty(len(dataset))
		apsnr   = np.empty(len(dataset))

		cnt = 0
		with torch.no_grad():
			for sequenceData in loader:
				sequenceData = SequenceData(dataset, sequenceData)
				output = model.inference(sequenceData)

				# compute losses
				relmse_val, smape_val, psnr_val = computeErrorMetrics(FLAGS.savedir, output, sequenceData.frameData[-1])

				arelmse[cnt] = relmse_val
				asmape[cnt]  = smape_val
				apsnr[cnt]   = psnr_val

				line = "%d, %1.8f, %1.8f, %2.8f \n" % (cnt, relmse_val, smape_val, psnr_val)
				fout.write(line)

				dumpResult(FLAGS.savedir, cnt, output, sequenceData.frameData[-1])
				cnt += 1

		line = "AVERAGES: %1.4f, %1.4f, %2.3f \n" % (np.mean(arelmse), np.mean(asmape), np.mean(apsnr))
		fout.write(line)

		# compute average values
		print("relMSE, SMAPE, PSNR \n")
		print("%1.4f, %1.4f,  %2.3f \n" % (np.mean(arelmse), np.mean(asmape), np.mean(apsnr)))

def loss_fn(output, target):
	return SMAPE(output, target)

def train(data_train, data_validation):
	# Setup dataloader
	datasets = []
	for d in data_train:
		datasets.append(SampleDataset(d[0], d[1], cropSize=FLAGS.cropsize, flags=FLAGS, limit=FLAGS.limit))
	dataset = torch.utils.data.ConcatDataset(datasets)
	loader = torch.utils.data.DataLoader(dataset, batch_size=FLAGS.batch, shuffle=True, num_workers=FLAGS.num_workers, drop_last=True)

	if FLAGS.validate:
		val_dataset = SampleDataset(data_validation[0], data_validation[1], cropSize=256, flags=FLAGS, limit=None, randomCrop=False)
		val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=FLAGS.batch, shuffle=False, num_workers=FLAGS.num_workers)
		
	# Enable for debugging
	# torch.autograd.set_detect_anomaly(True)

	# Get animation sequence header information
	sequenceHeader = datasets[0].getHeader()

	# Setup network model
	model = createNetwork(FLAGS, dataset, sequenceHeader)

	# Setup optimizer and scheduler 
	optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

	# Setup modeldir, create or resume from checkpoint if needed
	start_epoch = 1
	if FLAGS.resume and os.path.exists(FLAGS.modeldir):
		ckpt_name = latest_checkpoint(FLAGS.modeldir)
		print("-> Resuming from checkpoint: %s" % ckpt_name)
		checkpoint = torch.load(ckpt_name)
		start_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['model.state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer.state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler.state_dict'])
	elif os.path.exists(FLAGS.modeldir):
		print("ERROR: modeldir [%s] already exists, use --resume to continue training" % FLAGS.modeldir)
		sys.exit(1)

	mkdir(FLAGS.modeldir)

	with open(os.path.join(FLAGS.jobdir, 'output.log'), 'w') as fout:
		fout.write('LOG FILE: TRAINING LOSS \n')

	with open(os.path.join(FLAGS.jobdir, 'outputval.log'), 'w') as fout:
		fout.write('LOG FILE: VALIDATION LOSS \n')

	imagedir = os.path.join(FLAGS.jobdir, 'images') 
	mkdir(imagedir)

	val_loss = 1.0
	for epoch in range(start_epoch, NUM_EPOCHS+1):
		start_time = time.time()
		sum = 0.0
		num = 0.0
		# train
		for sequenceData in loader:
			sequenceData = SequenceData(dataset, sequenceData)

			augment(sequenceHeader, sequenceData)

			optimizer.zero_grad()
			output = model.forward(sequenceData, epoch)

			loss = loss_fn(output.color, sequenceData.frameData[0].target)

			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLAMP_N)
			torch.nn.utils.clip_grad_value_(model.parameters(), GRADIENT_CLAMP)
			optimizer.step()

			sum += loss.item()
			num += 1

		train_loss = sum / max(num, 1.0)

		# Compute validation loss
		if FLAGS.validate and epoch % VALIDATE_AFTER_EACH_X_EPOCHS == 0:
			val_sum = 0.0
			val_num = 0.0
			with torch.no_grad():
				for sequenceData in val_loader:
					sequenceData = SequenceData(val_dataset, sequenceData)
					output = model.forward(sequenceData, epoch)
					dumpResult(imagedir, epoch, output, sequenceData.frameData[-1])

					loss = loss_fn(output.color, sequenceData.frameData[0].target)
					val_sum = val_sum + loss.item()
					val_num = val_num + 1
				val_loss = val_sum / max(val_num, 1.0)

			with open(os.path.join(FLAGS.jobdir, 'outputval.log'), 'a') as fout:
				line = "%3d %1.6f \n" % (epoch, val_loss) 
				fout.write(str(line))

		duration = time.time() - start_time
		remaining = (NUM_EPOCHS-epoch)*duration/(60*60)
		timestring = getTimeString(remaining)
		print("Epoch %3d - Learn rate: %1.6f - train loss: %5.5f - validation loss: %5.5f - time %.1f ms (remaining %.1f %s) - time/step: %1.2f ms" 
			% (epoch, get_learning_rate(optimizer), train_loss, val_loss, duration*1000.0, remaining, timestring, duration*1000.0 / len(dataset)))

		with open(os.path.join(FLAGS.jobdir, 'output.log'), 'a') as fout:
			line = "%3d %1.6f \n" % (epoch, train_loss) 
			fout.write(str(line))

		if epoch % SAVE_AFTER_NUM_EPOCHS == 0 or epoch == NUM_EPOCHS:				
			torch.save({
				'epoch': epoch + 1,
				'train_loss': train_loss,
				'val_loss': val_loss,
				'model.state_dict': model.state_dict(),
				'optimizer.state_dict': optimizer.state_dict(),
				'scheduler.state_dict': scheduler.state_dict()
			},
			os.path.join(FLAGS.modeldir, "model_%04d.tar" % epoch))
		scheduler.step()

###############################################################################
# Main function
###############################################################################

if __name__ == '__main__':
	multiprocessing.freeze_support()

	print("Pytorch version:", torch.__version__)

	# Parse command line flags
	parser = argparse.ArgumentParser()
	parser.add_argument('--job', type=str, default='', help='Directory to store the trained model', required=True)
	parser.add_argument('--resume',  action='store_true', default=False, help='Resume training from latest checkpoint')
	parser.add_argument('--batch', type=int, default=4, help="Training batch size")
	parser.add_argument('--cropsize', type=int, default=128, help="Training crop size")
	parser.add_argument('--inference', action='store_true', default=False, help="Run inference instead of training, get checkpoint from job modeldir")
	parser.add_argument('--savedir', type=str, default='./out/', help='Directory to save inference data')
	parser.add_argument('--datadir', type=str, default='./', help='Training data directory')
	parser.add_argument('--network', default="PixelGather", choices=["SampleSplat","PixelGather","SampleGather", "PixelSplat", "Layer"], help="Set network type [SampleSplat,PixelGather,SampleGather,PixelSplat,Layer]")
	parser.add_argument('--limit', type=int, default=None, help="Limit the number of frames")
	parser.add_argument('--scenes', nargs='*', default=[], help="List of scenes")
	parser.add_argument('--valscene', type=str, default=None, help='Validation scene')
	parser.add_argument('--num_workers', type=int, default=8, help="Number of workers")
	parser.add_argument('--spp', type=float, default=8, help='Samples per pixel: 1-8')
	parser.add_argument('--kernel_size', type=int, default=17, help='Kernel size [17x17]')
	parser.add_argument('--config', type=str, default=None, help='Config file')

	FLAGS, unparsed = parser.parse_known_args()	

	# Read config file
	if FLAGS.config is not None:
		cfg = importlib.import_module(FLAGS.config[:-len('.py')] if FLAGS.config.endswith('.py') else FLAGS.config)

		for key in cfg.__dict__:
			if not key.startswith("__") and not inspect.ismodule(cfg.__dict__[key]):
				FLAGS.__dict__[key] = cfg.__dict__[key]

	FLAGS.savedir   = os.path.join(FLAGS.savedir, '')
	FLAGS.validate  = True
	FLAGS.num_workers = min(multiprocessing.cpu_count(), FLAGS.num_workers)

	# Add hash to the job directory to avoid collisions
	if not FLAGS.inference:
		uid = uuid.uuid4()
		FLAGS.job = FLAGS.job + "_" + str(str(uid.hex)[:8])

	print("Commandline arguments")
	print("----")
	for arg in sorted(vars(FLAGS)):
		print("%-12s %s" % (str(arg), str(getattr(FLAGS, arg))))
	print("----")

	script_path    = os.path.split(os.path.realpath(__file__))[0]
	all_jobs_path  = os.path.join(script_path, 'jobs')
	FLAGS.jobdir   = os.path.join(all_jobs_path, FLAGS.job)
	FLAGS.modeldir = os.path.join(FLAGS.jobdir, 'model')

	# Create input data
	data_train = [] # holds tuple of train and ref data file names
	for s in FLAGS.scenes:
		data_in  = os.path.join(FLAGS.datadir, s)
		data_ref = os.path.join(FLAGS.datadir, s[0:s.rfind("_")] + "_ref.h5")
		data_train.append((data_in, data_ref))

	# validation scene file name
	if FLAGS.valscene is None:
		print("--valscene required flag")
		sys.exit(1)
	data_in  = os.path.join(FLAGS.datadir, FLAGS.valscene)
	data_ref = os.path.join(FLAGS.datadir, FLAGS.valscene[0:FLAGS.valscene.rfind("_")] + "_ref.h5")
	data_validation = (data_in, data_ref)

	mkdir(all_jobs_path)
	mkdir(FLAGS.jobdir)

	if FLAGS.inference:
		inference(data_validation)
	else:
		train(data_train, data_validation)
