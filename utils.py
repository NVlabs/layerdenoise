# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os

from PIL import Image

import numpy as np
import torch

###############################################################################
# Some utility functions to make pytorch and numpy behave the same
###############################################################################

def _pow(x, y):
	if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
		return torch.pow(x, y)
	else:
		return np.power(x, y)

def _log(x):
	if isinstance(x, torch.Tensor):
		return torch.log(x)
	else:
		return np.log(x)

def _clamp(x, y, z):
	if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor) or isinstance(z, torch.Tensor):
		return torch.clamp(x, y, z)
	else:
		return np.clip(x, y, z)

###############################################################################
# Create a object with members from a dictionary
###############################################################################

class DictObject:
	def __init__(self, _dict):
		self.__dict__.update(**_dict)

def object_from_dict(_dict):
	return DictObject(_dict)

###############################################################################
# SMAPE Loss
###############################################################################

def SMAPE(d, r):
    denom = torch.abs(d) + torch.abs(r) + 0.01 
    return torch.mean(torch.abs(d-r) / denom)

###############################################################################
# relMSE Loss
###############################################################################

def relMSE(d, r):
	diff = d - r
	denom = torch.pow(r, 2.0) + 0.0001
	return torch.mean(torch.pow(diff, 2) / denom)

###############################################################################
# PSNR
###############################################################################

def PSNR(d, ref):
	MSE  = torch.mean(torch.pow(d  - ref, 2.0))
	PSNR  = 10. * torch.log10(1.0/MSE)
	return PSNR

###############################################################################
# Tonemapping
###############################################################################

def tonemap_log(f):
	fc = _clamp(f, 0.00001, 65536.0)
	return _log(fc + 1.0)

# Transfer function taken from https://arxiv.org/pdf/1712.02327.pdf
def tonemap_srgb(f):
	a = 0.055
	if isinstance(f, torch.Tensor):
		return torch.where(f > 0.0031308, _pow(f, 1.0/2.4)*(1 + a) - a, 12.92*f)
	else:
		return np.where(f > 0.0031308, _pow(f, 1.0/2.4)*(1 + a) - a, 12.92*f)


###############################################################################
# Image load/store
###############################################################################

def saveImg(img_file, img):
	# Convert image from chw to hwc
	hwc_img = np.swapaxes(np.swapaxes(img, -3, -2), -2, -1) if len(img.shape) == 3 else img
	if len(hwc_img.shape) == 3 and hwc_img.shape[2] == 1:
		hwc_img = np.squeeze(hwc_img, axis=2)
	if len(hwc_img.shape) == 3 and hwc_img.shape[2] == 2:
		hwc_img = np.concatenate((hwc_img, np.zeros_like(hwc_img[..., 0:1])), axis=2)

	# Save image
	img_array = (np.clip(hwc_img , 0.0, 1.0) * 255.0).astype(np.uint8)
	im = Image.fromarray(img_array)
	im.save(img_file)

###############################################################################
# Create a folder if it doesn't exist
###############################################################################

def mkdir(x):
	if not os.path.exists(x):
		os.mkdir(x)

###############################################################################
# Get time string with remaining time formatted
###############################################################################

def getTimeString(remaining):
	timestring = "hours"
	if (remaining < 1):
		remaining *= 60
		timestring = "minutes"
	if (remaining < 1):
		remaining *= 60
		timestring = "seconds"
	return timestring

