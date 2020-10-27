# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
import torch
import numpy as np

from torch_utils import WeightedFilter
from torch_utils_ref import WeightedFilterPy

##################################################
# Utility function
##################################################

GRAD_DICT = {}
def save_grad(name):
	def hook(grad):
		global GRAD_DICT
		GRAD_DICT[name] = grad
	return hook

def max_relative_err(x,y):
	return (torch.abs(x - y) / torch.abs(y).max()).max()

##################################################
# Networks
##################################################

class RefNet(torch.nn.Module):
	def __init__(self, input_w, weight_w, b, splat):
		super(RefNet, self).__init__() 
		self.c1 = torch.nn.Conv2d(input_w.shape[1], input_w.shape[0], input_w.shape[2], padding=input_w.shape[2]//2, bias=False)
		self.c2 = torch.nn.Conv2d(weight_w.shape[1], weight_w.shape[0], weight_w.shape[2], padding=weight_w.shape[2]//2, bias=False)
		self.c3 = WeightedFilterPy(input_w.shape[0], weight_w.shape[2], splat=splat)

		self.c1.weight.data = input_w.clone()
		self.c2.weight.data = weight_w.clone()
		self.c3.bias.data = b.clone()

	def forward(self, x, w):
		self.input = self.c1(x)
		self.weight = self.c2(w)
		self.input.register_hook(save_grad("ref_input_grad"))
		self.weight.register_hook(save_grad("ref_weight_grad"))
		return self.c3(self.input, self.weight)

class OurNet(torch.nn.Module):
	def __init__(self, input_w, weight_w, b, splat):
		super(OurNet, self).__init__() 
		self.c1 = torch.nn.Conv2d(input_w.shape[1], input_w.shape[0], input_w.shape[2], padding=input_w.shape[2]//2, bias=False)
		self.c2 = torch.nn.Conv2d(weight_w.shape[1], weight_w.shape[0], weight_w.shape[2], padding=weight_w.shape[2]//2, bias=False)
		self.c3 = WeightedFilter(input_w.shape[0], weight_w.shape[2], splat=splat)

		self.c1.weight.data = input_w.clone()
		self.c2.weight.data = weight_w.clone()
		self.c3.bias.data = b.clone()

	def forward(self, x, w):
		self.input = self.c1(x)
		self.weight = self.c2(w)
		self.input.register_hook(save_grad("our_input_grad"))
		self.weight.register_hook(save_grad("our_weight_grad"))
		return self.c3(self.input, self.weight)

##################################################
# Test
##################################################

for splat in [False, True]:
	print("Splatting: %s" % str(splat))

	num_tests = 10000
	kernel_size = 3
	img_size = 256
	batch_size = 1
	channels = 1

	e_forward = 0.0
	e_input_grad = 0.0
	e_weight_grad = 0.0
	for i in range(num_tests):
		print("%5d / %5d" % (i, num_tests), end="\r", flush=True)

		# Create random image & initialize random weights
		input = torch.randn((batch_size, channels, img_size, img_size)).cuda()
		target = torch.randn((batch_size, channels, img_size, img_size)).cuda()
		W = torch.randn((batch_size, kernel_size*kernel_size, img_size, img_size)).cuda()
	
		input_w = torch.randn((channels, channels, kernel_size, kernel_size)).cuda()
		weight_w = torch.randn((kernel_size*kernel_size, kernel_size*kernel_size, kernel_size, kernel_size)).cuda()
		b = torch.zeros((channels)).cuda()

		# Setup our and refernce networks
		ref_net = RefNet(input_w, weight_w, b, splat).cuda()
		our_net = OurNet(input_w, weight_w, b, splat).cuda()

		# Run forward pass
		ref_res = ref_net(input, W)
		our_res = our_net(input, W)

		# Compute loss and back propagate
		our_loss = torch.nn.MSELoss()(our_res, target)
		ref_loss = torch.nn.MSELoss()(ref_res, target)
		our_loss.backward()
		ref_loss.backward()

		fwd = max_relative_err(our_res, ref_res)
		igrad = max_relative_err(GRAD_DICT["our_input_grad"], GRAD_DICT["ref_input_grad"])
		wgrad = max_relative_err(GRAD_DICT["our_weight_grad"], GRAD_DICT["ref_weight_grad"])

		##################################################################
		# Debug prints

		#if fwd > e_forward:
		#	print("\nNew max forward error:\n", our_res - ref_res)
		#if igrad > e_input_grad:
		#	print("\nNew max input gradient error:\n", GRAD_DICT["our_input_grad"] - GRAD_DICT["ref_input_grad"])
		#if wgrad > e_weight_grad:
		#	print("\nNew max input gradient error:\n", (GRAD_DICT["our_weight_grad"] - GRAD_DICT["ref_weight_grad"]) / GRAD_DICT["ref_weight_grad"].max())

		# Find errors everywhere
		e_forward = max(e_forward, fwd)
		e_input_grad = max(e_input_grad, igrad)
		e_weight_grad = max(e_weight_grad, wgrad)

	print("Forward:     %f" % e_forward)
	print("Input grad:  %f" % e_input_grad)
	print("Weight grad: %f" % e_weight_grad)
