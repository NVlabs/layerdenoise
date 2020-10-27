# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import numpy as np

def flip_x(x):
	return x.flip(len(x.shape) - 1)

def flip_y(x):
	return x.flip(len(x.shape) - 2)

def rot90(x):
	return flip_y(torch.transpose(x, -2, -1))

###############################################################################
# Data augmentation utility functions
###############################################################################

def augment_rot90(sequenceHeader, sequenceData):
	# Randomly rotate by 90 degrees
	if np.random.random() < 0.5:
		# Iterate all frames in sequence
		for f in sequenceData.frameData:
			# Iterate all keys (different data streams / features etc.)
			for key in f.__dict__.keys():
				if key == "normals_depth":
					nd = rot90(getattr(f, key))
					nd = torch.stack([-nd.select(-3, 1),
									nd.select(-3, 0)] + 
									[nd.select(-3, i) for i in range(2, nd.shape[-3])], dim=-3)
					setattr(f, key, nd)
				else:
					setattr(f, key, rot90(getattr(f, key)))

def augment_flip_xy(sequenceData):
	# Randomly flip y axis
	if np.random.random() < 0.5:
		# Iterate all frames in sequence
		for f in sequenceData.frameData:
			# Iterate all keys (different data streams / features etc.)
			for key in f.__dict__.keys():
				if key == "normals_depth":
					nd = flip_y(getattr(f, key))
					nd = torch.stack([ nd.select(-3, 0),
									-nd.select(-3, 1)] + 
									[nd.select(-3, i) for i in range(2, nd.shape[-3])], dim=-3)
					setattr(f, key, nd)
				else:
					setattr(f, key, flip_y(getattr(f, key)))

	# Randomly flip x axis
	if np.random.random() < 0.5:
		# Iterate all frames in sequence
		for f in sequenceData.frameData:
			# Iterate all keys (different data streams / features etc.)
			for key in f.__dict__.keys():
				if key == "normals_depth":
					nd = flip_x(getattr(f, key))
					nd = torch.stack([-nd.select(-3, 0),
									nd.select(-3, 1)] + 
									[nd.select(-3, i) for i in range(2, nd.shape[-3])], dim=-3)
					setattr(f, key, nd)
				else:
					setattr(f, key, flip_x(getattr(f, key)))

def augment(sequenceHeader, sequenceData):
	augment_rot90(sequenceHeader, sequenceData)
	augment_flip_xy(sequenceData)


