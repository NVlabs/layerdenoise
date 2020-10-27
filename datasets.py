# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import h5py
import numpy as np

import utils

###############################################################################
# Image sequence data class 
###############################################################################

class SequenceHeader(object):
	def __init__(self, nSequence, resolution, cropSize, sequenceData):
		self.nSequence = nSequence
		self.resolution = resolution
		self.cropSize = cropSize
		
		shape_dict = dict([(key, value.shape) for key, value in sequenceData.frameData[0].__dict__.items()])
		self.frameShape = utils.object_from_dict(shape_dict)

class SequenceData(object):
	def makeCudaTensors(self, x):
		if isinstance(x, dict):
			_dict = {key : self.makeCudaTensors(value) for key, value in x.items()}
			return utils.object_from_dict(_dict)
		elif isinstance(x, (list, tuple)):
			return [self.makeCudaTensors(y) for y in x]
		elif torch.is_tensor(x):
			return x.cuda().float()
		else:
			return x

	def __init__(self, dataset, sequenceData):
		self.frameData = self.makeCudaTensors(sequenceData)

###############################################################################
# Sample dataset
###############################################################################
class SampleDataset(torch.utils.data.Dataset):
	def __init__(self, filename, filename_ref, cropSize, flags, limit=None, randomCrop=True):
		super().__init__()

		self.filename           = filename
		self.filename_ref       = filename_ref
		self.limit              = limit
		self.cropSize           = cropSize
		self.randomCrop         = randomCrop

		# Copy required FLAGS
		self._spp      = flags.spp

		# Parse out header information
		h5py_file     = h5py.File(self.filename, 'r')
		h5py_file_ref = h5py.File(self.filename_ref, 'r')

		self.resolution   = h5py_file['color'].shape[-2:]
		self.nDim         = len(h5py_file['color'].shape)
		self.nFramesPerClip = h5py_file['color'].shape[0]

		assert(self.nDim == 5) # Dataset with 5D tensors [frame, sample, channel, y, x]

		pcrop = self.cropSize
		if pcrop == None:
			pcrop = self.resolution[0]

		print("Dataset %s - Res: %dx%d, Crop: %dx%d" % (self.filename, self.resolution[0], self.resolution[1], pcrop, pcrop))

	def getHeader(self):
		return SequenceHeader(1, self.resolution, self.cropSize, SequenceData(self, self.__getitem__(0)))

	def __len__(self):
		return self.nFramesPerClip if self.limit is None else self.limit

	def __getitem__(self, idx):
		# Create random crop. This data augmentation is added to the reader as it affects disk I/O
		cw, ch = self.resolution[1], self.resolution[0]
		ow, oh = 0, 0
		if self.cropSize != None:
			cw, ch = self.cropSize, self.cropSize
			sw, sh = max(0, self.resolution[1] - cw), max(0, self.resolution[0] - ch)
			if self.randomCrop:
				ow, oh = np.random.randint(0, sw + 1), np.random.randint(0, sh + 1)

		h5py_file     = h5py.File(self.filename, 'r')
		h5py_file_ref = h5py.File(self.filename_ref, 'r')

		assert(self.nDim == 5)
		# Load data
		# Dataset with 5D tensors [frame, sample, channel, y, x] stored in fp16
		color         = h5py_file['color'][idx, 0:self._spp, ..., oh:oh+ch, ow:ow+cw]         # Linear radiance in HDR
		normals_depth = h5py_file['normals_depth'][idx, 0:self._spp, ..., oh:oh+ch, ow:ow+cw] # View space normals in xyz, normalized world space depth in w
		albedo        = h5py_file['albedo'][idx, 0:self._spp, ..., oh:oh+ch, ow:ow+cw]        # Albedo map at first hit
		specular      = h5py_file['specular'][idx, 0:self._spp, ..., oh:oh+ch, ow:ow+cw]      # Specular map at first hit
		uvt           = h5py_file['uvt'][idx, 0:self._spp, ..., oh:oh+ch, ow:ow+cw]           # Lens position (xy) and time (z)
		motionvecs    = h5py_file['motionvecs'][idx, 0:self._spp, ..., oh:oh+ch, ow:ow+cw]    # NDC Motion vectors in xy, signed CoC radius in z
		target        = h5py_file_ref['color'][idx, ..., oh:oh+ch, ow:ow+cw]                  # Reference radiance in linear HDR

		# Create object with frame data
		frame_dict = {
			"color" : np.clip(color, 0.0, 65535.0),
			"normals_depth" : normals_depth,
			"albedo" : albedo,
			"specular" : specular,
			"uvt" : uvt,
			"motionvecs" : motionvecs,
			"target" : np.clip(target, 0.0, 65535.0)
		}
		return [frame_dict]
