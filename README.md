Layered Denoiser with Per-Sample Input in PyTorch 
==================================================

Description
-----------
Training code for a layered denoiser that works on per-sample data,
as described in the paper: 
*Neural Denoising with Layer Embeddings*   
https://research.nvidia.com/publication/2020-06_Neural-Denoising-with   

The code base also includes a few variants of per-sample denoisers, which are (simplified)
networks adapted from the paper: 
*Sample-based Monte Carlo Denoising using a Kernel-Splatting Network*   
https://groups.csail.mit.edu/graphics/rendernet/   
 
For business inquiries, please contact researchinquiries@nvidia.com
For press and other inquiries, please contact Hector Marinez at hmarinez@nvidia.com
 
License
-------

Copyright &copy; 2020, NVIDIA Corporation. All rights reserved.

This work is made available under the [NVIDIA Source Code License](https://nvlabs.github.io/layerdenoise/license.html).

Citation
--------

```
@article{Munkberg2020,
  author = {Jacob Munkberg and Jon Hasselgren},
  title = {Neural Denoising with Layer Embeddings},
  journal = {Computer Graphics Forum},
  volume = {39},
  number = {4},
  pages = {1--12},
  year = {2020}
}
```

Requirements
------------
- Python (tested on Anaconda Python 3.7)
- Pytorch 1.5 or newer
- h5py for loading hdf5 files
- Visual Studio 2019 and CUDA 10.2 are required to build the included PyTorch extension [torch_utils](./torch_utils/) on Windows.

Configure PyTorch environment
-----------------------------

Create a PyTorch Anaconda environment
```
conda create -n pytorch python=3.7
activate pytorch
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install -c anaconda h5py
```

In the same environment install `torch_utils`, which is included in the `torch_utils` folder of this repository.
and run `python setup.py install` to build and install the package on your local computer.
Please check the [README.md](./torch_utils/README.md) for detailed installation instructions. 

Docker
------

Navigate to the folder where you've cloned this repository and build the docker image
`docker build --tag ldenoiser:latest -f docker/Dockerfile .`   

Launch a container   
`docker run --gpus device=0 --shm-size 16G --rm -v /raid:/raid -it ldenoiser:latest bash` 

Usage
-----

Open a command line with PyTorch support (typically, on Windows: `activate pytorch` to activate an environment in Anaconda).

Download example training data from https://drive.google.com/drive/folders/1nh7awGg9SyXcdenFUt4aw5vQxyS0dSCw?usp=sharing
and place it in the `./data/` folder. Note that this is a small example set, not the full set used in the paper.

Training: `python train.py --job myjob --config cfg.py --network Layer`

The checkpoints are automatically stored in the `models` folder.

Note that we append a random hash to each jobname. 

Example: For job `test_ccc3e302`
- Debug images at `./jobs/test_ccc3e302/images`
- Checkpoints at `./jobs/test_ccc3e302/model`

Stop training: `ctrl+c`

Inference: `python train.py --job [jobID] --config cfg.py --inference`. The latest checkpoint from the `models` directory is used.
The output images are stored in the `./out` directory. Use the switch `--savedir` to specify another folder.

Output format from inference run:
- `img00000input.png` noisy input image
- `img00000ref.png`   target image
- `img00000out.png`   denoised image 


Run a pretrained model
----------------------

Download the pretrained weights `model_0700.tar` from https://drive.google.com/drive/folders/1nh7awGg9SyXcdenFUt4aw5vQxyS0dSCw?usp=sharing
and the testset `valsetegsr16k_*.h5` from the same location.

Place the weights in 
`[installation dir]/jobs/server/model/model_0700.tar`
and the dataset in the `./data/` folder.

Run inference with 
`python train.py --job server --config cfg.py --inference --savedir results --network Layer`

The folder `results` will be populated with the denoised images (input, denoised and reference images).
The expected error metrics for this trained network on the set of images in `valsetegsr16k_*.h5` are:
```
relMSE,  SMAPE,   PSNR
0.0263, 0.0346, 33.885
```

Dataset generation
------------------

Datasets are generated in hdf5 format. A small example dataset can be downloaded from
https://drive.google.com/drive/folders/1nh7awGg9SyXcdenFUt4aw5vQxyS0dSCw?usp=sharing

The datasets are 5D tensors on the form: `[frames, samples, channels, height, width]`

The current datasets have the header:
```
...\git\layerdl\data>h5dump -H indoorC_input.h5
HDF5 "indoorC_input.h5" {
GROUP "/" {
   DATASET "albedo" {
      DATATYPE  16-bit little-endian floating-point
      DATASPACE  SIMPLE { ( 128, 8, 3, 256, 256 ) / ( 128, 8, 3, 256, 256 ) }
   }
   DATASET "color" {
      DATATYPE  16-bit little-endian floating-point
      DATASPACE  SIMPLE { ( 128, 8, 3, 256, 256 ) / ( 128, 8, 3, 256, 256 ) }
   }
   DATASET "motionvecs" {
      DATATYPE  16-bit little-endian floating-point
      DATASPACE  SIMPLE { ( 128, 8, 3, 256, 256 ) / ( 128, 8, 3, 256, 256 ) }
   }
   DATASET "normals_depth" {
      DATATYPE  16-bit little-endian floating-point
      DATASPACE  SIMPLE { ( 128, 8, 4, 256, 256 ) / ( 128, 8, 4, 256, 256 ) }
   }
   DATASET "specular" {
      DATATYPE  16-bit little-endian floating-point
      DATASPACE  SIMPLE { ( 128, 8, 4, 256, 256 ) / ( 128, 8, 4, 256, 256 ) }
   }
   DATASET "uvt" {
      DATATYPE  16-bit little-endian floating-point
      DATASPACE  SIMPLE { ( 128, 8, 3, 256, 256 ) / ( 128, 8, 3, 256, 256 ) }
   }
}
}
```
