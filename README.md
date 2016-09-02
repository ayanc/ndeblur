# Neural Blind Motion Deblurring
Copyright (C) 2016, Ayan Chakrabarti <ayanc@ttic.edu>

This is a reference implementation of the algorithm described in the
paper:

Ayan Chakrabarti, ["**A Neural Approach to Blind Motion Deblurring**"](http://arxiv.org/abs/1603.04771), 
Proc. of the European Conference on Computer Vision (ECCV) 2016. 

This implementation is being made available for non-commercial research
use only. If you find this code useful in your research, please consider 
citing the above paper.

Please see the [project page](http://www.ttic.edu/chakrabarti/ndeblur/) 
and contact <ayanc@ttic.edu> with any questions.

### Requirements

1. You will need to download the trained neural model, available as
   a MAT file at the [project page](http://www.ttic.edu/chakrabarti/ndeblur/).
   
2. This implementation requires a modern CUDA-capable GPU (it has been
   tested on an NVIDIA Titan X), and a recent version of MATLAB's
   Parallel Computing Toolbox that supports the `GPUArray` class.
   
3. To run the full algorithm with the final non-blind deconvolution
   step, you will also need to download an implementation of the EPLL
   method described in the ICCV 2011 paper "**From Learning Models of
   Natural Image Patches to Whole Image Restoration**", by Daniel
   Zoran and Yair Weiss, and add it to your MATLAB path. This
   implementation can be downloaded from Daniel Zoran's
   [page](http://people.csail.mit.edu/danielzoran/).
   
4. We provide a couple of utility functions in the `sunUtil/`
   directory for running experiments on the Sun *et al.* 2013
   benchmark. After you download the dataset from this
   [page](http://cs.brown.edu/~lbsun/deblur2013/deblur2013iccp.html),
   please edit the `loadSDB.m` file to set paths accordingly.

### Usage

The two top level functions for performing deblurring are `ndeblur1`
and `ndeblur2`. The first function applies the local neural network on
all patches, and forms an initial estimate of the sharp image by
averaging their outputs. The second function then uses this initial
estimate to estimate a global motion blur kernel, and then calls EPLL
to do non-blind deconvolution. The following shows example usage of
these functions (see their documentation, using `help funcname` in
MATLAB, for more information):

```MATLAB
>>> model = load('/path/to/model.mat'); % Load trained neural model
>>> out_navg = ndeblur1(blurry_input,model); % Compute neural average output
>>> out_final = ndeblur2(blurry_input,out_navg); % Compute final output
```

Please see the `training/` sub-directory if you wish to train your
own model.
