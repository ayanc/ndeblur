# Neural Deblurring Training Code
Copyright (C) 2016, Ayan Chakrabarti <ayanc@ttic.edu>

This is the code used for training the neural network described in 
the paper. It is a little more involved to use than the test-time
deblurring code. Please read through the following steps, individual
.m files, and the paper for details.


1. Compile `mConv.cu` with MEX (with latest matlab version, should be as
   simple as using `mexcuda`).

2. Generate a training and validation set of kernels using `make_kernels`.
   Typically, you'll want a much larger number of training kernels than 
   val kernels. I'd suggest generating and saving these to a .mat file:
   generating them takes some time, and you also want to use the same 
   val kernels to be able to compare val error across multiple runs.

   Do `help make_kernels` to see the params used in the paper.
   
3. You'll also need a training and validation set of sharp image
   patches. I generated these by extracting patches from the Pascal
   VOC dataset.

   In the current setup, these patches should be of size 105x105. This
   is because most of the code is setup to use 65x65 blurred patches,
   which we'll generate on the fly from these sharp patches and our
   generated kernels (which will be of size 41x41 if you use the
   recommended parameters in make_kernels).

   Store them as variables as say pdata (training) and vdata
   (val). pdata should be 105 x 105 x N (for a very large N), and
   vdata should be 105 x 105 x M, where M is also the number of
   validation kernels. I used M = 2964 in my experiments.
   

4. You'll see that the trained model has two variables `net` and
   `wts`. `net` defines the connectivity structure of the network, and 
   `wts` stores the weights. These shape of these weights also implicitly
   define the number of units in each layer.

   To begin training, you'll need to create an initial .mat file that
   has three variables: net, wts which is randomly initialized, and
   grad which is the same size as wts but initialized to zero.

   You can create these variables using:
   
   `net = nw_struct; wts = nw_wts; grad = zGrad(wts);`


5. The last step before training is to create the `tfm` variable that
   stores the Fourier transform parameters. Use `getTFM` to get an
   initial version, followed by `dec_tfm` to find de-correlating
   transforms for the spectral bands in the input transform.

   Save this `tfm` variable for use throughout training.


6. `tStep` is the function to use to do training. You need to call it
   with wname such that `[wname ".mat"]` is the file where you stored
   the net,wts,grad variables. The last option to tStep is the number
   of "training steps" to do.

   Each step corresponds to doing a 1000 iterations of SGD, followed
   by computing the validation error (the val error is computed over
   10xM patches, generated deterministically from your validation
   patches and kernels).


7. Once you're done training, create a model .mat file by saving the
   following variables:

   `load([wname '.mat'],'net','best_wts');` 
   `wts = best_wts;`
   `save('-mat','model.mat','tfm','net','wts');`
