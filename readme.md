## Semantics through Time: Semi-supervised Segmentation of Aerial Videos with Iterative Label Propagation

SegProp - Semantic label propagation code.

The paper can be found [here](https://arxiv.org/abs/2010.01910) (accepted at ACCV 2020 as oral presentation).

#### Preamble

This codebase was developed with Python 3.6 & PyTorch 1.3.1.

Requirements:
```
torch
numpy
scipy
opencv_python
h5py
imageio
matplotlib
scikit-image
```

#### Basic usage

Our Ruralscapes dataset can be found on the project [homepage](https://sites.google.com/site/aerialimageunderstanding/semantics-through-time-semi-supervised-segmentation-of-aerial-videos).

###### Optical flow
The algorithm requires precalculated optical flow.
In our paper we used [FlowNet 2.0](https://arxiv.org/abs/1612.01925) for estimation, but any available optical flow should work.

H5 database files containing a `'flow'` field are used, two per each clip:
> Forward in time: [frame0 -> frame1, frame1 -> frame2 etc.]
>
> Bacward in time: [frame1 -> frame0, frame2 -> frame1 etc.]

Optical flow numpy arrays are of the shape:
> [no_frames - 1, width, height, 2]
>, with dim3 = (x_vectors, y_vectors)

###### Running

There's a `ruralscapes_demo.py` file showing basic code usage. It can also be used to reproduce our results with minimal configuration (once optical flows are avaliable).

`segprop.py` contains the main algorithm functions and details parameters.

