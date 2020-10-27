# Free View Synthesis
Code repository for "Free View Synthesis", ECCV 2020.


## Setup

Install the following Python packages in your Python environment

```
- numpy (1.19.1)
- scikit-image (0.15.0)
- pillow (7.2.0)
- pytorch (1.6.0)
- torchvision (0.7.0)
```

Clone the repository and initialize the submodule

```bash
git clone https://github.com/intel-isl/FreeViewSynthesis.git
cd FreeViewSynthesis
git submodule update --init --recursive
```

Finally, build the Python extension needed for preprocessing

```
cd ext/preprocess
cmake -DCMAKE_BUILD_TYPE=Release .
make 
```


## Run Free View Synthesis

Make sure you adapted the paths in `config.py`!

Then run the evaluation via 

```bash
python exp.py --net rnn_vgg16unet3_gruunet4.64.3 --cmd eval --iter last --eval-dsets tat-subseq --eval-scale 0.5
```

This will run the pretrained network on the four Tanks and Temples sequences.

To train the network from scratch you can run

```bash
python exp.py --net rnn_vgg16unet3_gruunet4.64.3 --cmd retrain
```


## Data

- [Tanks and Temples](https://storage.googleapis.com/isl-datasets/FreeViewSynthesis/ibr3d_tat.tar.gz)
- [New Recordings](https://storage.googleapis.com/isl-datasets/FreeViewSynthesis/ibr3d_own.tar.gz)

We provide the preprocessed Tanks and Temples dataset as we used it for training and evaluation [here](https://storage.googleapis.com/isl-datasets/FreeViewSynthesis/ibr3d_tat.tar.gz). 
Our new recordings can be downloaded in a preprocessed version from [here](https://storage.googleapis.com/isl-datasets/FreeViewSynthesis/ibr3d_own.tar.gz). 

We used [COLMAP](https://colmap.github.io/) for camera registration, multi-view stereo and surface reconstruction on full resolution. 
The packages above contain the already undistorted and registered images.
In addition, we provide the estimated camera calibrations, rendered depthmaps used for warping, and closest source image information. 

In more detail, a single folder `ibr3d_*_scale` (where `scale` is the scale factor with respect to the original images) contains:

- `im_XXXXXXXX.[png|jpg]` the downsampled images used as source images, or as target images.
- `dm_XXXXXXXX.npy` the rendered depthmaps based on the COLMAP surface reconstruction.
- `Ks.npy` contains the `3x3` intrinsic camera matrices, where `Ks[idx]` corresponds to the depth map `dm_{idx:08d}.npy`.
- `Rs.npy` contains the `3x3` rotation matrices from the world coordinate system to camera coordinate system.
- `ts.npy` contains the `3` translation vectors from the world coordinate system to camera coordinate system.
- `count_XXXXXXXX.npy` contains the overlap information from target images to source images. I.e., the number of pixels that can be mapped from the target image to the individual source images. `np.argsort(np.load('count_00000000.npy'))[::-1]` will give you the sorted indices of the most overlapping source images.

Use [`np.load`](https://numpy.org/doc/stable/reference/generated/numpy.load.html) to load the numpy files.

We use the Tanks and Temples dataset for training except the following scenes that are used for evaluation.

- train/Truck
	`[172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196]`
- intermediate/M60
	`[94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129]`
- intermediate/Playground
	`[221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252]`
- intermediate/Train
	`[174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248]`

The numbers below the scene name indicate the indices of the target images that we used for evaluation.


## Citation

Please cite our [paper](http://vladlen.info/papers/FVS.pdf) if you find this work useful.

```bib
@inproceedings{Riegler2020FVS,
  title={Free View Synthesis},
  author={Riegler, Gernot and Koltun, Vladlen},
  booktitle={European Conference on Computer Vision},
  year={2020}
}
```

## Video

[![Free View Synthesis Video](https://img.youtube.com/vi/JDJPn3ZtfZs/0.jpg)](https://www.youtube.com/watch?v=JDJPn3ZtfZs)


