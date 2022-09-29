#!/usr/bin/env python
# coding: utf-8

# # Preprocessing and registration
# This notebook does basic preprocessing and registration for lightsheet data
# Note v2 uses my saved data that was downsampled from imaris
# version 3a contains downsampled data from imaris with origin and pixel set with metadata. It also has a fix where sampling was done on padded coordinates rather than unpadded

import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from os.path import join as pathjoin
import time
import sys
import argparse
sys.path.append('..')
import imp
import donglab_workflows as dw
imp.reload(dw)

# TODO: This should change,house emlddmm module locally
sys.path.append('/home/dtward/data/csh_data/emlddmm')
import emlddmm
imp.reload(emlddmm)

# Constants

# TODO: change the location of these atlases, need to have a local copy or someway to copy them
atlas_names = [
    '/home/dtward/data/AllenInstitute/allen_vtk/ara_nissl_50.vtk',
    '/home/dtward/data/AllenInstitute/allen_vtk/average_template_50.vtk',
]
# TODO: Find out what this is
seg_name = '/home/dtward/data/AllenInstitute/allen_vtk/annotation_50.vtk'



def register(target_name=None, output_prefix=None, savename=None, device='cuda:0'):

    # Prepare directory
    output_directory = os.path.split(output_prefix)[0]
    if output_directory:
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)

    # load target image
    target_data = np.load(target_name,allow_pickle=True)

    J = target_data['I'][None]
    J = J.astype(np.float32)
    J /= np.mean(np.abs(J))
    xJ = target_data['xI']
    dJ = [x[1] - x[0] for x in xJ]
    J0 = np.copy(J)
    if 'w' in target_data:
        W = target_data['w']
    elif 'W' in target_data:
        W = target_data['W']
    else:
        W = np.ones_like(J[0])
        # or actually
        W = (J[0] > 0).astype(float)
    W = (J[0] > 0).astype(float)

    # 8
    emlddmm.draw(W[None])

    # 9
    fig, ax = emlddmm.draw(J, xJ, vmin=np.min(J[W[None] > 0.9]))
    fig.suptitle('Downsampled lightsheet data')
    figopts = {'dpi': 300, 'quality': 90}
    fig.savefig(output_prefix + 'downsampled.jpg', **figopts)
    fig.canvas.draw()

    # 11
    I = []
    for atlas_name in atlas_names:
        xI, I_, title, names = emlddmm.read_data(atlas_name)
        I_ = I_.astype(np.float32)
        I_ /= np.mean(np.abs(I_))
        I.append(I_)

    I = np.concatenate(I)
    dI = [x[1] - x[0] for x in xI]
    XI = np.meshgrid(*xI, indexing='ij')

    xI0 = [np.copy(x) for x in xI]
    I0 = np.copy(I)

    # 12
    fig, ax = emlddmm.draw(I, xI, vmin=0)
    fig.canvas.draw()

    # 13
    # next is to transform the high resolution data
    xS, S, title, names = emlddmm.read_data(seg_name)
    # we want to visualize the above with S
    labels, inds = np.unique(S, return_inverse=True)

    colors = np.random.rand(len(labels), 3)
    colors[0] = 0.0

    RGB = colors[inds].reshape(S.shape[1], S.shape[2], S.shape[3], 3).transpose(-1, 0, 1, 2)
    # RGB = np.zeros((3,S.shape[1],S.shape[2],S.shape[3]))

    # for i,l in enumerate(labels):
    #    RGB[:,S[0]==l] = colors[i][...,None]
    fig, ax = emlddmm.draw(RGB)
    plt.subplots_adjust(wspace=0, hspace=0, right=1)
    fig.canvas.draw()

    #########################
    # Initial preprocessing
    # Target preprocessing
    #########################

    # 15
    # background
    J = J0 - np.quantile(J0[W[None] > 0.9], 0.1)
    J[J < 0] = 0
    # adjust dynamic range
    J = J ** 0.25

    # adjust mean value
    J /= np.mean(np.abs(J))
    fig, ax = emlddmm.draw(J, xJ, vmin=0)
    fig.canvas.draw()
    fig.suptitle('Preprocessed lightsheet data')
    fig.savefig(output_prefix + 'processed.jpg', **figopts)

    fig, ax = plt.subplots()
    ax.hist(J.ravel(), 100, log=True)
    fig.canvas.draw()

    ########################
    # Atlas preprocessing
    ########################

    # 16
    # pad
    # since I will downsample by 4, I want to pad with 4x4x4
    npad = 4
    I = np.pad(I0, ((0, 0), (npad, npad), (npad, npad), (npad, npad)))
    for i in range(npad):
        xI = [np.concatenate((x[0][None] - d, x, x[-1][None] + d)) for d, x in zip(dI, xI0)]

    # 17
    # adjust nissl image dynamic range
    I[0] = I[0] ** 0.5
    I[0] /= np.mean(np.abs(I[0]))
    I[1] /= np.mean(np.abs(I[1]))

    # 18
    fig, ax = emlddmm.draw(I, xI, vmin=0)
    fig.canvas.draw()

    # 19
    # initial affine
    A0 = np.eye(4)
    # make sure to keep sign of Jacobian
    # A0 = np.diag((1.3,1.3,1.3,1.0))@A0
    # flip x0,x1
    A0 = np.array([[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) @ A0
    # flip x0,x2
    A0 = np.array([[0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) @ A0

    XJ = np.meshgrid(*xJ, indexing='ij')
    A0[:3, -1] = np.mean(XJ, axis=(-1, -2, -3))

    # check it
    tform = emlddmm.Transform(A0, direction='b')
    AI = emlddmm.apply_transform_float(xI, I, tform.apply(XJ))
    fig, ax = emlddmm.draw(np.concatenate((AI[:2], J)), xJ, vmin=0)
    fig.canvas.draw()

    # TODO: stopped at 20, it looks like from on out it's mostly different configurations so need to address this to see what's parsable


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Basic preprocessing and registration for lightsheet data')
    parser.add_argument('target_name',
                        help='Name of target file')
    parser.add_argument('output_directory', '-od',
                        help='Where should outputs go?')
    parser.add_argument('output_file', '-of',
                        help='Name of save file')
    parser.add_argument('use_cuda', '-c',
                        help='Specifies if cuda should be used',
                        action='store_true')
    parser.add_argument('config_file', '-cf',
                        help='Path to configuration file')
    # TODO: this might need to be a constant
    parser.add_argument('cuda_device', '-d',
                        'Specify device for use with cuda')

    args = parser.parse_args()
    target_name = args.target_name
    output_prefix = args.output_directory
    savename = args.output_file
    use_cuda = args.use_cuda
    device = args.cuda_device

    register(target_name=target_name, output_prefix=output_prefix, savename=savename, use_cuda=use_cuda, device=device)