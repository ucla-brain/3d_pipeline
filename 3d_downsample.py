#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import time
import src.donglab_workflows as dw
from src.tiffstack import TifStack
import argparse

# this notebook will downsample lightsheet data
# from here
#  /panfs/dong/3D_stitched_LS/20220725_SW220510_02_LS_6x_1000z
#  note we will  now load voxel size from the data itself
# and we load the extent as well
#

# blocksize and chunksize for looking for areas with no data and loading quickly
BLOCKSIZE = 64 #
CHUNKSIZE = 32 #
# TODO: find way to specifiy this 
TEMP_OUTDIR = '/panfs/dong/Luis/testing/3d_pipe_test/temp_folder'

def downsample(input_path, image_type, output_filename, dI, res, channel):
    dataset_string = f'DataSet/ResolutionLevel 0/TimePoint 0/Channel {channel}/Data'  # not used for Tifs
    power = np.ones(1, dtype=np.float32) * 0.125

    # 5
    if dI is None and image_type == 'ims':
        f = h5py.File(input_path, 'r')
        dI = dw.imaris_get_pixel_size(f)
        xI = dw.imaris_get_x(f)
        f.close()

    if output_filename is None:
        output_filename = os.path.splitext(os.path.split(input_path)[-1])[0] + '_ch_' + str(channel) + '_pow_' + str(
            power) + '_down.npz'
    # output_filename = 'SYTO16_488_086780_109130_down.npz'

    print(f'Input path is {input_path}')
    print(f'Output filename is {output_filename}')
    print(f'Resolution is {dI}')
    print(f'Desired resolution is {res}')
    print(f'Dataset string is {dataset_string}')
    print(f'tmp output dir is {TEMP_OUTDIR}')

    # temporary output dir
    os.makedirs(TEMP_OUTDIR, exist_ok=True)

    # 6
    # I want 50 micron
    down = np.floor(res / dI).astype(int)
    print(f'Downsampling factors are {down}')
    print(f'Downsampled res {dI * down}')

    # 8
    ##################
    # load the data
    ##################

    if image_type == 'tif':
        # TODO: check that this actually should be input_path
        # data = TifStack(input_directory)
        data = TifStack(input_path)

    elif image_type == 'ims':
        data_ = h5py.File(input_path, mode='r')
        data = data_[dataset_string]

    # 10
    print(f'Dataset shape {data.shape}')

    # 11

    nI = np.array(data.shape)
    # xI = [np.arange(n)*d - (n-1)/2.0*d for n,d in zip(nI,dI)] # already computed above
    # NOTE: the imaging data is smaller than the saved data because the saved data is a multiple of 64
    nIreal = np.array([len(x) for x in xI])

    # 12
    xId = [dw.downsample(x, [d]) for x, d in zip(xI, down)]
    dId = [x[1] - x[0] for x in xId]

    # 13

    # okay now I have to iterate over the dataset
    # note this is currently not doing wieghts
    # we need to save intermediate outputs (each slice) in case of errors
    fig, ax = plt.subplots(2, 2)
    ax = ax.ravel()
    working = []
    working2 = []
    workingw = []
    output = []
    output2 = []
    outputw = []
    start = time.time()
    for i in range(data.shape[0]):
        starti = time.time()
        outname = os.path.join(TEMP_OUTDIR, f'{i:06d}_s.npy')

        if os.path.exists(outname):
            # what happens if it fails in the middle of a chunk?
            sd = np.load(outname)
            s2d = np.load(outname.replace('_s', '_s2'))
            wd = np.load(outname.replace('_s', '_w'))
        else:
            # load a whole chunk
            if not i % CHUNKSIZE:
                data_chunk = data[i:i + CHUNKSIZE]
            # use this for weights
            # s_all = data[i,:,:]
            # it's possible that this will fail if I haven't defined data_chunk yet
            try:
                s_all = data_chunk[i % CHUNKSIZE, :, :]
            except:
                # we need to load, not starting at i
                # but at the beginning of the chunk
                data_chunk = data[i // CHUNKSIZE:i // CHUNKSIZE + CHUNKSIZE]
                s_all = data_chunk[i % CHUNKSIZE, :, :]
            s = s_all[:nIreal[1] + 1,
                :nIreal[2] + 1] ** power  # test reduce dynamic range before downsampling with this power
            s2 = s ** 2
            # w = (s>0).astype(float)
            # this is not a good way to get weights,
            # we need to look for a 64x64 block of all zeros

            s_all_block = s_all.reshape(s_all.shape[0] // BLOCKSIZE, BLOCKSIZE, s_all.shape[1] // BLOCKSIZE, BLOCKSIZE)
            tmp = np.logical_not(np.all(s_all_block == 0, axis=(1, 3))).astype(np.uint8)
            s_all_w = np.ones_like(s_all_block)
            s_all_w *= tmp[:, None, :, None]
            s_all_w = s_all_w.reshape(s_all.shape)
            w = s_all_w[:nIreal[1] + 1, :nIreal[2] + 1].astype(power.dtype)

            sd = dw.downsample((s * w), down[1:])
            s2d = dw.downsample((s2 * w), down[1:])
            wd = dw.downsample(w, down[1:])
            sd /= wd
            sd[np.isnan(sd)] = 0.0
            s2d /= wd
            s2d[np.isnan(s2d)] = 0.0

            np.save(outname, sd)
            np.save(outname.replace('_s', '_w'), wd)
            np.save(outname.replace('_s', '_s2'), s2d)

        ax[0].cla()
        wd0 = wd > 0.0
        if np.any(wd0):
            vmin = np.min(sd[wd0])
            vmax = np.max(sd[wd0])
        else:
            vmin = None
            vmax = None
        ax[0].cla()
        ax[0].imshow(sd, vmin=vmin, vmax=vmax)
        ax[2].cla()
        ax[2].imshow(wd, vmin=0, vmax=1)
        working.append(sd)
        working2.append(s2d)
        workingw.append(wd)

        if len(working) == down[0]:
            workingw_stack = np.stack(workingw)
            out = dw.downsample(np.stack(working) * workingw_stack, [down[0], 1, 1])
            out2 = dw.downsample(np.stack(working2) * workingw_stack, [down[0], 1, 1])
            outw = dw.downsample(workingw_stack, [down[0], 1, 1])
            out /= outw
            out[np.isnan(out)] = 0.0
            out2 /= outw
            out2[np.isnan(out2)] = 0.0
            outstd = out2 - out ** 2
            outstd[outstd < 0] = 0
            outstd = np.sqrt(outstd)
            wd0 = (wd > 0.0)[None]
            if np.any(wd0):
                outshow = (out[0] - np.min(out[wd0])) / (np.quantile(out[wd0], 0.99) - np.min(out[wd0]))
                outshowstd = (outstd[0] - np.min(outstd[wd0])) / (np.quantile(outstd[wd0], 0.99) - np.min(outstd[wd0]))
            else:
                outshow = (out[0] - np.min(out)) / (np.quantile(out, 0.99) - np.min(out))
                outshowstd = (outstd[0] - np.min(outstd)) / (np.quantile(outstd, 0.99) - np.min(outstd))
            ax[1].cla()
            ax[1].imshow(np.stack((outshow, outshowstd, outshow), -1))
            ax[3].cla()
            ax[3].imshow(outw[0], vmin=0, vmax=1)
            output.append(out)
            output2.append(out2)
            outputw.append(outw)
            working = []
            workingw = []
            working2 = []
        fig.suptitle(f'slice {i} of {data.shape[0]}')
        fig.canvas.draw()
        print(f'Finished loading slice {i} of {data.shape[0]}, time {time.time() - starti} s')
    output = np.concatenate(output)
    Id = output
    wd = np.concatenate(outputw)
    print(f'Finished downsampling, time {time.time() - start}')

    np.savez(output_filename,I=Id,I2=np.concatenate(output2),xI=np.array(xId,dtype='object'),w=wd) # note specify object to avoid "ragged" warning

    fig, ax = dw.draw_slices(Id, xId)
    fig.suptitle(output_filename)
    fig.savefig(output_filename.replace('npz', 'jpg'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script downsamples lightsheet data')
    parser.add_argument('input_path',
                        help='Path to input image')
    parser.add_argument('--channel', '-c',
                        required=True,
                        help='Channel being processed')
    parser.add_argument('--image_type', '-t',
                        required=True,
                        help='Type of image, can be ims or Tif')
    parser.add_argument('--output_filename', '-o',
                        help='Name of output file, will automatically generate if none given')
    parser.add_argument('--dI', '-d')
                        # TODO: 'determine what dI means'
    parser.add_argument('--resolution', '-res',
                        help='Micron size used for resolution',
                        default=50.0)


    args = parser.parse_args()
    input_path = args.input_path
    image_type = args.image_type
    output_filename = args.output_filename
    dI = args.dI
    res = float(args.resolution)
    channel = args.channel

    downsample(input_path=input_path, image_type=image_type, output_filename=output_filename,
               dI=dI, res=res, channel=channel)
