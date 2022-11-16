#!/usr/bin/env python
# coding: utf-8

# I will deal with high res by working in blocks.


# NOTE
# if any part of a chunk is accessed, the whole chunk is read.
# we need to update our chunking here
# not sure if I actually can


import numpy as np
import tifffile
import h5py
import matplotlib.pyplot as plt
import sys
import emlddmm
import src.donglab_workflows as dw
import torch
import time
import os
import argparse


def transform(image_name=None, transformation_name=None, atlas_name=None, output_path=None, output_prepend=None):
        
    datastr = 'DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data'

    os.makedirs(output_path,exist_ok=True)
    tformdata = np.load(transformation_name,allow_pickle=True).item()

    # set up a domain
    with h5py.File(image_name, 'r') as f:

        dJ = dw.imaris_get_pixel_size(f)
        xJ = dw.imaris_get_x(f)
        xJ0 = np.array([x[0] for x in xJ])
        nJ = np.array([len(x) for x in xJ])

        xI_,I_,title_,names_ = emlddmm.read_data(atlas_name)

        xI = [np.arange(x[0],x[-1],dx) for x,dx in zip(xI_,dJ)]

        nI = np.array([len(x) for x in xI])

        blocksize = 256
        nblocks = np.ceil(nI/blocksize).astype(int)

        tform_ = emlddmm.compose_sequence(
                        [
                            emlddmm.Transform(tformdata['v'],domain=tformdata['xv']),
                            emlddmm.Transform(tformdata['A'])
                        ], 
                        np.meshgrid(*tformdata['xv'],indexing='ij')
                    )

        # make an image for a given slice
        fig,ax = plt.subplots()
        downshow = 5
        I = np.zeros(nI[1:], dtype=np.uint8)
        # loop over slices
        s0 = 0 # should be 0, just for testing
        for s in range(s0,nI[0]):
            starttime = time.time()
            outname = os.path.join(output_path,f'{output_prepend}{s:06d}.tif')
            print(outname)
            # if this file exists, skip it
            if os.path.exists(outname):
                continue
            # loop over blocks?
            i0 = 0 # should be 0
            for i in range(i0,nblocks[1]):
                # indices
                indi = np.arange(blocksize) + i * blocksize
                indi = indi[indi < nI[1]]
                # locations
                xi = xI[1][indi]
                for j in range(nblocks[2]):
                    # find the indices
                    indj = np.arange(blocksize) + j * blocksize
                    indj = indj[indj < nI[2]]            
                    # get the locations
                    xj = xI[2][indj]
                    X = np.stack(np.meshgrid(xI[0][s],xi,xj,indexing='ij'))
                    # regenerating this transform every time is slow, but for now I'm not going to worry
                    # better would be just to generate it on the xv grid once, then interpolate it
                    #tform = emlddmm.compose_sequence(
                    #    [
                    #        emlddmm.Transform(tformdata['v'],domain=tformdata['xv']),
                    #        emlddmm.Transform(tformdata['A'])
                    #    ], 
                    #    X
                    #)
                    #start = time.time()
                    tform = emlddmm.apply_transform_float(tformdata['xv'],tform_,torch.tensor(X,dtype=tform_.dtype))
                    #print(f'calculating tform {time.time()-start}')
                    
                    # now we need a bounding box
                    inds = (tform.numpy() - xJ0[:,None,None,None])/dJ[:,None,None,None]
                    # boudnaries
                    inds[inds<0] = 0
                    inds[0,inds[0]>=nJ[0]] = nJ[0]-1
                    inds[1,inds[1]>=nJ[1]] = nJ[1]-1
                    inds[2,inds[2]>=nJ[2]] = nJ[2]-1
                    mininds = np.floor(np.min(inds,axis=(1,2,3))).astype(int)
                    maxinds = np.ceil(np.max(inds,axis=(1,2,3))).astype(int)            
                    # need some boundary conditions, if I get an error I'll deal with it then
                    # now we load the data
                    
                    start = time.time()
                    data = f[datastr][
                        mininds[0]:maxinds[0]+1,
                        mininds[1]:maxinds[1]+1,
                        mininds[2]:maxinds[2]+1
                    ]
                    #print(f'loading data {time.time()-start}')
                    # okay now I've got a data cube, I need to sample it
                    indsr = np.round(inds).astype(int)
                    # subtract my index
                    indsr0 = indsr - mininds[:,None,None,None]
                    # okay I got an error here
                    # I did bc above, don't need to do again
                    #indsr0[indsr0 < 0] = 0
                    #indsr0[0,indsr0[0]>=data.shape[0]] = data.shape[0]-1
                    #indsr0[1,indsr0[1]>=data.shape[1]] = data.shape[1]-1
                    #indsr0[2,indsr0[2]>=data.shape[2]] = data.shape[2]-1
                    outputdata = data[indsr0[0],indsr0[1],indsr0[2]]

                    # something wrong with below?
                    indij = np.meshgrid(indi,indj,indexing='ij')
                    I[indij[0],indij[1]] = outputdata
                    
                    
                    #ax.cla()
                    #ax.imshow(dw.downsample(I**0.25,[8,8]))
                    #ax.set_title(f'slice {s} row {i} col {j}')
                    #fig.canvas.draw()
                    
                    
                    
                    # dne this column
                start = time.time()
                ax.cla()
                ax.imshow(dw.downsample(I**0.125,[16,16]))
                ax.set_title(f'slice {s} row {i} col {j}')
                fig.canvas.draw()
                #print(f'drawing {time.time() - start}')
                # done this row
            # done this slice, now we have to save it
            tifffile.imwrite(outname,I,compression='jpeg')
            print(f'took {time.time() - starttime}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script applies saved transforms to lightsheet data')
    parser.add_argument('image_name',
                        help='Path to input image')
    parser.add_argument('transformation_path',
                        help='Path to transformation file')
    parser.add_argument('atlas', '-a',
                        help='Specify the filename of the image to be transformed (atlas)')
    parser.add_argument('output_directory', '-d',
                        help='Where should outputs go?')
    parser.add_argument('output_file', '-o',
                        help='Name of save file')


    args = parser.parse_args()
    image_name = args.image_name
    transformation_name = args.transformation_path
    atlas_name = args.atlas
    output_path = args.output_directory
    output_prepend = args.output_file


    transform(image_name=image_name, transformation_name=transformation_name, 
              atlas_name=atlas_name, output_path=output_path, output_prepend=output_prepend)

    