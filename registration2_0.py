#!/usr/bin/env python
# coding: utf-8


# meeting notes on april 24, 2024
# Luis pasted his current file
# Luis, can you paste your whole previous py file here?
# 3d_pipeline.py
# note, daniel usually uses "sphinx" with "napoleon"
# for documentating code: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
# NOTE THIS FMOST ATLAS IS 10 UM, this is hard coded because it's not described anywhere in the tif file
# NOTE about weighted image mathing for lightsheet.  the weights data['w'] did not end up being useful.  We will set all weights to 1.
# NOTE when generating figures, minval, maxval parameters are hard coded and are used to map
# grayscale images to 0,1.  These might need to be changed for new data.
# NOTE in preprocess atlas, there was an error where xI was not updating itself in the loop
# NOTE atlas and target orientation are different for this example
# the initial affine transform function can input three leter strings (e.g. "RAS") to set atlas and
# target orientation
# NOTE the main work we want to do is in "register" and "run_registration"
# NOTE Sumit wanted some speciic outputs, and did not want many of the of the outputs we
# had produced previously.  Today we will generate Sumit's outputs.  Luis will decide what to do with the others
# potentially just comment them out.
# NOTE this code makes some copies that are unnecessary. J, J0, J_.  We can probably remove these, but have to be careful about it.
# if you're not worried about memory we don't have to do anything.
# let's say for today, we'll not worry about.  If you run into memory problems,
# we can look at removing these copies next time.  Or Luis can try if he wants.
# NOTE in run registration I am removing the resolution input argument
# we will just assume 10,10,10 for atlas and target both.

import argparse
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime
import pandas as pd
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tifffile # needed to load the fmost atlas labels
import torch # so we can specify to use the torch.float32 data type
import sys

import emlddmm

FIG_OPTIONS = {'dpi': 300, 'format': 'jpg'}

def get_target(data):
    '''
    This function loads a target image from our npz data file.
    It extracts it from the dictionary, casts it to float32
    and normalizes it.

    Parameters
    ----------
    data : dictionary
        dictionary loaded from an npz file.  It should have the field 'I'

    Returns
    -------
    target :  numpy array
        The image data in a  1 x slice  x row x col nupmy array

    '''
    target = data['I'][None]
    target = target.astype(np.float32)
    target /= np.mean(np.abs(target))
    return target

def get_atlas(atlases):
    '''
    Load atlases from a list of filenames.

    For FMOST, we will only load one atlas which is a tif file.

    But the code will still support the Allen atlas, which
    Daniel has saved as .vtk files.

    Parameters
    ----------
    atlases : list of strings
        A list of strings of filenames, either .vtk or .tif to load an atlas image

    Returns
    -------
    I : numpy array
        A natlases x slices x rows x cols numpy array storing imaging data
    xI : list of numpy array
        Location of the pixels in the slice, row, col dimensions (generally interpretted as z,y,x)
    '''
    I = []
    for atlas_name in atlases:
        if str(atlas_name).endswith('.vtk'):
            xI,I_,title,names = emlddmm.read_data(atlas_name)
            I_ = I_.astype(np.float32)
            I_ /= np.mean(np.abs(I_))
        elif str(atlas_name).endswith('.tif'):
            I_ = tifffile.imread(atlas_name)
            # NOTE THIS FMOST ATLAS IS 10 UM
            ATLAS_RESOLUTION = 10.0
            xI = [np.arange(n)*ATLAS_RESOLUTION - (n-1)*ATLAS_RESOLUTION/2 for n in I_.shape]
            if I_.ndim == 3:
                I_ = I_[None]
        I.append(I_)
    I = np.concatenate(I) # need to comment this out for fmost
    return I, xI

def get_seg_atlas(seg):
    ''' We probably will not use this function for FMOST
    becuase we don't have a grayscale image, we only have a segmenttion which is loaded above.
    '''
    xS, S, _, _ = emlddmm.read_data(seg)
    return S, xS

def get_RGB(seg):
    '''
    Convert an image full of integer labels, to an RGB image with random colors

    This is for visualization.
    '''
    labels,inds = np.unique(seg,return_inverse=True)

    colors = np.random.rand(len(labels),3)
    colors[0] = 0.0

    return colors[inds].reshape(seg.shape[1],seg.shape[2],seg.shape[3],3).transpose(-1,0,1,2)

def get_voxel_spacing(data):
    '''
    Parameters
    ----------
    data : dictionary
        Dictionary loaded from an npz file with a field 'xI' storing voxel locations

    Returns
    -------
    data['xI'] : list of numpy arrays
        Voxel locations in the slice, row, col dimensions.  usually units of microns
    '''
    return data['xI']

def get_w(data, image):
    ''' Get weights from npz file.
    For our lightsheet images, these tend to be really bad.
    For FMOST, I will just set the weights to 1.
    '''
    if 'w' in data:
        return data['w']
    elif 'W' in data:
        return data['W']
    else:
        return (image[0]>0).astype(float)

def get_origin(vox_data):
    '''
    Input a list of voxel locations (slice, row, col), and output the first element .
    '''
    return np.array([vox_data[0][0], vox_data[1][0], vox_data[2][0]])

def save_metadata(target, segmentation, outpath, affine, a_orientation, t_orientation, preprocessed):
    with open(os.path.join(outpath, "registration_metadata.txt"), 'w') as f:
        metadata = f"Date: {str(datetime.now())}\n"
        metadata += f"Downsampled File: {target}\n"
        metadata += f"Segmentation File: {segmentation}\n"
        metadata += f"Initial Affine: {affine}\n"
        metadata += f"Atlas Orientation: {a_orientation}\n"
        metadata += f"Target Orientation: {t_orientation}\n"
        metadata += f"Input Image Preprocessed: {preprocessed}\n"
        f.write(metadata)

def save_figure(figure, name, outpath="", title=""):
    print(f"Generating {name} figure")
    figure.suptitle(title)
    figure.savefig(os.path.join(outpath, name + ".jpg"), **FIG_OPTIONS)
    plt.close()

def generate_atlas_space_figure(I, S, RGB, Jt, outdir):
    '''
    Daniel wants to leave this as is, but these minval, maxval parameters
    are used to normalize contrast.   They may not work well for new datasets.
    We may need to modify them.
    '''
    # view the transformed target with labels
    minval = 1.5
    maxval = 2.9
    minval = 0.0
    maxval = 5.0
    alpha = 0.3
    alpha = 0.75
    fig = plt.figure(figsize=(7,7))
    n = 4
    slices = np.round(np.linspace(0,I.shape[1],n+2)[1:-1]).astype(int)
    for i in range(n):
        ax = fig.add_subplot(3,n,i+1)

        # get slices
        RGB_ = RGB[:,slices[i]].transpose(1,2,0)
        S_ = S[0,slices[i]]
        Jt_ = np.copy(Jt[0,slices[i]])
        Jt_ -= minval
        Jt_ /= maxval-minval
        # find boundaries
        border = S_ != np.roll(S_,shift=1,axis=0)
        border |= S_ != np.roll(S_,shift=-1,axis=0)
        border |= S_ != np.roll(S_,shift=1,axis=1)
        border |= S_ != np.roll(S_,shift=-1,axis=1)

        # draw
        ax.imshow(alpha*border[...,None]*RGB_ + ((1-alpha*border)*Jt_)[...,None])
        if i>0:
            ax.set_xticks([])
            ax.set_yticks([])

    slices = np.round(np.linspace(0,I.shape[2],n+2)[1:-1]).astype(int)
    for i in range(n):
        ax = fig.add_subplot(3,n,i+1 + n)

        # get slices
        RGB_ = RGB[:,:,slices[i]].transpose(1,2,0)
        S_ = S[0,:,slices[i]]
        Jt_ = np.copy(Jt[0,:,slices[i]])
        Jt_ -= minval
        Jt_ /= maxval-minval
        # find boundaries
        border = S_ != np.roll(S_,shift=1,axis=0)
        border |= S_ != np.roll(S_,shift=-1,axis=0)
        border |= S_ != np.roll(S_,shift=1,axis=1)
        border |= S_ != np.roll(S_,shift=-1,axis=1)

        # draw
        ax.imshow(alpha*border[...,None]*RGB_ + ((1-alpha*border)*Jt_)[...,None])
        if i>0:
            ax.set_xticks([])
            ax.set_yticks([])

    slices = np.round(np.linspace(0,I.shape[3],n+2)[1:-1]).astype(int)
    for i in range(n):
        ax = fig.add_subplot(3,n,i+1 + n+n)

        # get slices
        RGB_ = RGB[:,:,:,slices[i]].transpose(1,2,0)
        S_ = S[0,:,:,slices[i]]
        Jt_ = np.copy(Jt[0,:,:,slices[i]])
        Jt_ -= minval
        Jt_ /= maxval-minval
        # find boundaries
        border = S_ != np.roll(S_,shift=1,axis=0)
        border |= S_ != np.roll(S_,shift=-1,axis=0)
        border |= S_ != np.roll(S_,shift=1,axis=1)
        border |= S_ != np.roll(S_,shift=-1,axis=1)

        # draw
        ax.imshow(alpha*border[...,None]*RGB_ + ((1-alpha*border)*Jt_)[...,None])
        if i>0:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle('Atlas space')
    fig.subplots_adjust(wspace=0,hspace=0,left=0.0,right=1,bottom=0,top=0.95)
    fig.savefig(os.path.join(outdir, 'atlas_space.jpg'), **FIG_OPTIONS)

def generate_target_space_figure(J, RGBt, St, outdir):
    '''
    Daniel wants to leave this as is, but these minval, maxval parameters
    are used to normalize contrast.   They may not work well for new datasets.
    We may need to modify them.
    '''
    minval = 1.5
    maxval = 2.9
    minval = 0.0
    maxval = 5.0
    alpha = 0.3
    alpha = 0.75
    # Generate transformed labels with the target
    n = 4
    fig = plt.figure(figsize=(8,5))
    slices = np.round(np.linspace(0,J.shape[1],n+2)[1:-1]).astype(int)
    for i in range(n):
        ax = fig.add_subplot(3,n,i+1)

        # get slices
        RGB_ = RGBt[:,slices[i]].transpose(1,2,0)
        S_ = St[0,slices[i]]
        Jt_ = np.copy(J[0,slices[i]])
        Jt_ -= minval
        Jt_ /= maxval-minval
        # find boundaries
        border = S_ != np.roll(S_,shift=1,axis=0)
        border |= S_ != np.roll(S_,shift=-1,axis=0)
        border |= S_ != np.roll(S_,shift=1,axis=1)
        border |= S_ != np.roll(S_,shift=-1,axis=1)

        # draw
        ax.imshow(alpha*border[...,None]*RGB_ + ((1-alpha*border)*Jt_)[...,None])
        if i>0:
            ax.set_xticks([])
            ax.set_yticks([])

    slices = np.round(np.linspace(0,J.shape[2],n+2)[1:-1]).astype(int)
    for i in range(n):
        ax = fig.add_subplot(3,n,i+1 + n)

        # get slices
        RGB_ = RGBt[:,:,slices[i]].transpose(1,2,0)
        S_ = St[0,:,slices[i]]
        Jt_ = np.copy(J[0,:,slices[i]])
        Jt_ -= minval
        Jt_ /= maxval-minval
        # find boundaries
        border = S_ != np.roll(S_,shift=1,axis=0)
        border |= S_ != np.roll(S_,shift=-1,axis=0)
        border |= S_ != np.roll(S_,shift=1,axis=1)
        border |= S_ != np.roll(S_,shift=-1,axis=1)

        # draw
        ax.imshow(alpha*border[...,None]*RGB_ + ((1-alpha*border)*Jt_)[...,None])
        if i>0:
            ax.set_xticks([])
            ax.set_yticks([])

    slices = np.round(np.linspace(0,J.shape[3],n+2)[1:-1]).astype(int)
    for i in range(n):
        ax = fig.add_subplot(3,n,i+1 + n+n)

        # get slices
        RGB_ = RGBt[:,:,:,slices[i]].transpose(1,2,0)
        S_ = St[0,:,:,slices[i]]
        Jt_ = np.copy(J[0,:,:,slices[i]])
        Jt_ -= minval
        Jt_ /= maxval-minval
        # find boundaries
        border = S_ != np.roll(S_,shift=1,axis=0)
        border |= S_ != np.roll(S_,shift=-1,axis=0)
        border |= S_ != np.roll(S_,shift=1,axis=1)
        border |= S_ != np.roll(S_,shift=-1,axis=1)

        # draw
        ax.imshow(alpha*border[...,None]*RGB_ + ((1-alpha*border)*Jt_)[...,None])
        if i>0:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle('Target space')
    fig.subplots_adjust(wspace=0,hspace=0,left=0.0,right=1,bottom=0,top=0.95)
    fig.savefig(os.path.join(outdir, 'target_space.jpg'), **FIG_OPTIONS)

def preprocess_target(target, W):
    ''' for lightsheet to fmost, Daniel used a simpler preprocessing
    '''
    # background
    #processed_target = target - np.quantile(target[W[None]>0.9],0.1)
    #processed_target[processed_target<0] = 0
    # adjust dynamic range
    #processed_target = processed_target**0.25
    Jpower = 1/8
    processed_target = target ** Jpower
    # adjust mean value
    processed_target /= np.mean(np.abs(processed_target))

    return processed_target

def preprocess_atlas(atlas, dI,  xI0):
    # since downsample by 4, pad with 4x4x4
    # at higher resolution we will do more padding
    npad = 16
    I = np.pad(atlas, ((0,0),(npad,npad),(npad,npad),(npad,npad)) )
    for _ in range(npad):
        xI0 = [np.concatenate((x[0][None] - d, x, x[-1][None] + d)) for d, x in zip(dI,xI0)]

    # adjust nissl image dynamic range
    #I[0] = I[0]**0.5
    #I[0] /= np.mean(np.abs(I[0]))
    #I[1] /= np.mean(np.abs(I[1]))

    # normalization is not necessary anymore, since we are working with a label image directly

    return I, xI0

def get_initial_affine(A0, XJ, atlas_orientation='PIR', target_orientation='SAL'):
    '''
    We added optional input arguments in case the orientation is not the same as our test case
    '''
    if A0 == None:
        A0 = np.eye(4)

        # these letters are commonly used in human MRI
        L0 = emlddmm.orientation_to_orientation(atlas_orientation,target_orientation)
        A0[:3,:3] = L0
        A0[:3,-1] = np.mean(XJ,axis=(-1,-2,-3))

    else:
        A0 = np.fromstring(A0,sep=',').reshape(4,4)

    return A0

def run_registration(J, xJ, I, xI, W, A0, dI, device):

    # start at the lowest scale with only linear transform
    config0 = {
        'device':device,
        'dtype':torch.float32, # try single
        'n_iter':200, 'downI':[20,20,20], 'downJ':[20,20,20], # start at 200 microns
        'priors':[0.9,0.05,0.05],'update_priors':False,
        'update_muA':0,'muA':[np.quantile(J,0.99)],
        'update_muB':0,'muB':[np.quantile(J,0.01)],
        'update_sigmaM':0,'update_sigmaA':0,'update_sigmaB':0,
        'order':1,'n_draw':50,'n_estep':3,'slice_matching':0,'v_start':1000,
        'eA':5e4,'A':A0,'full_outputs':True,
        'dv':1000.0, # set dv really big here since I'm not using it
   }

    config0['sigmaM'] = 2.0
    config0['sigmaB'] = 4.0
    config0['sigmaA'] = 8.0
    # let's downsample first, this can help with memory
    config0['downI'] = [1,1,1]
    config0['downJ'] = [1,1,1]
    xId,Id = emlddmm.downsample_image_domain(xI,I,[20,20,20])
    xJd,Jd,Wd = emlddmm.downsample_image_domain(xJ,J,[20,20,20],W=W)

    # free up any possibly memory
    #import gc
    #import time
    #gc.collect()
    #time.sleep(1)
    #gc.collect()
    out = emlddmm.emlddmm(xI=xId,I=Id,xJ=xJd,J=Jd, W0=Wd, **config0)

    #out.pop('WM')
    #out.pop('WA')
    #out.pop('WB')
    #out.pop('W0')
    #out.pop('coeffs')
    #import gc
    #gc.collect()
    #time.sleep(1)
    #gc.collect()



    # second run, with deformation
    # use the same resolution, but with deformation
    config1 = dict(config0)
    config1['A'] = out['A']
    config1['eA'] = config0['eA']*0.1
    config1['a'] = 1000.0

    config1['n_iter']= 2000
    config1['v_start'] = 0
    config1['ev'] = 1e-2
    config1['ev'] = 2e-3 # reduce since I decreased sigma
    config1['dv'] = 1000.0
    config1['local_contrast'] = [16,16,16]
    config1['sigmaR'] = 1e4 # this looks pretty good

    # use the same resolution as previous cell
    #gc.collect()
    #time.sleep(1)
    #gc.collect()
    out1 = emlddmm.emlddmm(xI=xId,I=Id,xJ=xJd,J=Jd, W0=Wd, **config1)
    # delete some things I don't need
    #out1.pop('WM')
    #out1.pop('WA')
    #out1.pop('WB')
    #out1.pop('W0')
    #out1.pop('coeffs')
    #gc.collect()
    #time.sleep(1)
    #gc.collect()


    # third run, with deformation and higher resolution
    config2 = dict(config1)
    config2['A'] = out1['A']
    config2['n_iter']= 1000
    config2['v'] = out1['v']

    # now go to 100 microns
    xId,Id = emlddmm.downsample_image_domain(xI,I,[10,10,10])
    xJd,Jd,Wd = emlddmm.downsample_image_domain(xJ,J,[10,10,10],W=W)


    # free up any possible memory
    #import gc
    #import time
    #gc.collect()
    #time.sleep(1)
    #gc.collect()
    out2 = emlddmm.emlddmm(xI=xId,I=Id,xJ=xJd,J=Jd, W0=Wd, **config2)
    # delete some things I don't need
    #out2.pop('WM')
    #out2.pop('WA')
    #out2.pop('WB')
    #out2.pop('W0')
    #out2.pop('coeffs')
    #gc.collect()
    #time.sleep(1)
    #gc.collect()


    # again at a higher resolution
    # add one more spatial scale, 50um
    # on the next run we do les downsampling
    config3 = dict(config2)
    config3['A'] = out2['A']
    config3['v'] = out2['v']
    # a small number of iterations with a smaller step size to finish up
    config3['n_iter'] = 100
    config3['eA'] = config3['eA']/4
    config3['ev'] = config3['ev']/4

    # now go to 50 microns
    xId,Id = emlddmm.downsample_image_domain(xI,I,[5,5,5])
    xJd,Jd,Wd = emlddmm.downsample_image_domain(xJ,J,[5,5,5],W=W)


    # free up any possibly memory
    #import gc
    #import time
    #gc.collect()
    #time.sleep(1)
    #gc.collect()


    # there seems to be an issue with the initial velocity
    # when I run this twice, I'm reusing it
    out3 = emlddmm.emlddmm(xI=xId,I=Id,xJ=xJd,J=Jd, W0=Wd, **config3)
    # the matching energy here is way way lower, why would that be?

    # delete some things I don't need
    # out3.pop('WM')
    # out3.pop('WA')
    # out3.pop('WB')
    # out3.pop('W0')
    # out3.pop('coeffs')
    # gc.collect()
    # time.sleep(1)
    # gc.collect()

    return out3



def generate_error_figure(fig, outdir):
    axs = fig.get_axes()
    for ax in axs:
        ims = ax.get_images()
        for im in ims:
            im.set_cmap('twilight')
            clim = im.get_clim()
            lim = np.max(np.abs(clim))
            im.set_clim(np.array((-1,1))*lim)
    save_figure(fig, name="err", outpath=outdir)

def get_ontology(ontology_path):
    parent_column = 7  # 8 for allen, 7 for yongsoo
    label_column = 0  # 0 for both
    shortname_column = 2  # 3 for allen, 2 for yongsoo
    longname_column = 1  # 2 for allen, 1 for yongsoo

    ontology = dict()
    with open(ontology_path) as f:
        csvreader = csv.reader(f, delimiter=',', quotechar='"')
        count = 0
        for row in csvreader:
            if count == 0:
                headers = row
                print(headers)
            else:
                if not row[parent_column]:
                    parent = -1
                else:
                    parent = int(row[parent_column])
                ontology[int(row[label_column])] = (row[shortname_column],row[longname_column],parent)
            count += 1
    return ontology

def get_descendents_and_self_dict(ontology):
        # we need to find all the descendants of a given label
    # first we'll get children
    children = dict()
    for o in ontology:
        parent = ontology[o][-1]
        if parent not in children:
            children[parent] = []
        children[parent].append(o)


    # now we go from children to descendents
    descendents = dict(children)
    for o in descendents:
        for child in descendents[o]:
            if child in descendents: # if I don't do this i get a key error 0
                descendents[o].extend(descendents[child])
    descendents[0] = []

    descendents_and_self = dict(descendents)
    for o in ontology:
        if o not in descendents_and_self:
            descendents_and_self[o] = [o]
        else:
            descendents_and_self[o].append(o)
    return descendents_and_self

def compute_face_normals(verts,faces,normalize=False):
    e1 = verts[faces[:,1]] - verts[faces[:,0]]
    e2 = verts[faces[:,2]] - verts[faces[:,1]]
    n = np.cross(e1,e2)/2.0
    if normalize:
        n /= np.sqrt(np.sum(n**2,1,keepdims=True))
        pass
    return n

def generate_bboxes(xJ, St, descendents_and_self, labels, ontology, outpath):
    bbox = dict()

    for l in labels:
        # skip background
        if l == 0:
            continue

        Sl = St == l

        # include all the descendents
        for o in descendents_and_self[l]:
            Sl = np.logical_or(Sl,St==o)

        bbox2 = xJ[2][np.nonzero(np.sum(Sl,(0,1,2))>0)[0][[0,-1]]]
        bbox1 = xJ[1][np.nonzero(np.sum(Sl,(0,1,3))>0)[0][[0,-1]]]
        bbox0 = xJ[0][np.nonzero(np.sum(Sl,(0,2,3))>0)[0][[0,-1]]]
        bbox[l] = (bbox2[0],bbox2[1],bbox1[0],bbox1[1],bbox0[0],bbox0[1],ontology[l][0],ontology[l][1])

    df = pd.DataFrame(bbox).T
    bbox_headings = ('x0','x1','y0','y1','z0','z1','short name','long name')
    df.columns=bbox_headings
    df.index.name = 'id'
    df.to_csv(os.path.join(outpath, 'bboxes.csv'))

def generate_3d_surface_plot(verts, faces, ontology, label, outpath):
    fig = plt.figure()
    surf = Poly3DCollection(verts[faces])
    n = compute_face_normals(verts,faces,normalize=True)
    surf.set_color(n*0.5+0.5)
    fig.clf()
    ax = fig.add_subplot(projection='3d')
    ax.add_collection3d(surf)
    xlim = (np.min(verts[:,0]),np.max(verts[:,0]))
    ylim = (np.min(verts[:,1]),np.max(verts[:,1]))
    zlim = (np.min(verts[:,2]),np.max(verts[:,2]))
    # fix aspect ratio
    r = [np.diff(x) for x in (xlim,ylim,zlim)]
    rmax = np.max(r)
    c = [np.mean(x) for x in (xlim,ylim,zlim)]
    xlim = (c[0]-rmax/2,c[0]+rmax/2)
    ylim = (c[1]-rmax/2,c[1]+rmax/2)
    zlim = (c[2]-rmax/2,c[2]+rmax/2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    
    ax.set_title(f'structure {label}, {ontology[label][1]} ({ontology[label][0]})')    
    fig.savefig(os.path.join(outpath, f'structure_{label:012d}_surface.jpg'))
    plt.close()

def generate_npz_mesh_file(l, verts, faces, normals, values, ontology, origin, outpath):
    
    # let's save this
    readme_dct = {'notes' : 'Data are saved in ZYX order',
                  'atlas_id' : ontology[l][1],
                  'id' : ontology[l][0] }
    readme = str(readme_dct)
    # Clean id to prevent region names interfering with file name
    clean_id = readme_dct["id"]
    for char in ['/', "' ", ', ', " "]:
        clean_id = clean_id.replace(char, '_')

    struct_des_fname = os.path.join(outpath, f'structure_{l:012d}_surface_{clean_id}.npz')
    np.savez(struct_des_fname, verts=verts,faces=faces,normals=normals,values=values,readme=readme, origin=origin)

def register():
    parser = argparse.ArgumentParser(description="Register images in atlas")

    parser.add_argument("target_name",type=pathlib.Path,help="Name of the file to be registered")
    parser.add_argument("output_prefix",type=pathlib.Path,help="Location of the output file(s)")
    parser.add_argument("savename",type=pathlib.Path,help="Name of file once it is saved")
    parser.add_argument("atlas_names",type=pathlib.Path,help="Location of the atlas images")
    parser.add_argument("-v","--ventricle",type=pathlib.Path,default=None,help="Location of the ventricle opening npz file")
    parser.add_argument("-on", "--onto_name", default=None, help="Location of atlas ontology")
    parser.add_argument("-d","--device",default='cuda:0',help="Device used for pytorch")
    parser.add_argument("-ao","--atlas_orientation",default='PIR',
                        help="3 letters, one for each image axis: R/L, A/P, S/I")
    parser.add_argument("-to","--target_orientation",default='SAL',
                        help="3 letters, one for each image axis: R/L, A/P, S/I")
    parser.add_argument("-a","--A0",type=str,default=None,help="Affine transformation (Squeezed to 16x1 + Sep by ',')")
    parser.add_argument("-j","--jpg", action="store_true", help="Generate 3d jpegs for each structure")
    parser.add_argument("-pp","--preprocessed", action="store_true", help="Specifies if image has been previously preprocessed")

    args = parser.parse_args()

    target_name   = args.target_name
    output_prefix = args.output_prefix
    atlas_name    = args.atlas_names
    ventricle_map = args.ventricle
    savename      = args.savename
    ontology_name = args.onto_name
    jpg           = args.jpg
    device        = args.device
    A0            = args.A0
    atlas_orient  = args.atlas_orientation
    target_orient = args.target_orientation
    preprocessed  = args.preprocessed


    # Validations
    assert os.path.isfile(target_name), f"Downsample file was not found: {target_name}"
    assert os.path.isfile(atlas_name), f"Atlas image file was not found: {atlas_name}"
    assert os.path.isfile(ontology_name), f"Atlas ontology file was not found: {ontology_name}"
    if not os.path.exists(pathlib.Path(output_prefix)):
        os.mkdir(output_prefix)
    assert os.path.isdir(output_prefix), f"Output directory does not exist: {output_prefix}"
    downshow = 16

    # Record metadata
    print("Recording metadata")
    save_metadata(target_name, atlas_name, output_prefix, 'None', atlas_orient, target_orient, preprocessed)

    #####################
    ## LOADING IMAGES

    print("Loading target and voxel spacing")
    # Load the target image
    target_data = np.load(target_name,allow_pickle=True)

    # get image
    J = get_target(target_data)
    # get voxel spacing variable
    xJ = get_voxel_spacing(target_data)
    # make copy of target
    J0 = np.copy(J) # we probably don't need this copy, but if you're not worried about memory it's fine
    # get W variable
    W = get_w(target_data, image=J)
    # on april 24, 2024 daniel says we should just set W to 1 (i.e. unweighted image registration)
    W = W*0.0 + 1.0

    # Save downsample image figure
    fig, _ = emlddmm.draw(J,xJ,vmin=np.min(J[W[None]>0.9]))
    preprocessed_title = "" if not preprocessed else " (preprocessed)"
    save_figure(fig, "downsample", title=f'Downsampled lightsheet data{preprocessed_title}', outpath=output_prefix)
    print("Loading atlas")
    I, xI = get_atlas([atlas_name])

    dI = [x[1] - x[0] for x in xI]
    xI0 = [ np.copy(x)for x in xI]
    I0 = np.copy(I)
    # this copy is used for visualization after registration.
    # if it's too much memory, we can just load it again later rather than making a copy
    print("Loading and transforming segmentation atlas")
    ##########################
    ## PREPROCESSING

    print("Preprocessing target and atlas")
    # Target preprocessing
    if not preprocessed:
        J = preprocess_target(target=J0, W=W)

    # Save processed target figure
    fig, _ = emlddmm.draw(J,xJ,vmin=0)
    save_figure(fig, "processed", title="Preprocessed lightsheet data", outpath=output_prefix)

    # Atlas preprocessing
    I, xI = preprocess_atlas(atlas=I0, dI=dI, xI0=xI0)
    # in our work we have assigned random colors to each label
    labels,inds = np.unique(I.ravel(),return_inverse=True)
    np.random.seed(1) # for reproducibility
    d = 3
    colors = np.random.rand(len(labels),d).astype(np.float32)
    colors[0] = 0
    I = colors[inds].reshape(I.shape[1:]+(d,)).transpose(-1,0,1,2)
    # NOTE we have converted our atlas to an RGB image wiht random colors that will be used for reistration

    # Load the ventricle opening map and apply it to our atlas
    # this will open the ventricles
    if ventricle_map is not None:
        data = np.load(ventricle_map,allow_pickle=True)
        tform = emlddmm.Transform(data['phii'].transpose(-1,0,1,2),domain=data['xv'])
        I_ = np.zeros_like(I)
        fig = plt.figure()
        for i in range(I.shape[1]):
            if not i%100:
                print(i)
                fig,ax = emlddmm.draw(I_[:,::downshow,::downshow,::downshow],[x[::downshow] for x in xI],vmin=0,vmax=1,fig=fig)
                fig.canvas.draw()
            tmp = emlddmm.compose_sequence([tform],[xI[0][i], xI[1],xI[2]])
            I_[:,i] = emlddmm.apply_transform_float(xI,I,tmp)[:,0]
        I = I_

    #####################
    ## REGISTRATION

    print("Loading initial affine")
    XJ = np.meshgrid(*xJ,indexing='ij')
    A0 = get_initial_affine(A0=A0, XJ=XJ, atlas_orientation=atlas_orient, target_orientation=target_orient)

    # Visualize initial transformation
    tform = emlddmm.Transform(A0,direction='b')
    AI = emlddmm.apply_transform_float(xI,I,tform.apply(XJ))
    fig, _ = emlddmm.draw(np.concatenate((AI[:2],J)),xJ,vmin=0)
    save_figure(fig, "initial_atlas_space", title="Initial Affine Transformation", outpath=output_prefix)

    print("Registering")
    # now we want to register
    out2 = run_registration(J=J, xJ=xJ, I=I, xI=xI, W=W, A0=A0, dI=dI,
                            device=device) # removed resolution argument

    # Saving registration output
    np.save(os.path.join(output_prefix, savename), np.array([out2],dtype=object))

    # Saving figures
    save_figure(out2['figI'], name='transformed.jpg', outpath=output_prefix)
    save_figure(out2['figfI'], name='contrast.jpg', outpath=output_prefix)
    save_figure(out2['figErr'], name='err.jpg', outpath=output_prefix)
    generate_error_figure(out2['figErr'], outdir=output_prefix)


    ##############################
    ## SUMIT's OUTPUTS
    # he asked for a voxel to voxel map from atlas to target, and vice versa
    # with units of voxels (not microns) and origin in the corner
    dJ = [x[1] - x[0] for x in xJ]
    data = np.load(ventricle_map,allow_pickle=True)
    tform_vl = emlddmm.Transform(data['phi'].transpose(-1,0,1,2),domain=data['xv'])
    tform_v = emlddmm.Transform(out2['v'],'b',domain=out2['xv'])
    to_microns_target = np.eye(4)
    to_microns_target[:3,:3] = np.diag(dJ)
    to_microns_target[:3,-1] = [x[0] for x in xJ]
    to_microns_atlas = np.eye(4)
    to_microns_atlas[:3,:3] = np.diag(dI)
    to_microns_atlas[:3,-1] = [x[0] for x in xI]
    TFORM = np.zeros((3,)+J.shape[1:],dtype=np.float32)
    for i in range(J.shape[1]):
        if not i%100: print(f'{i} of {J.shape[1]-1}')

        tform = emlddmm.compose_sequence(
            [
                # we don't need to_microns_target, because we're already in microns
                emlddmm.Transform(out2['A'],'b'),
                tform_v,
                tform_vl,
                emlddmm.Transform(to_microns_atlas,'b')
            ],
            [xJ[0][i],xJ[1],xJ[2]]
        )
        TFORM[:,i] = (tform[:,0].to(torch.float32)).numpy()

    np.save(os.path.join(output_prefix,'target_pixel_to_atlas_pixel.npy'), TFORM)

    # now the other direction
    tform_vl = emlddmm.Transform(data['phii'].transpose(-1,0,1,2),domain=data['xv'])
    tform_v = emlddmm.Transform(out2['v'],'f',domain=out2['xv'])
    TFORM = np.zeros((3,)+I.shape[1:],dtype=np.float32)
    #gc.collect()
    for i in range(I.shape[1]):
        if not i%100: print(f'{i} of {I.shape[1]-1}')

        tform = emlddmm.compose_sequence(
            [
                # we don't need to_microns_atlas, because we're already in microns
                tform_vl,
                tform_v,
                emlddmm.Transform(out2['A']),
                emlddmm.Transform(to_microns_target,'b')
            ],
            [xI[0][i],xI[1],xI[2]]
        )
        TFORM[:,i] = (tform[:,0].to(torch.float32)).numpy()
    np.save(os.path.join(output_prefix,'atlas_pixel_to_target_pixel.npy'),TFORM)

    print(f"Ended at: {str(datetime.now())}\n")
    print("Stopping script here")
    sys.exit(0)
    assert False, "End of the line.  Review output please"

    # NOTE: Sumit did not ask for any more outputs
    # Luis and Daniel did not look at anything below here together
    # we can come back to this later, or you can end the code here.
    # the figures should be fine, but we will to revisit the structures bounding boxes
    # and surfaces, because we now have a different ontology.
    # if this is important, daniel will start working on it.

    ##############################
    ## VISUALIZATIONS

    print("Generating registration and transformation visualizations")

    ## next is to transforom the high resolution data
    #S, xS = get_seg_atlas(seg_name)
    # since our atlas image is already a segmentation we don't use another atlas
    xS = xI
    S = I0 # note this is a shallow copy, not using more memory
    # we want to visuze the above with S
    RGB = get_RGB(seg=S)


    # compute transform for atlas and labels
    deformation = emlddmm.Transform(out2['v'],domain=out2['xv'],direction='b')
    affine = emlddmm.Transform(out2['A'],direction='b')
    tform = emlddmm.compose_sequence([affine,deformation],XJ)

    # keeping reference to affine for saving
    affine_np = affine.data.numpy()

    # transform the atlas and labels, notice different domains
    It = emlddmm.apply_transform_float(xI,I,tform).cpu().numpy()
    RGBt = emlddmm.apply_transform_float(xS,RGB,tform).cpu().numpy()
    St = emlddmm.apply_transform_int(xS,S,tform,double=True,padding_mode='zeros').cpu().numpy()

    fig, _ = emlddmm.draw(np.stack((It[0]*0.5,It[1]*0.5,J0[0]*1.5)),xJ,)
    fig.subplots_adjust(wspace=0,hspace=0,right=1)
    fig.savefig(os.path.join(output_prefix, 'IJsave.jpg'))

    fig, _ = emlddmm.draw(It,xJ)
    fig.subplots_adjust(wspace=0,hspace=0,right=1)
    fig.savefig(os.path.join(output_prefix, 'Isave.jpg'))

    fig, _ = emlddmm.draw(J,xJ)
    plt.subplots_adjust(wspace=0,hspace=0,right=1)
    fig.savefig(os.path.join(output_prefix, 'Jsave.jpg'))

    # transform the target to atlas
    # for visualizatoin, we want to sample at xS so we can view it relative to the
    XS = np.stack(np.meshgrid(*xS,indexing='ij'))
    deformation = emlddmm.Transform(out2['v'],domain=out2['xv'],direction='f')
    affine = emlddmm.Transform(out2['A'],direction='f')
    tformi = emlddmm.compose_sequence([deformation,affine,],XS)
    #J_ = J0**0.25
    J_ = np.copy(J)
    Jt = emlddmm.apply_transform_float(xJ,J_,tformi,padding_mode='zeros').cpu().numpy()

    # view the transformed target
    fig,ax = emlddmm.draw(Jt,xS,vmin=np.quantile(J_,0.02),vmax=np.quantile(J_,0.98))
    fig.subplots_adjust(wspace=0,hspace=0,right=1)

    np.savez(os.path.join(output_prefix, "affine_and_transform_values"),
            affine1=affine_np,
            tform1=tform,
            affine2=affine.data.numpy(),
            tform2=tformi)

    # Generate atlas space visualization
    generate_atlas_space_figure(I=I, S=S, RGB=RGB, Jt=Jt, outdir=output_prefix)

    # Generate target space visualization
    generate_target_space_figure(J=J, RGBt=RGBt, St=St, outdir=output_prefix)

    ############################
    ## STRUCTURES

    print("Preparing Bounding Boxes")
    ### Get bounding boxes for striatum or another structure
    # Get ontology from csv path
    ontology = get_ontology(ontology_name)

    # Get descendents and self dict from ontology
    descendents_and_self = get_descendents_and_self_dict(ontology)

    labels = np.unique(St)
    print("Saving Bounding boxes to file")
    # Generating bounding boxes
    generate_bboxes(xJ=xJ, St=St, descendents_and_self=descendents_and_self,
                    labels=labels, ontology=ontology, outpath=output_prefix)

    # Create marching cube surfaces

    oJ = [x[0] for x in xJ]
    dJ = [x[1] - x[0] for x in xJ]
    origin = get_origin(xJ)

    print("Generating descendent meshes")
    # Generate descendent meshes
    for l in ontology:
        print(f'starting {l}')
        # skip background
        if l == 0:
            print('skipping 0')
            continue

        Sl = St == l
        count0 = np.sum(Sl)
        # do marching cubes
        print('adding ',end='')
        for o in descendents_and_self[l]:
            print(f'{o},',end='')
            Sl = np.logical_or(Sl,St==o)
        count1 = np.sum(Sl)
        if count0 != count1:
            print(f'Structure {l} shows differences')
        if count1 == 0:
            print(f'no voxels for structure {l}')
            continue

        verts,faces,normals,values = marching_cubes(Sl[0]*1.0,level=0.5,spacing=dJ)
        # deal with the offsets
        verts += oJ

        generate_npz_mesh_file(l=l, verts=verts, faces=faces, normals=normals, 
                               values=values, ontology=ontology, outpath=output_prefix,
                               origin=origin)
        generate_3d_surface_plot(verts=verts, faces=faces, ontology=ontology, label=l, outpath=output_prefix)
    print(f"Registration process complete.  Output folder {output_prefix}")

if __name__ == "__main__":
    register()
