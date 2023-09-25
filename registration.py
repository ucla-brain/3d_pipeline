#!/usr/bin/env python
# coding: utf-8

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
import emlddmm

FIG_OPTIONS = {'dpi': 300, 'format': 'jpg'}

def get_target(data):
    target = data['I'][None]
    target = target.astype(np.float32)
    target /= np.mean(np.abs(target))
    return target

def get_atlas(atlases):
    I = []
    for atlas_name in atlases:
        xI, I_, _, _ = emlddmm.read_data(atlas_name)
        I_ = I_.astype(np.float32)
        I_ /= np.mean(np.abs(I_))
        I.append(I_)

    I = np.concatenate(I)
    return I, xI

def get_seg_atlas(seg):
    xS, S, _, _ = emlddmm.read_data(seg)
    return S, xS

def get_RGB(seg):
    labels,inds = np.unique(seg,return_inverse=True)

    colors = np.random.rand(len(labels),3)
    colors[0] = 0.0

    return colors[inds].reshape(seg.shape[1],seg.shape[2],seg.shape[3],3).transpose(-1,0,1,2)

def get_voxel_spacing(data):
    return data['xI']

def get_w(data, image):
    if 'w' in data:
        return data['w']
    elif 'W' in data:
        return data['W']
    else:
        return (image[0]>0).astype(float)

def get_origin(vox_data):
    return np.array([vox_data[0][0], vox_data[1][0], vox_data[2][0]])

def save_metadata(target, segmentation, outpath, affine):
    with open(os.path.join(outpath, "registration_metadata.txt"), 'w') as f:
        metadata = f"Date: {str(datetime.now())}\n"
        metadata += f"Downsampled File: {target}\n"
        metadata += f"Segmentation File: {segmentation}\n"
        metadata += f"Initial Affine: {affine}\n"
        f.write(metadata)

def save_figure(figure, name, outpath="", title=""):
    print(f"Generating {name} figure")
    figure.suptitle(title)
    figure.savefig(os.path.join(outpath, name + ".jpg"), **FIG_OPTIONS)
    plt.close()

def generate_atlas_space_figure(I, S, RGB, Jt, outdir):
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
    # background
    processed_target = target - np.quantile(target[W[None]>0.9],0.1)
    processed_target[processed_target<0] = 0
    # adjust dynamic range
    processed_target = processed_target**0.25
    # adjust mean value
    processed_target /= np.mean(np.abs(processed_target))

    return processed_target

def preprocess_atlas(atlas, dI,  xI0):
    # since downsample by 4, pad with 4x4x4
    npad = 4
    I = np.pad(atlas, ((0,0),(npad,npad),(npad,npad),(npad,npad)) )
    for _ in range(npad):
        xI = [np.concatenate((x[0][None] - d, x, x[-1][None] + d)) for d, x in zip(dI,xI0)]

    # adjust nissl image dynamic range
    I[0] = I[0]**0.5
    I[0] /= np.mean(np.abs(I[0]))
    I[1] /= np.mean(np.abs(I[1]))

    return I, xI

def get_initial_affine(A0, XJ):
    if A0 == None:
        A0 = np.eye(4)
        A0[:3,-1] = np.mean(XJ,axis=(-1,-2,-3))
    else:
        A0 = np.fromstring(A0,sep=',').reshape(4,4)

    return A0

def run_registration(J, xJ, I, xI, W, A0, dI, device, resolution):
    config0 = {
        'device':device,
        'n_iter':200, 'downI':[8, 8, 8], 'downJ':[8, 8, 8],
        'priors':[0.9,0.05,0.05],'update_priors':False,
        'update_muA':0,'muA':[np.quantile(J,0.99)],
        'update_muB':0,'muB':[0.0],
        'update_sigmaM':0,'update_sigmaA':0,'update_sigmaB':0,
        'sigmaM':0.25,'sigmaB':0.5,'sigmaA':1.25,
        'order':1,'n_draw':50,'n_estep':3,'slice_matching':0,'v_start':1000,
        'eA':5e4,'A':A0,'full_outputs':True,
    }

    if resolution == 50:
        config0['downI'] = [4, 4, 4]
        config0['downJ'] = [4, 4, 4]

    # update my sigmas (august 22)
    config0['sigmaM'] = 1.0
    config0['sigmaB'] = 2.0
    config0['sigmaA'] = 5.0
    I_ = np.stack((I[0]-np.mean(I[0]),I[1]-np.mean(I[1]),
                (I[0]-np.mean(I[0]))**2,
                (I[0]-np.mean(I[0]))*(I[1]-np.mean(I[1])),
                (I[1]-np.mean(I[0]))**2,))

    out = emlddmm.emlddmm(xI=xI,I=I_,xJ=xJ,J=J, W0=W, **config0)

    # second run, with deformation
    config1 = dict(config0)
    config1['A'] = out['A']
    config1['eA'] = config0['eA']*0.1
    config1['a'] = 1000.0
    config1['sigmaR'] = 5e4 # 1e4 gave really good results, but try 2e4, also good, I showed this in my slides
    config1['n_iter']= 2000
    config1['v_start'] = 0
    config1['ev'] = 1e-2
    config1['ev'] = 2e-3 # reduce since I decreased sigma
    config1['v_res_factor'] = config1['a']/dI[0]/4 # what is the resolution of v, as a multiple of that in I
    config1['local_contrast'] = [16,16,16]

    if resolution == 50:
        config1['v_res_factor'] = config1['a']/dI[0]/2

    I_ = np.stack((I[0]-np.mean(I[0]),I[1]-np.mean(I[1]),
                (I[0]-np.mean(I[0]))**2,
                (I[0]-np.mean(I[0]))*(I[1]-np.mean(I[1])),
                (I[1]-np.mean(I[0]))**2,))


    out1 = emlddmm.emlddmm(xI=xI,I=I_,xJ=xJ,J=J, W0=W, **config1)

    # on the next run we do les downsampling
    config2 = dict(config1)
    config2['A'] = out1['A']
    config2['n_iter']= 1000
    config2['v'] = out1['v']
    config2['downI'] = [4, 4, 4]
    config2['downJ'] = [4, 4, 4]

    if resolution == 50:
        config2['downJ'] = [2, 2, 2]
        config2['downI'] = [2, 2, 2]

    # there seems to be an issue with the initial velocity
    # when I run this twice, I'm reusing it
    # the matching energy here is way way lower, why would that be?
    return emlddmm.emlddmm(xI=xI,I=I_,xJ=xJ,J=J, W0=W, **config2)

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
    parser.add_argument("seg_name",type=pathlib.Path,help="Location of segmentation file")
    parser.add_argument("savename",type=pathlib.Path,help="Name of file once it is saved")
    parser.add_argument("atlas_names",type=pathlib.Path,nargs='+',help="Location of the atlas images")
    parser.add_argument("-on", "--onto_name", default=None, help="Location of atlas ontology")
    parser.add_argument("-d","--device",default='cuda:0',help="Device used for pytorch")
    parser.add_argument("-a","--A0",type=str,default=None,help="Affine transformation (Squeezed to 16x1 + Sep by ',')")
    parser.add_argument("-res","--resolution", type=np.float32, choices=[20.0, 50.0], default=20, help="Resoultion used during downsampling")
    parser.add_argument("-j","--jpg", action="store_true", help="Generate 3d jpegs for each structure")

    args = parser.parse_args()

    target_name   = args.target_name
    output_prefix = args.output_prefix
    atlas_names   = args.atlas_names
    seg_name      = args.seg_name
    savename      = args.savename
    ontology_name = args.onto_name
    resolution    = args.resolution
    jpg           = args.jpg
    device        = args.device
    A0            = args.A0

    # Validations
    assert os.path.isfile(target_name), f"Downsample file was not found: {target_name}"
    assert os.path.isfile(seg_name), f"Atlas segmentation file was not found: {seg_name}"
    assert os.path.isfile(atlas_names[0]), f"Atlas image file was not found: {atlas_names[0]}"
    assert os.path.isfile(atlas_names[1]), f"Atlas image file was not found: {atlas_names[1]}"
    assert os.path.isfile(ontology_name), f"Atlas ontology file was not found: {ontology_name}"
    if not os.path.exists(pathlib.Path(output_prefix)):
        os.mkdir(output_prefix)
    assert os.path.isdir(output_prefix), f"Output directory does not exist: {output_prefix}"

    # Record metadata
    print("Recording metadata")
    save_metadata(target_name, seg_name, output_prefix, A0)

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
    J0 = np.copy(J)
    # get W variable
    W = get_w(target_data, image=J)

    # Save downsample image figure
    fig, _ = emlddmm.draw(J,xJ,vmin=np.min(J[W[None]>0.9]))
    save_figure(fig, "downsample", title='Downsampled lightsheet data', outpath=output_prefix)

    print("Loading atlas")
    I, xI = get_atlas(atlas_names)

    dI = [x[1] - x[0] for x in xI]
    xI0 = [ np.copy(x)for x in xI]
    I0 = np.copy(I)

    print("Loading and transforming segmentation atlas")
    # next is to transforom the high resolution data
    S, xS = get_seg_atlas(seg_name)
    # we want to visuze the above with S
    RGB = get_RGB(seg=S)

    ##########################
    ## PREPROCESSING

    print("Preprocessing target and atlas")
    # Target preprocessing
    J = preprocess_target(target=J0, W=W)

    # Save processed target figure
    fig, _ = emlddmm.draw(J,xJ,vmin=0)
    save_figure(fig, "processed", title="Preprocessed lightsheet data", outpath=output_prefix)

    # Atlas preprocessing
    I, xI = preprocess_atlas(atlas=I0, dI=dI, xI0=xI0)

    #####################
    ## REGISTRATION

    print("Loading initial affine")
    XJ = np.meshgrid(*xJ,indexing='ij')
    A0 = get_initial_affine(A0=A0, XJ=XJ)

    # Visualize initial transformation
    tform = emlddmm.Transform(A0,direction='b')
    AI = emlddmm.apply_transform_float(xI,I,tform.apply(XJ))
    fig, _ = emlddmm.draw(np.concatenate((AI[:2],J)),xJ,vmin=0)
    save_figure(fig, "initial_atlas_space", title="Initial Affine Transformation", outpath=output_prefix)

    print("Registering")
    # now we want to register
    out2 = run_registration(J=J, xJ=xJ, I=I, xI=xI, W=W, A0=A0, dI=dI,
                            device=device, resolution=resolution)

    # Saving registration output
    np.save(os.path.join(output_prefix, savename), np.array([out2],dtype=object))

    # Saving figures
    save_figure(out2['figI'], name='transformed.jpg', outpath=output_prefix)
    save_figure(out2['figfI'], name='contrast.jpg', outpath=output_prefix)
    save_figure(out2['figErr'], name='err.jpg', outpath=output_prefix)
    generate_error_figure(out2['figErr'], outdir=output_prefix)

    ##############################
    ## VISUALIZATIONS

    print("Generating registration and transformation visualizations")
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