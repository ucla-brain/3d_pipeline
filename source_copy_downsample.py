#!/usr/bin/env python
# coding: utf-8

# In[1]:


# this notebook will downsample lightsheet data
# from here
#  /panfs/dong/3D_stitched_LS/20220725_SW220510_02_LS_6x_1000z
#  note we will  now load voxel size from the data itself
# and we load the extent as well
#


# in v01 I look at standard deviation too
# 
# I'd also like to find the minimum value in the dataset other than 0

# NOTE
# if any part of a chunk is accessed, the whole chunk is read.
# 
# I should be able to speed things up by reading 64 slices at a time (potentially).
# 
# "
# Typical chunk sizes are 128x128x64 or 256x256x16. The optimal chunk size is determined by the geometry of the image and it is not easy to specify rules for reproducing exactly the chunk sizes that Imaris will write into the hdf-file.
# "

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
import os
from glob import glob
from os.path import join as pathjoin
import h5py
import time

import imp
import sys
sys.path.append('..')
import donglab_workflows as dw
imp.reload(dw)
import PIL.Image as Image
Image.MAX_IMAGE_PIXELS = None
import tifffile as tf # for 16 bit tiff
import h5py


# In[3]:


# todo, move into dongloab workflows and use for dragonfly as well


# # downsample lightsheet

# The mit data is stored in tif stacks
# the size is
# Image resolution is 1.8 µm x 1.8 µm x 2.0 µm (xyz), and the stack can be found at our network storage space at BMAP.
# 
# 

# In[4]:


# input path, can be a directory or a filename
input_path = '/home/dtward/bmaproot/panfs/dong/3D_stitched_LS/20220725_SW220510_02_LS/SW220510_02_LS_6x_1000z.ims'
image_type = 'ims' # can be ims or Tif

output_filename = None # generate automatically if None

# we need a temporary output directory for intermediate results (each slice)
outdir = '/home/dtward/bmaproot/nafs/dtward/dong/donglab_resample_lightsheet_good_2022_09_06_tmp'

# res is the desired voxel size
dI = None # if none will load from data
res = 50.0 # perhaps we could use the 25 micron atlas (this can be any float)
channel = 0
dataset_string = f'DataSet/ResolutionLevel 0/TimePoint 0/Channel {channel}/Data' # not used for Tifs

# power to reduce dynamic range
power = np.ones(1,dtype=np.float32)*0.125

# blocksize and chunksize for looking for areas with no data and loading quickly
blocksize = 64 # 
chunksize = 32 # 


# In[5]:


if dI is None and image_type == 'ims':
    f = h5py.File(input_path,'r')
    dI = dw.imaris_get_pixel_size(f)    
    xI = dw.imaris_get_x(f)
    f.close()
    
if output_filename is None:
    output_filename = os.path.splitext(os.path.split(input_path)[-1])[0] + '_ch_' + str(channel) + '_pow_' + str(power) + '_down.npz'
#output_filename = 'SYTO16_488_086780_109130_down.npz'    

print(f'Input path is {input_path}')
print(f'Output filename is {output_filename}')
print(f'Resolution is {dI}')
print(f'Desired resolution is {res}')
print(f'Dataset string is {dataset_string}')
print(f'tmp output dir is {outdir}')

# temporary output dir
os.makedirs(outdir,exist_ok=True)


# In[6]:


# I want 50 micron
down = np.floor(res/dI).astype(int)
print(f'Downsampling factors are {down}')
print(f'Downsampled res {dI*down}')


# In[7]:


# build a tif class with similar interface
class TifStack:
    '''We need a tif stack with an interface that will load a slice one at a time
    We assume each tif has the same size
    We assume 16
    '''
    def __init__(self,input_directory,pattern='*.tif'):
        self.input_directory = input_directory
        self.pattern = pattern
        self.files = glob(pathjoin(input_directory,pattern))
        self.files.sort()
        test = Image.open(self.files[0])
        self.nxy = test.size
        test.close()
        self.nz = len(self.files)
        self.shape = (self.nz,self.nxy[1],self.nxy[0]) # note, it is xy not rowcol
    def __getitem__(self,i):
        return tf.imread(self.files[i])/(2**16-1)
    def __len__(self):
        return len(self.files)
    def close(self):
        pass # nothing necessary
    


# In[8]:


# load the data

if image_type == 'tif':
    data = TifStack(input_directory)
    
elif image_type == 'ims':
    data_ = h5py.File(input_path,mode='r')
    data = data_[dataset_string]
    


# In[10]:


print(f'Dataset shape {data.shape}')


# In[11]:


nI = np.array(data.shape)
#xI = [np.arange(n)*d - (n-1)/2.0*d for n,d in zip(nI,dI)] # already computed above
# NOTE: the imaging data is smaller than the saved data because the saved data is a multiple of 64
nIreal = np.array([len(x) for x in xI])


# In[12]:


xId = [dw.downsample(x,[d]) for x,d in zip(xI,down)]
dId = [x[1]-x[0] for x in xId]


# In[13]:


dId


# In[ ]:


# okay now I have to iterate over the dataset
# note this is currently not doing wieghts
# we need to save intermediate outputs (each slice) in case of errors
fig,ax = plt.subplots(2,2)
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
    outname = os.path.join(outdir,f'{i:06d}_s.npy')
    


    if os.path.exists(outname):
        # what happens if it fails in the middle of a chunk?
        sd = np.load(outname)
        s2d = np.load(outname.replace('_s','_s2'))
        wd = np.load(outname.replace('_s','_w'))
    else:
        # load a whole chunk
        if not i%chunksize:
            data_chunk = data[i:i+chunksize]
        # use this for weights
        #s_all = data[i,:,:]
        # it's possible that this will fail if I haven't defined data_chunk yet
        try:
            s_all = data_chunk[i%chunksize,:,:]
        except:
            # we need to load, not starting at i
            # but at the beginning of the chunk
            data_chunk = data[i//chunksize:i//chunksize+chunksize]
            s_all = data_chunk[i%chunksize,:,:]
        s = s_all[:nIreal[1]+1,:nIreal[2]+1]**power # test reduce dynamic range before downsampling with this power
        s2 = s**2
        #w = (s>0).astype(float)
        # this is not a good way to get weights, 
        # we need to look for a 64x64 block of all zeros
        
        s_all_block = s_all.reshape(s_all.shape[0]//blocksize,blocksize,s_all.shape[1]//blocksize,blocksize)
        tmp = np.logical_not(np.all(s_all_block==0,axis=(1,3))).astype(np.uint8)
        s_all_w = np.ones_like(s_all_block)
        s_all_w *= tmp[:,None,:,None]
        s_all_w = s_all_w.reshape(s_all.shape)
        w = s_all_w[:nIreal[1]+1,:nIreal[2]+1].astype(power.dtype)

        
        sd = dw.downsample((s*w),down[1:])
        s2d = dw.downsample((s2*w),down[1:])
        wd = dw.downsample(w,down[1:])
        sd /= wd
        sd[np.isnan(sd)] = 0.0
        s2d /= wd
        s2d[np.isnan(s2d)] = 0.0
        
        np.save(outname,sd)
        np.save(outname.replace('_s','_w'),wd)
        np.save(outname.replace('_s','_s2'),s2d)
    
    ax[0].cla()
    wd0 = wd>0.0
    if np.any(wd0):
        vmin = np.min(sd[wd0])
        vmax = np.max(sd[wd0])
    else:
        vmin = None
        vmax = None
    ax[0].cla()
    ax[0].imshow(sd,vmin=vmin,vmax=vmax)
    ax[2].cla()
    ax[2].imshow(wd,vmin=0,vmax=1)
    working.append(sd)
    working2.append(s2d)
    workingw.append(wd)
    
    if len(working) == down[0]:
        workingw_stack = np.stack(workingw)
        out = dw.downsample(np.stack(working)*workingw_stack,[down[0],1,1])
        out2 = dw.downsample(np.stack(working2)*workingw_stack,[down[0],1,1])
        outw = dw.downsample(workingw_stack,[down[0],1,1])        
        out /= outw
        out[np.isnan(out)] = 0.0
        out2 /= outw
        out2[np.isnan(out2)] = 0.0
        outstd = out2 - out**2
        outstd[outstd<0]=0
        outstd = np.sqrt(outstd)
        wd0 = (wd>0.0)[None]
        if np.any(wd0):
            outshow = (out[0] - np.min(out[wd0]))/(np.quantile(out[wd0],0.99) - np.min(out[wd0]))
            outshowstd = (outstd[0] - np.min(outstd[wd0]))/(np.quantile(outstd[wd0],0.99) - np.min(outstd[wd0]))
        else:
            outshow = (out[0] - np.min(out))/(np.quantile(out,0.99) - np.min(out))
            outshowstd = (outstd[0] - np.min(outstd))/(np.quantile(outstd,0.99) - np.min(outstd))
        ax[1].cla()
        ax[1].imshow(np.stack((outshow,outshowstd,outshow),-1))
        ax[3].cla()
        ax[3].imshow(outw[0],vmin=0,vmax=1)
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


# In[ ]:


np.savez(output_filename,I=Id,I2=np.concatenate(output2),xI=np.array(xId,dtype='object'),w=wd) # note specify object to avoid "ragged" warning


# In[ ]:


fig,ax = dw.draw_slices(Id,xId)
fig.suptitle(output_filename)
fig.savefig(output_filename.replace('npz','jpg'))


# In[ ]:


out.shape


# In[ ]:




