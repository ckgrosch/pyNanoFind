import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb

def binarize_map(y_predOG,threshold):
    y_pred = y_predOG.copy()
    if y_pred.flags.owndata != True:
        raise RuntimeError('Data copy did not work - stopped before data overwrite occurred')
    for y in y_pred:
        y[y > threshold] = 1
        y[y <= threshold] = 0
    return y_pred

def label_stack(bin_stack):
    """Input is stack of binarized predicted segmenation maps. Output is stack of labeled images"""
    label_images = []
    for img in bin_stack:
        lbl = label(img)
        label_images.append(lbl)
    label_images = np.array(label_images)
    return label_images

def regionprop_stack(label_images,min_area = 500, max_area = 65500):
    """Input is region labelled images and min_area which defines the minimum number of pixels
    a region must contain to be saved. Output contains the area and bounding boox coordinates"""
    region_list = []
    for img in label_images:
        img_data = []
        for region in regionprops(img):
            if region.area >= min_area:
                img_data.append([region.area,region.bbox,region.image])
        region_list.append(img_data)
    return region_list

def particle_stack(images,region_list):
    """Input is TEM image stack and the output of regionprop_stack. Output is a stack of
    images of segmented out particles"""
    particles = []
    particle_idx = []
    for idx,img in enumerate(images):
        for idx2,region in enumerate(region_list[idx]):
            minr, minc, maxr, maxc = region[1]
            img_section = img[minr:maxr,minc:maxc]
            img_section = img_section.reshape((img_section.shape[0],img_section.shape[1]))*region[2]
            particles.append(img_section)
            particle_idx.append((idx,idx2))
    return(particles,particle_idx)

def segment_particles(bin_stack,images,min_area=500, max_area = 65500, save = False):
    """Inputs: binarized segmentation maps, corresponding TEM images, minimum pixel area of regions of interest"""
    label_images = label_stack(bin_stack)
    region_list = regionprop_stack(label_images,min_area,max_area)
    particles, particle_idx = particle_stack(images,region_list)
    if save == True:
        dir_str = input('Enter directory to save to: ')
        fname = input('Enter filename without extension: ')
        np.save(dir_str+'/'+fname+'_particleImgs.npy',np.array(particles))
        np.save(dir_str+'/'+fname+'_particleIdxs.npy',np.array(particle_idx))
    return(particles, particle_idx)

def pad_stack(particles,pad_size = 256):
    center = pad_size // 2
    pad_stack = []
    for idx,img in enumerate(particles):
        if img.shape[0] > pad_size or img.shape[1] > pad_size:
            print('skipping image of index: ',idx)
        else:
            pad = np.zeros((pad_size,pad_size))
            x_center = img.shape[0]//2
            y_center = imgs.shape[1]//2
            pad[center-x_center:center+x_center,center-y_center:center+y_center] = img
            pad_stack.append(pad)
    return pad_stack
