#!/bin/python

import os
import argparse
from glob import glob
import nibabel as nib

parser = argparse.ArgumentParser(
                description='Create subject-specific grey matter mask',
                epilog=('Example: python make_gm_mask.py --sub=FLT02 '
                        ' --space=MNI152NLin2009cAsym --fwhm=1.5 '
                        ' --bidsroot=/PATH/TO/BIDS/DIR/ ' 
                        ' --fmriprep_dir=/PATH/TO/FMRIPREP/DIR/'))

parser.add_argument("--sub", help="participant id", 
                    type=str)
parser.add_argument("--space", help="space label", 
                    type=str)
parser.add_argument("--fwhm", help="spatial smoothing full-width half-max", 
                    type=float)
parser.add_argument("--bidsroot", 
                    help="top-level directory of the BIDS dataset", 
                    type=str)
parser.add_argument("--fmriprep_dir", 
                    help="directory of the fMRIprep preprocessed dataset", 
                    type=str)

args = parser.parse_args()

subject_id = args.sub
space_label=args.space
fwhm = args.fwhm
bidsroot = args.bidsroot
fmriprep_dir = args.fmriprep_dir

''' define other inputs '''
nilearn_dir = os.path.join(bidsroot, 'derivatives', 'nilearn')

# masking function
def generate_mask(subject_id, bidsroot, func_example_fpath, sub_mask_dir, space_label):
    from nilearn.image import concat_imgs, mean_img, binarize_img, resample_to_img

    # create binarized gray matter mask
    cau_list = sorted(glob(sub_mask_dir + f'/sub-{subject_id}_space-{space_label}_*CAU*.nii.gz'))
    put_list = sorted(glob(sub_mask_dir + f'/sub-{subject_id}_space-{space_label}_*PUT*.nii.gz'))
    nac_list = sorted(glob(sub_mask_dir + f'/sub-{subject_id}_space-{space_label}_*NAc*.nii.gz'))
    gp_list  = sorted(glob(sub_mask_dir + f'/sub-{subject_id}_space-{space_label}_*GP*.nii.gz'))
    masks_list = cau_list + put_list + nac_list + gp_list
        
    # get the mean image and binarize all voxels > 0
    concatenated_img = concat_imgs(masks_list)
    mean_concat_img = mean_img(concatenated_img)
    
    str_bin_img = binarize_img(mean_concat_img, threshold=0)
    mask_func_img = resample_to_img(str_bin_img, func_example_fpath, interpolation='nearest')
    
    labelname = 'striatum'
    out_basename = f'sub-{subject_id}_space-{space_label}_mask-{labelname}.nii.gz'
    out_fpath = os.path.join(sub_mask_dir, out_basename)
    nib.save(mask_func_img, out_fpath)
    print('saved mask image to', out_fpath)
    
    return out_fpath

''' run function '''
fmriprep_func_dir = os.path.join(bidsroot, 'derivatives', 
                                 'denoised_fmriprep-22.1.1',
                                 f'sub-{subject_id}', 'func')
 
func_example_fpath = sorted(glob(fmriprep_func_dir+f'/*{space_label}*bold.nii.gz'))[0]

sub_mask_dir = os.path.join(nilearn_dir, 
                            'masks', 
                            f'sub-{subject_id}', 
                            f'space-{space_label}', 
                            'masks-tian-S3')   
if not os.path.exists(sub_mask_dir):
    os.makedirs(sub_mask_dir)

mask_fpath = generate_mask(subject_id, bidsroot, func_example_fpath, 
                           sub_mask_dir, space_label)

