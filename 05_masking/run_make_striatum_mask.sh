#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH -c 2
bidsroot=/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/

for sub in FLT02 FLT03 FLT04 FLT05 FLT06 FLT07 FLT08 FLT09 FLT10 FLT11 FLT12 FLT13 FLT14 FLT15 FLT17 FLT18 FLT19 FLT20 FLT21 FLT22 FLT23 FLT24 FLT25 FLT26 FLT28 FLT30; do
  echo $sub
  python make_striatum_mask.py --sub=$sub \
    --space=MNI152NLin2009cAsym \
    --fwhm=0.00 \
    --bidsroot=$bidsroot \
    --fmriprep_dir=$bidsroot/derivatives/denoised_fmriprep-22.1.1/
done

