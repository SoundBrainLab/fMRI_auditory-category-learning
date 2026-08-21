#!/bin/bash

#for subpath in /ix1/bchandrasekaran/krs228/data/FLT/data_denoised/sub*/; do 
#  subid=$(basename $subpath)

for subid in FLT02 FLT03 FLT04 FLT05 FLT06 FLT07 FLT08 FLT09 FLT10 FLT11 FLT12 FLT13 FLT14 FLT15 FLT17 FLT18 FLT19 FLT20 FLT21 FLT22 FLT23 FLT24 FLT25 FLT26 FLT28 FLT30; do
  echo $subid
  for atlas_label in tian_S2 subcort_aud carpet_dseg carpet_pfc; do
      echo $atlas_label
      sbatch run_make_atlas_region_masks.sh $subid $atlas_label
    done
done
