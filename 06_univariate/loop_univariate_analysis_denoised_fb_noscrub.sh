#!/bin/bash

# same analyzed (non-Mandarin) n=12 sample as loop_qa_feedback_scrubbing_impact.sh
for subid in FLT02 FLT04 FLT06 FLT09 FLT11 FLT12 FLT13 FLT14 FLT20 FLT25 FLT28 FLT30; do
  echo $subid
  sbatch run_univariate_analysis_denoised_fb_noscrub.sh $subid
done
