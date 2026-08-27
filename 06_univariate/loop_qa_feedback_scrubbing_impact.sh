#!/bin/bash

# the actual analyzed (non-Mandarin) sample the manuscript's feedback results
# are based on -- n=12, confirmed against group_level_all_ROI.ipynb's own
# cached sub_list_nman (REVISION_PLAN.md, Workstream 0). Not the full
# 26-subject processing list, since this diagnostic is specifically about
# explaining the analyzed sample's striatal feedback results.
for subid in FLT02 FLT04 FLT06 FLT09 FLT11 FLT12 FLT13 FLT14 FLT20 FLT25 FLT28 FLT30; do
  echo $subid
  sbatch run_qa_feedback_scrubbing_impact.sh $subid
done
