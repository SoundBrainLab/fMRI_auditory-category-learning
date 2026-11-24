#!/bin/bash

# the 25 analyzed participants, both groups (FLT07 excluded: QA; there is no FLT16)
for subid in FLT02 FLT03 FLT04 FLT05 FLT06 FLT08 FLT09 FLT10 FLT11 FLT12 FLT13 FLT14 FLT15 FLT17 FLT18 FLT19 FLT20 FLT21 FLT22 FLT23 FLT24 FLT25 FLT26 FLT28 FLT30; do
  echo $subid
  sbatch run_ppi_roi.sh $subid
done
