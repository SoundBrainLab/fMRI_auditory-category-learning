#!/bin/bash

sing_dir=/ix1/bchandrasekaran/krs228/software/singularity_images/

module load singularity

singularity build $sing_dir/fmriprep-25.2.5.simg docker://nipreps/fmriprep:25.2.5