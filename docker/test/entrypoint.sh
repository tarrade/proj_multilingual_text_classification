#!/bin/bash
printenv
conda info --envs
conda info
. "${DL_ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate env_multilingual_class
conda info -e
exec "$@"