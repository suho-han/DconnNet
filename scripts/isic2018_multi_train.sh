#!/bin/bash

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

# This sweep is the CHASE batch entrypoint.
# Default fold scope is 1-fold unless overridden.
# Output names follow:
#   binary -> output/1fold/binary_<conn_num>_bce/
#   dist   -> output/1fold/<label_mode>_<conn_num>_<dist_aux_loss>/

EPOCHS="${MULTI_TRAIN_EPOCHS:-500}"
FOLDS="${MULTI_TRAIN_FOLDS:-1}"
EXTRA_ARGS=("$@")

 

for conn_num in 8 24; do
    bash scripts/isic2018_train.sh \
        --conn_num "${conn_num}" \
        --label_mode binary \
        --epochs "${EPOCHS}" \
        --folds "${FOLDS}" \
        --device 1 \
        "${EXTRA_ARGS[@]}"
done

for conn_num in 8 24; do
    for dist_aux_loss in gjml_sf_l1 smooth_l1; do
        for label_mode in dist dist_inverted; do
            bash scripts/isic2018_train.sh \
                --conn_num "${conn_num}" \
                --label_mode "${label_mode}" \
                --epochs "${EPOCHS}" \
                --dist_aux_loss "${dist_aux_loss}" \
                --dist_sf_l1_gamma 1.0 \
                --folds "${FOLDS}" \
                --device 1 \
                "${EXTRA_ARGS[@]}"
        done
    done
done