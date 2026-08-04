#!/bin/bash
# Condor entrypoint for full pretraining on c02 at 4 GPUs (50% of the box).
# Condor sets CUDA_VISIBLE_DEVICES to the 4 GPUs it granted; hand those to the
# bare-metal driver so torchrun uses exactly them. COMBO/PAIRS come from the
# submit environment.
export GPUS="${CUDA_VISIBLE_DEVICES}"
cd /home/rianatri/Multimodal-PyHealth-ssl
exec bash scripts/run_full_pretrain_local.sh
