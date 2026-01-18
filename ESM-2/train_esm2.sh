#!/bin/bash
export LOG_DIR="/data/joshi/utils/ESM2_revised/logs"
export TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"
mkdir -p ${LOG_DIR}

echo "Training started: $(date)" | tee ${LOG_FILE}
echo "Log file: ${LOG_FILE}" | tee -a ${LOG_FILE}

torchrun --nproc_per_node=8 /home/joshi/experiments/ESM2_revised/train.py >> ${LOG_FILE} 2>&1

echo "Training ended: $(date)" | tee -a ${LOG_FILE}
