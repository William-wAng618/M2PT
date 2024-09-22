#!/bin/bash
conda init
source /opt/conda/etc/profile.d/conda.sh
conda activate llava
pip install openpyxl
cd /wudi/llava0530/LLaVA-main
python twdebug/llm_eval_score.py
