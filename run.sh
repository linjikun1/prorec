#!/bin/bash

# ================= 配置区域 =================
# 设置目标优化等级后缀 (脚本会将配置文件中的 OLD_TAG 替换为 NEW_TAG)
OLD_TAG="x64_O3"
NEW_TAG="x64_O1"

# 显卡设置
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4

# 环境变量 (解决 OOM 碎片化)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

# Python 解释器路径
PYTHON_EXEC="/data1/linjk/envs/torchenv/bin/python"
TORCHRUN_EXEC="/data1/linjk/envs/torchenv/bin/torchrun"
ACCELERATE_EXEC="/data1/linjk/envs/torchenv/bin/accelerate"
# ===========================================

# 切换到 src 目录
cd src || { echo "Directory src not found!"; exit 1; }
CONFIG_DIR="scripts/configs"
DATA_SCRIPT_DIR="../data"

echo "=========================================================="
echo "准备将配置文件和数据脚本中的 $OLD_TAG 修改为 $NEW_TAG ..."
echo "=========================================================="

# 1. 批量修改 yaml 配置文件 (原地修改)
sed -i "s/$OLD_TAG/$NEW_TAG/g" $CONFIG_DIR/*.yaml

# 2. 批量修改 data 目录下的 python 脚本 (原地修改，确保数据生成路径正确)
sed -i "s/$OLD_TAG/$NEW_TAG/g" $DATA_SCRIPT_DIR/*.py

echo "配置文件修改完成。开始执行 Pipeline..."

# 定义错误处理函数
check_status() {
    if [ $? -ne 0 ]; then
        echo "Error: Step $1 failed. Exiting."
        exit 1
    fi
}

# ================= 流程开始 =================

echo "----------------------------------------------------------"
echo "[Step 0] Preparing CASP Data"
echo "Running: python ../data/run_casp_data_cg.py"
echo "----------------------------------------------------------"
$PYTHON_EXEC ../data/run_casp_data_cg.py
check_status 0

echo "----------------------------------------------------------"
echo "[Step 1] CASP Source Pre-alignment"
echo "----------------------------------------------------------"
$TORCHRUN_EXEC --nproc_per_node=$NPROC_PER_NODE run_casp.py $CONFIG_DIR/train_casp_moco.yaml
check_status 1

echo "----------------------------------------------------------"
echo "[Step 2] CASP Signature Pre-alignment"
echo "----------------------------------------------------------"
$TORCHRUN_EXEC --nproc_per_node=$NPROC_PER_NODE run_casp_signature.py $CONFIG_DIR/train_casp_moco_sig.yaml
check_status 2

echo "----------------------------------------------------------"
echo "[Step 3] Prober Training"
echo "----------------------------------------------------------"
$TORCHRUN_EXEC --nproc_per_node=$NPROC_PER_NODE run_prober.py $CONFIG_DIR/train_prober.yaml
check_status 3

echo "----------------------------------------------------------"
echo "[Step 4a] Preparing Probing Data (Signature)"
echo "Running: python ../data/probed_data_cg.py"
echo "----------------------------------------------------------"
$PYTHON_EXEC ../data/probed_data_cg.py
check_status 4a

echo "----------------------------------------------------------"
echo "[Step 4b] Probing Function Signature"
echo "----------------------------------------------------------"
$ACCELERATE_EXEC launch --num_processes=$NPROC_PER_NODE big_model_quantized_probing.py $CONFIG_DIR/probe_quantized_codellama-34b-4bit-unfreeze.yaml
check_status 4b

echo "----------------------------------------------------------"
echo "[Step 5] Score and Filter Signatures"
echo "----------------------------------------------------------"
$PYTHON_EXEC score_and_filter_signature.py $CONFIG_DIR/filter_sig.yaml
check_status 5

echo "----------------------------------------------------------"
echo "[Step 6a] Preparing Probing Data (Body/Continue)"
echo "Running: python ../data/probed_continue_data_cg.py"
echo "----------------------------------------------------------"
$PYTHON_EXEC ../data/probed_continue_data_cg.py
check_status 6a

echo "----------------------------------------------------------"
echo "[Step 6b] Probing Continue (Body Generation)"
echo "----------------------------------------------------------"
$ACCELERATE_EXEC launch --num_processes=$NPROC_PER_NODE big_model_quantized_probing_continue.py $CONFIG_DIR/probe_continue.yaml
check_status 6b

echo "=========================================================="
echo "All steps finished successfully for $NEW_TAG!"
echo "=========================================================="
