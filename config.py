"""
项目配置模块

功能：
1. 统一管理项目路径
2. 提供全局配置常量

使用方法：
    import config
    trainPath = os.path.join(config.DATASETS_DIR, "train.csv")
"""

import os

# =============================================================================
# 路径配置
# =============================================================================

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据集目录
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")

# 输出目录
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(OUTPUTS_DIR, "model")
PREDICT_DIR = os.path.join(OUTPUTS_DIR, "predict")

# =============================================================================
# 文件名配置
# =============================================================================

# 原始数据文件
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

# 预处理后的数据文件
TRAIN_PROCESSED_FILE = "train_processed.csv"
TEST_PROCESSED_FILE = "test_processed.csv"

# 提交文件
SUBMISSION_FILE = "gender_submission.csv"
