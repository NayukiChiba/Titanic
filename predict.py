"""
预测模块

功能：
1. 加载最佳模型
2. 对测试集进行预测
3. 生成 Kaggle 提交文件

使用方法：
    python predict.py

前置条件：
    1. 先运行 python train.py 训练模型
    2. 再运行 python evaluate.py 评估并选择最佳模型
"""

import os
from typing import Any

import numpy as np
import pandas as pd

import config

# =============================================================================
# 模型加载
# =============================================================================


def loadModel(filepath: str) -> Any:
    """
    从文件加载模型

    Args:
        filepath: 模型文件路径

    Returns:
        加载的模型
    """
    import joblib

    model = joblib.load(filepath)
    return model


def loadBestModel() -> tuple[Any, str]:
    """
    加载最佳模型

    Returns:
        (模型实例, 模型名称)
    """
    import joblib

    bestInfoPath = os.path.join(config.MODEL_DIR, "best_model_info.pkl")
    if not os.path.exists(bestInfoPath):
        raise FileNotFoundError("最佳模型信息不存在，请先运行 evaluate.py")

    bestInfo = joblib.load(bestInfoPath)
    model = loadModel(bestInfo["path"])
    return model, bestInfo["name"]


# =============================================================================
# 数据加载
# =============================================================================


def loadTestData() -> tuple[pd.DataFrame, pd.Series]:
    """
    加载测试集数据

    Returns:
        (测试特征 DataFrame, PassengerId Series)
    """
    testFilepath = os.path.join(config.DATASETS_DIR, config.TEST_PROCESSED_FILE)
    if not os.path.exists(testFilepath):
        raise FileNotFoundError(
            "测试集预处理文件不存在，请先运行 featureEngineering.py"
        )

    testDf = pd.read_csv(testFilepath)
    passengerIds = testDf["PassengerId"]
    XTest = testDf.drop(columns=["PassengerId"])
    return XTest, passengerIds


def loadTrainFeatureColumns() -> list[str]:
    """
    加载训练集特征列名（用于对齐测试集）

    Returns:
        特征列名列表
    """
    import joblib

    valDataPath = os.path.join(config.MODEL_DIR, "val_data.pkl")
    if not os.path.exists(valDataPath):
        raise FileNotFoundError("验证集数据不存在，请先运行 train.py")

    valData = joblib.load(valDataPath)
    return list(valData["X"].columns)


# =============================================================================
# 预测与提交
# =============================================================================


def predict(model: Any, XTest: pd.DataFrame) -> np.ndarray:
    """
    使用模型进行预测

    Args:
        model: 训练后的模型
        XTest: 测试特征

    Returns:
        预测结果数组
    """
    predictions = model.predict(XTest)
    return predictions


def createSubmission(
    passengerIds: pd.Series, predictions: np.ndarray, outputPath: str
) -> None:
    """
    创建 Kaggle 提交文件

    Args:
        passengerIds: 乘客 ID
        predictions: 预测结果
        outputPath: 输出文件路径

    提交格式：
        PassengerId,Survived
        892,0
        893,1
        ...
    """
    submissionDf = pd.DataFrame({"PassengerId": passengerIds, "Survived": predictions})
    submissionDf.to_csv(outputPath, index=False)


# =============================================================================
# 主函数
# =============================================================================


def main():
    """
    主函数：执行预测流程

    流程：
        1. 加载最佳模型
        2. 加载测试集
        3. 对齐特征列
        4. 生成预测
        5. 创建提交文件
    """
    print("=" * 60)
    print("🔮 Titanic 预测")
    print("=" * 60)

    # 1. 加载最佳模型
    print("\n🔧 加载最佳模型...")
    try:
        model, modelName = loadBestModel()
        print(f"   使用模型: {modelName}")
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        print("   请先运行: python train.py && python evaluate.py")
        return

    # 2. 加载测试集
    print("\n📂 加载测试集...")
    try:
        XTest, passengerIds = loadTestData()
        print(f"   测试集: {XTest.shape}")
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        return

    # 3. 对齐特征列（确保顺序和列名一致）
    print("\n🔄 对齐特征列...")
    try:
        trainColumns = loadTrainFeatureColumns()
        XTest = XTest.reindex(columns=trainColumns, fill_value=0)
        print(f"   特征数: {len(trainColumns)}")
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        return

    # 4. 生成预测
    print("\n🔮 生成预测...")
    predictions = predict(model, XTest)
    print(f"   预测样本数: {len(predictions)}")
    print(
        f"   存活预测: {sum(predictions)} ({sum(predictions) / len(predictions) * 100:.1f}%)"
    )

    # 5. 创建提交文件
    print("\n📝 生成提交文件...")
    os.makedirs(config.PREDICT_DIR, exist_ok=True)
    submissionPath = os.path.join(config.PREDICT_DIR, config.SUBMISSION_FILE)
    createSubmission(passengerIds, predictions, submissionPath)
    print(f"   已保存: {submissionPath}")

    print("\n" + "=" * 60)
    print("✅ 预测完成！")
    print(f"   提交文件: {submissionPath}")
    print("=" * 60)


if __name__ == "__main__":
    main()
