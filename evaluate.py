"""
模型评估模块

功能：
1. 评估单个模型
2. 交叉验证
3. 多模型对比
4. 选择最佳模型

使用方法：
    python evaluate.py

前置条件：
    先运行 python train.py 生成模型文件
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


def loadAllModels() -> dict[str, Any]:
    """
    加载所有已训练的模型

    Returns:
        模型字典 {模型名称: 模型实例}
    """
    modelNames = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM"]
    models = {}
    for name in modelNames:
        modelPath = os.path.join(config.MODEL_DIR, f"{name}.pkl")
        if os.path.exists(modelPath):
            models[name] = loadModel(modelPath)
            print(f"   已加载: {name}")
    return models


# =============================================================================
# 模型评估
# =============================================================================


def evaluateModel(model: Any, XVal: pd.DataFrame, yVal: pd.Series) -> dict[str, float]:
    """
    评估单个模型

    Args:
        model: 训练后的模型
        XVal: 验证特征
        yVal: 验证目标

    Returns:
        评估指标字典 {指标名: 值}

    评估指标：
        - accuracy: 准确率
        - precision: 精确率
        - recall: 召回率
        - f1: F1 分数
        - auc: ROC-AUC 值
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    yPred = model.predict(XVal)
    yProba = model.predict_proba(XVal)[:, 1]
    metrics = {
        "accuracy": accuracy_score(yVal, yPred),
        "precision": precision_score(yVal, yPred),
        "recall": recall_score(yVal, yPred),
        "f1": f1_score(yVal, yPred),
        "auc": roc_auc_score(yVal, yProba),
    }
    return metrics


def crossValidate(
    model: Any, X: pd.DataFrame, y: pd.Series, cv: int = 5
) -> dict[str, float]:
    """
    交叉验证评估

    Args:
        model: 模型实例（未训练）
        X: 全部特征
        y: 全部目标
        cv: 折数，默认 5

    Returns:
        交叉验证结果 {指标名: 均值}
    """
    from sklearn.model_selection import cross_val_score

    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cvResults = {}
    for score in scoring:
        scores = cross_val_score(model, X, y, cv=cv, scoring=score)
        cvResults[score] = np.mean(scores)
    return cvResults


def evaluateAllModels(
    models: dict[str, Any], XVal: pd.DataFrame, yVal: pd.Series
) -> pd.DataFrame:
    """
    评估所有模型并返回对比结果

    Args:
        models: 训练后的模型字典
        XVal: 验证特征
        yVal: 验证目标

    Returns:
        评估结果 DataFrame，每行一个模型，每列一个指标
    """
    results = []
    for name, model in models.items():
        print(f"评估模型: {name}...")
        metrics = evaluateModel(model, XVal, yVal)
        metrics["model"] = name
        results.append(metrics)
    resultsDf = pd.DataFrame(results).set_index("model")
    return resultsDf


def selectBestModel(resultsDf: pd.DataFrame, metric: str = "f1") -> str:
    """
    根据指定指标选择最佳模型

    Args:
        resultsDf: 评估结果 DataFrame
        metric: 用于选择的指标，默认 "f1"

    Returns:
        最佳模型名称
    """
    bestModelName = resultsDf[metric].idxmax()
    return bestModelName


# =============================================================================
# 主函数
# =============================================================================


def main():
    """
    主函数：执行模型评估流程

    流程：
        1. 加载验证集数据
        2. 加载所有模型
        3. 评估所有模型
        4. 选择最佳模型
        5. 在全量数据上重训最佳模型并保存
    """
    print("=" * 60)
    print("📊 Titanic 模型评估")
    print("=" * 60)

    # 1. 加载验证集数据
    print("\n📂 加载验证集数据...")
    import joblib

    valDataPath = os.path.join(config.MODEL_DIR, "val_data.pkl")
    if not os.path.exists(valDataPath):
        print("❌ 错误: 验证集数据不存在！")
        print("   请先运行: python train.py")
        return
    valData = joblib.load(valDataPath)
    XVal = valData["XVal"]
    yVal = valData["yVal"]
    X = valData["X"]
    y = valData["y"]
    print(f"   验证集: {XVal.shape}")

    # 2. 加载所有模型
    print("\n🔧 加载模型...")
    models = loadAllModels()
    if not models:
        print("❌ 错误: 没有找到任何模型文件！")
        print("   请先运行: python train.py")
        return

    # 3. 评估所有模型
    print("\n📈 评估模型...")
    resultsDf = evaluateAllModels(models, XVal, yVal)
    print("\n" + "=" * 60)
    print("📈 模型评估结果")
    print("=" * 60)
    print(resultsDf.to_string())

    # 4. 选择最佳模型
    bestModelName = selectBestModel(resultsDf, metric="f1")
    print(
        f"\n🏆 最佳模型: {bestModelName} (F1={resultsDf.loc[bestModelName, 'f1']:.4f})"
    )

    # 5. 在全量数据上重训最佳模型
    print(f"\n🔄 在全量数据上重训 {bestModelName}...")
    from train import getModels, trainModel

    bestModel = getModels()[bestModelName]
    bestModel = trainModel(bestModel, X, y)

    # 6. 保存全量重训后的最佳模型
    bestModelPath = os.path.join(config.MODEL_DIR, f"{bestModelName}_final.pkl")
    joblib.dump(bestModel, bestModelPath)
    print(f"   已保存: {bestModelPath}")

    # 7. 保存最佳模型名称（供 predict.py 使用）
    bestInfoPath = os.path.join(config.MODEL_DIR, "best_model_info.pkl")
    joblib.dump({"name": bestModelName, "path": bestModelPath}, bestInfoPath)
    print(f"   已保存: {bestInfoPath}")

    print("\n" + "=" * 60)
    print("✅ 模型评估完成！")
    print("   下一步: python predict.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
