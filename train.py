"""
模型训练模块

功能：
1. 数据加载与划分
2. 多模型训练
3. 模型持久化

使用方法：
    python train.py
"""

import os
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

import config

# =============================================================================
# 数据加载与划分
# =============================================================================


def loadProcessedData(
    trainFilename: str, testFilename: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载预处理后的数据

    Args:
        trainFilename: 训练集文件名
        testFilename: 测试集文件名

    Returns:
        (训练集 DataFrame, 测试集 DataFrame)
    """
    trainFilepath = os.path.join(config.DATASETS_DIR, trainFilename)
    testFilepath = os.path.join(config.DATASETS_DIR, testFilename)

    # 检查预处理文件是否存在
    if not os.path.exists(trainFilepath) or not os.path.exists(testFilepath):
        print("❌ 错误: 预处理数据文件不存在！")
        print(
            f"   缺失: {trainFilepath if not os.path.exists(trainFilepath) else testFilepath}"
        )
        print("   请先运行特征工程: python featureEngineering.py")
        raise FileNotFoundError("预处理数据文件不存在，请先运行 featureEngineering.py")

    trainDf = pd.read_csv(trainFilepath)
    testDf = pd.read_csv(testFilepath)
    return trainDf, testDf


def splitFeatureTarget(
    df: pd.DataFrame, targetCol: str = "Survived"
) -> tuple[pd.DataFrame, pd.Series]:
    """
    分离特征和目标变量

    Args:
        df: 包含特征和目标的 DataFrame
        targetCol: 目标列名，默认 "Survived"

    Returns:
        (特征 DataFrame, 目标 Series)
    """
    X = df.drop(columns=[targetCol])
    y = df[targetCol]
    return X, y


def trainTestSplit(
    X: pd.DataFrame, y: pd.Series, testSize: float = 0.2, randomState: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    划分训练集和验证集

    Args:
        X: 特征 DataFrame
        y: 目标 Series
        testSize: 验证集比例，默认 0.2
        randomState: 随机种子，默认 42

    Returns:
        (X_train, X_val, y_train, y_val)
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=testSize, random_state=randomState, stratify=y
    )
    return X_train, X_val, y_train, y_val


# =============================================================================
# 模型定义
# =============================================================================


def createLogisticRegression() -> Any:
    """创建逻辑回归模型"""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=1000)
    return model


def createRandomForest(
    nEstimators: int = 100, randomState: int = 42, nJobs: int = -1
) -> Any:
    """创建随机森林模型"""
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=nEstimators, random_state=randomState, n_jobs=nJobs
    )
    return model


def createXGBoost(randomState: int = 42) -> Any:
    """创建 XGBoost 模型"""
    from xgboost import XGBClassifier

    model = XGBClassifier(eval_metric="logloss", random_state=randomState)
    return model


def createLightGBM(randomState: int = 42) -> Any:
    """创建 LightGBM 模型"""
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(random_state=randomState, verbose=-1)
    return model


def getModels() -> dict[str, Any]:
    """
    获取所有待训练的模型

    Returns:
        模型字典 {模型名称: 模型实例}
    """
    models = {
        "LogisticRegression": createLogisticRegression(),
        "RandomForest": createRandomForest(),
        "XGBoost": createXGBoost(),
        "LightGBM": createLightGBM(),
    }
    return models


# =============================================================================
# 模型训练
# =============================================================================


def trainModel(model: Any, XTrain: pd.DataFrame, yTrain: pd.Series) -> Any:
    """
    训练单个模型

    Args:
        model: sklearn 兼容的模型实例
        XTrain: 训练特征
        yTrain: 训练目标

    Returns:
        训练后的模型
    """
    model.fit(XTrain, yTrain)
    return model


def trainAllModels(
    models: dict[str, Any], XTrain: pd.DataFrame, yTrain: pd.Series
) -> dict[str, Any]:
    """
    训练所有模型

    Args:
        models: 模型字典 {名称: 模型实例}
        XTrain: 训练特征
        yTrain: 训练目标

    Returns:
        训练后的模型字典 {名称: 训练后模型}
    """
    trainedModels = {}
    for name, model in models.items():
        print(f"训练模型: {name}...")
        trainedModel = trainModel(model, XTrain, yTrain)
        trainedModels[name] = trainedModel
    return trainedModels


# =============================================================================
# 模型持久化
# =============================================================================


def saveModel(model: Any, filepath: str) -> None:
    """
    保存模型到文件

    Args:
        model: 训练后的模型
        filepath: 保存路径
    """
    import joblib

    joblib.dump(model, filepath)


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


# =============================================================================
# 主函数
# =============================================================================


def main():
    """
    主函数：执行模型训练流程

    流程：
        1. 加载预处理后的数据
        2. 分离特征和目标
        3. 划分训练集和验证集
        4. 创建并训练所有模型
        5. 保存模型
    """
    print("=" * 60)
    print("🤖 Titanic 模型训练")
    print("=" * 60)

    # 1. 加载数据
    print("\n📂 加载数据...")
    trainDf, _ = loadProcessedData("train_processed.csv", "test_processed.csv")
    print(f"   训练集: {trainDf.shape}")

    # 2. 分离特征和目标
    print("\n🔍 分离特征和目标...")
    X, y = splitFeatureTarget(trainDf, targetCol="Survived")
    print(f"   特征数: {X.shape[1]}, 样本数: {X.shape[0]}")

    # 3. 划分训练集和验证集
    print("\n✂️  划分训练集和验证集...")
    XTrain, XVal, yTrain, yVal = trainTestSplit(X, y, testSize=0.2, randomState=42)
    print(f"   训练集: {XTrain.shape[0]}, 验证集: {XVal.shape[0]}")

    # 4. 创建模型
    print("\n🔧 创建模型...")
    models = getModels()
    print(f"   模型列表: {list(models.keys())}")

    # 5. 训练所有模型
    print("\n🚀 训练模型...")
    trainedModels = trainAllModels(models, XTrain, yTrain)

    # 6. 保存模型
    print("\n💾 保存模型...")
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    for modelName, model in trainedModels.items():
        modelPath = os.path.join(config.MODEL_DIR, f"{modelName}.pkl")
        saveModel(model, modelPath)
        print(f"   已保存: {modelPath}")

    # 7. 保存验证集信息（供 evaluate.py 使用）
    valDataPath = os.path.join(config.MODEL_DIR, "val_data.pkl")
    import joblib

    joblib.dump({"XVal": XVal, "yVal": yVal, "X": X, "y": y}, valDataPath)
    print(f"   已保存: {valDataPath}")

    print("\n" + "=" * 60)
    print("✅ 模型训练完成！")
    print("   下一步: python evaluate.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
