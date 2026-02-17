"""
模型训练模块

功能：
1. 数据加载与划分
2. 多模型训练与对比
3. 交叉验证评估
4. 模型持久化

使用方法：
    python modelTraining.py

处理流程：
    加载数据 → 划分数据集 → 训练模型 → 交叉验证 → 评估对比 → 保存模型
"""

import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# =============================================================================
# 第一部分：数据加载与划分
# =============================================================================


def loadProcessedData(
    trainFilename: str, testFilename: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载预处理后的数据

    Args:
        trainFile: 训练集文件路径
        testFile: 测试集文件路径

    Returns:
        (训练集 DataFrame, 测试集 DataFrame)

    提示：
        - 使用 pd.read_csv() 加载数据
        - 训练集路径: datasets/train_processed.csv
        - 测试集路径: datasets/test_processed.csv
    """
    trainFilepath = os.path.join("datasets", trainFilename)
    testFilepath = os.path.join("datasets", testFilename)
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

    提示：
        - X = df.drop(columns=[targetCol])
        - y = df[targetCol]
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

    提示：
        - 使用 sklearn.model_selection.train_test_split
        - 设置 stratify=y 保持类别比例
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=testSize, random_state=randomState, stratify=y
    )
    return X_train, X_val, y_train, y_val


# =============================================================================
# 第二部分：模型定义
# =============================================================================


def createLogisticRegression() -> Any:
    """
    创建逻辑回归模型

    Returns:
        LogisticRegression 实例

    提示：
        - from sklearn.linear_model import LogisticRegression
        - 可设置 max_iter=1000 避免收敛警告
    """
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=1000)
    return model


def createRandomForest(
    nEstimators: int = 100, randomState: int = 42, nJobs: int = -1
) -> Any:
    """
    创建随机森林模型

    Args:
        nEstimators: 树的数量，默认 100
        randomState: 随机种子，默认 42
        nJobs: 并行任务数，默认 -1（使用所有 CPU 核心）

    Returns:
        RandomForestClassifier 实例

    提示：
        - from sklearn.ensemble import RandomForestClassifier
    """
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=nEstimators, random_state=randomState, n_jobs=nJobs
    )
    return model


def createXGBoost(randomState: int = 42) -> Any:
    """
    创建 XGBoost 模型

    Args:
        randomState: 随机种子，默认 42

    Returns:
        XGBClassifier 实例

    提示：
        - from xgboost import XGBClassifier
        - 设置 use_label_encoder=False, eval_metric='logloss'
    """
    from xgboost import XGBClassifier

    model = XGBClassifier(
        use_label_encoder=False, eval_metric="logloss", random_state=randomState
    )
    return model


def createLightGBM(randomState: int = 42) -> Any:
    """
    创建 LightGBM 模型

    Args:
        randomState: 随机种子，默认 42

    Returns:
        LGBMClassifier 实例

    提示：
        - from lightgbm import LGBMClassifier
        - 设置 verbose=-1 关闭训练日志
    """
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(random_state=randomState, verbose=-1)
    return model


def getModels() -> dict[str, Any]:
    """
    获取所有待训练的模型

    Returns:
        模型字典 {模型名称: 模型实例}

    示例返回：
        {
            "LogisticRegression": LogisticRegression(),
            "RandomForest": RandomForestClassifier(),
            "XGBoost": XGBClassifier(),
            "LightGBM": LGBMClassifier()
        }
    """
    models = {
        "LogisticRegression": createLogisticRegression(),
        "RandomForest": createRandomForest(),
        "XGBoost": createXGBoost(),
        "LightGBM": createLightGBM(),
    }
    return models


# =============================================================================
# 第三部分：模型训练
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

    提示：
        - model.fit(XTrain, yTrain)
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
# 第四部分：模型评估
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

    提示：
        - from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        - yPred = model.predict(XVal)
        - yProba = model.predict_proba(XVal)[:, 1]  # 用于计算 AUC
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

    提示：
        - from sklearn.model_selection import cross_val_score
        - 可以对多个 scoring 指标进行评估
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


# =============================================================================
# 第五部分：模型持久化
# =============================================================================


def saveModel(model: Any, filepath: str) -> None:
    """
    保存模型到文件

    Args:
        model: 训练后的模型
        filepath: 保存路径

    提示：
        - import joblib
        - joblib.dump(model, filepath)
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

    提示：
        - import joblib
        - model = joblib.load(filepath)
    """
    import joblib

    model = joblib.load(filepath)
    return model


# =============================================================================
# 第六部分：预测与提交
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
# 第七部分：主函数
# =============================================================================


def main():
    """
    主函数：执行完整的模型训练流程

    流程：
        1. 加载预处理后的数据
        2. 分离特征和目标
        3. 划分训练集和验证集
        4. 创建模型
        5. 训练所有模型
        6. 评估并对比
        7. 选择最佳模型
        8. 在测试集上预测
        9. 生成提交文件
    """
    print("=" * 60)
    print("🤖 Titanic 模型训练")
    print("=" * 60)

    # 1. 加载数据
    print("\n📂 加载数据...")
    trainDf, testDf = loadProcessedData("train_processed.csv", "test_processed.csv")
    print(f"   训练集: {trainDf.shape}, 测试集: {testDf.shape}")

    # 2. 分离特征和目标（训练集有 Survived 列，测试集没有）
    print("\n🔍 分离特征和目标...")
    X, y = splitFeatureTarget(trainDf, targetCol="Survived")
    # 测试集需要保留 PassengerId 用于提交
    testPassengerIds = testDf["PassengerId"]
    XTest = testDf.drop(columns=["PassengerId"])
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

    # 6. 评估并对比
    print("\n📊 评估模型...")
    resultsDf = evaluateAllModels(trainedModels, XVal, yVal)
    print("\n" + "=" * 60)
    print("📈 模型评估结果")
    print("=" * 60)
    print(resultsDf.to_string())

    # 7. 选择最佳模型（按 F1 分数）
    bestModelName = resultsDf["f1"].idxmax()
    bestModel = trainedModels[bestModelName]
    print(
        f"\n🏆 最佳模型: {bestModelName} (F1={resultsDf.loc[bestModelName, 'f1']:.4f})"
    )

    # 8. 保存最佳模型
    print("\n💾 保存模型...")
    os.makedirs("outputs/model", exist_ok=True)
    saveModel(bestModel, f"outputs/model/{bestModelName}.pkl")
    print(f"   已保存: outputs/model/{bestModelName}.pkl")

    # 9. 在测试集上预测
    print("\n🔮 生成预测...")
    predictions = predict(bestModel, XTest)

    # 10. 生成提交文件
    print("\n📝 生成提交文件...")
    os.makedirs("outputs/predict", exist_ok=True)
    createSubmission(testPassengerIds, predictions, "outputs/predict/submission.csv")
    print("   已保存: outputs/predict/submission.csv")

    print("\n" + "=" * 60)
    print("✅ 模型训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
