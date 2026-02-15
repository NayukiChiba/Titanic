"""
探索性数据分析模块

功能：
1. 读入与概览(已完成):    形状、前几行/后几行、数据类型、info、describe
2. 质量检查(已完成):     缺失值、重复值、异常值、离群点
3. 单变量分析(已完成):    数值列分布（直方图/箱线图）、类别列分布（计数图）
4. 目标变量分析(已完成):   目标分布、类别不平衡情况
5. 特征与目标关系(已完成): 类别-目标均值、数值-目标箱线/分布对比
6. 特征之间关系(已完成):   相关性、共线性、交互关系
7. 初步处理建议(已完成):   缺失填补/删除、异常处理、编码方式、特征工程

使用方法：
    python eda.py
"""

import os
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


# 数据类，负责加载数据并提供基本的概览功能
class Data:
    def __init__(self, filePath: str):
        self.filePath = filePath
        self.data = self.loadData()

    def loadData(self) -> pd.DataFrame:
        """加载 CSV 数据文件"""
        self.data = pd.read_csv(self.filePath)
        return self.data

    def getShape(self) -> tuple:
        """获取数据的形状"""
        return self.data.shape

    def getColumns(self) -> pd.Index:
        """获取数据的列名"""
        return self.data.columns

    def getHead(self, n: int = 5) -> pd.DataFrame:
        """获取数据的前几行"""
        return self.data.head(n)

    def getTail(self, n: int = 5) -> pd.DataFrame:
        """获取数据的后几行"""
        return self.data.tail(n)

    def getInfo(self) -> None:
        """获取数据的基本类型"""
        return self.data.info()

    def getDescribe(self) -> pd.DataFrame:
        """获取数据的描述性统计信息"""
        return self.data.describe()

    def getValueCounts(self) -> dict[str, pd.Series]:
        """每一列的取值数量"""
        valueCounts = {}
        for col in self.data.columns:
            valueCounts[col] = self.data[col].value_counts()
        return valueCounts

    def getAllInfo(self) -> None:
        """获取数据集的全部基本信息"""
        print("数据集形状:")
        print(self.data.shape)

        print("数据集的列名:\n", self.data.columns)

        print("数据集的前5行:\n", self.data.head())

        print("数据集后5行:\n", self.data.tail())

        print("数据集信息:")
        print(self.data.info())

        print("数据集描述性统计信息:\n", self.data.describe())

        print("数据集每一列的取值数量:\n", self.data.value_counts())

    def checkMissing(self) -> pd.Series:
        """检查缺失值"""
        missingValues = self.data.isnull().sum()
        if missingValues.sum() > 0:
            print("有缺失值! 缺失的列和数量为: ")
            print(missingValues[missingValues > 0])
        else:
            print("没有缺失值")
        return missingValues[missingValues > 0]

    def checkDuplicate(self) -> int:
        """检查重复值"""
        duplicateCount = self.data.duplicated().sum()
        print(f"重复值数量: {duplicateCount}")
        return duplicateCount

    def checkOutlier(self) -> pd.DataFrame:
        """检查异常值"""
        numericCols = self.data.select_dtypes(include=[np.number]).columns
        results = []

        for col in numericCols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            iqrOutliers = (self.data[col] < lower) | (self.data[col] > upper)

            z = (self.data[col] - self.data[col].mean()) / self.data[col].std()
            zscoreOutliers = (z < -3) | (z > 3)

            results.append(
                {
                    "column": col,
                    "IQR异常值": int(iqrOutliers.sum()),
                    "Z-score异常值": int(zscoreOutliers.sum()),
                    "IQR上界": upper,
                    "IQR下界": lower,
                }
            )

        print("数值列的异常值检查结果: ")
        report = pd.DataFrame(results)
        print(report)
        return report

    def runAllChecks(self) -> None:
        """一次性跑完所有质量检查"""
        self.checkMissing()
        self.checkDuplicate()
        self.checkOutlier()

    def getNonOutlier(self, col: str, method: str = "IQR") -> pd.DataFrame:
        """获取剔除异常值后的数据"""
        if method == "IQR":
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            return self.data[(self.data[col] >= lower) & (self.data[col] <= upper)]
        elif method == "Z-score":
            z = (self.data[col] - self.data[col].mean()) / self.data[col].std()
            return self.data[(z >= -3) & (z <= 3)]
        else:
            raise ValueError("模式错误, 请选择 'IQR' 或 'Z-score'")

    def getSummary(self) -> pd.DataFrame:
        """获取数据集摘要信息"""
        summaryDict = {}
        for col in self.data.columns:
            summaryDict[col] = {
                "数据类型": self.data[col].dtype,
                "非空值数量": self.data[col].notnull().sum(),
                "缺失值数量": self.data[col].isnull().sum(),
                "唯一值数量": self.data[col].nunique(),
            }
        return pd.DataFrame(summaryDict).T

    def getQualityReport(self) -> dict[str, Any]:
        """获取数据质量报告"""
        missing = self.checkMissing()
        duplicates = self.checkDuplicate()
        outliers = self.checkOutlier()
        report = {
            "缺失值": missing,
            "重复值数量": duplicates,
            "异常值": outliers,
        }
        return report


class Plotter:
    """可视化绑定类，负责绑定 Data 对象并提供可视化方法"""

    def __init__(self, data: Data):
        # 保存 Data 对象与原始 DataFrame
        self.source = data
        self.data = data.data

    def plotHist(self, col: str, dropOutliers: bool = False) -> None:
        """绘制直方图"""
        plotData = self.data
        if dropOutliers:
            plotData = self.source.getNonOutlier(col)
        plt.figure(figsize=(8, 6))
        ax = sns.histplot(plotData[col].dropna(), bins=30, kde=True)
        for container in ax.containers:
            ax.bar_label(container, fmt="%d", padding=2)
        plt.title(f"{col} 分布", fontsize=14, weight="bold")
        plt.xlabel(col)
        plt.ylabel("频数")
        plt.tight_layout()
        plt.show()

    def plotCount(self, col: str, data: pd.DataFrame | None = None) -> None:
        """绘制计数图"""
        plt.figure(figsize=(8, 6))
        plotData = data if data is not None else self.data
        ax = sns.countplot(x=col, data=plotData)
        for container in ax.containers:
            ax.bar_label(container, fmt="%d", padding=3)

        # 美化细节
        ax.set_title(f"{col} 频数分布", fontsize=14, weight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("数量")

        plt.tight_layout()
        plt.show()


# 目标变量分析
def analyzeTarget(dataObj: Data, targetCol: str = "Survived") -> None:
    """
    分析目标变量的分布情况

    Args:
        dataObj: Data 对象
        targetCol: 目标变量列名
    """
    data = dataObj.data
    target = data[targetCol]

    # 检查缺失
    missing = target.isnull().sum()
    if missing > 0:
        print(f"目标变量 {targetCol} 有 {missing} 个缺失值")
    else:
        print(f"目标变量 {targetCol} 没有缺失值")

    # 查看目标变量类别分布
    counts = target.value_counts().sort_index()
    percent = (counts / counts.sum()) * 100
    report = pd.DataFrame({targetCol: counts, "百分比": percent})
    print(f"目标变量 {targetCol} 的分布:\n{report}")

    # 计算不平衡程度
    if len(counts) > 1:
        imbalanceRatio = counts.max() / counts.min()
        print(f"类别不平衡程度 (最大类别数量 / 最小类别数量): {imbalanceRatio:.2f}")

        majorityClassPercent = percent.max()
        print(f"最大的类别占比: {majorityClassPercent:.2f}%")

        # 计算熵
        p = counts / counts.sum()
        entropy = -np.sum(p * np.log2(p))
        print(f"目标变量的熵: {entropy:.4f}")
    else:
        print("目标变量只有一个类别, 无法计算不平衡程度和熵")

    # 可视化目标变量分布
    plt.figure(figsize=(8, 6))
    order = counts.index.tolist()
    ax = sns.countplot(x=targetCol, data=data, order=order)
    for i, patch in enumerate(ax.patches):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + 5,
            f"{counts[i]} ({percent[i]:.1f}%)",
            ha="center",
        )
    ax.set_title(f"{targetCol} 分布", fontsize=14, weight="bold")
    ax.set_xlabel(targetCol)
    ax.set_ylabel("数量")
    plt.tight_layout()
    plt.show()


def analyzeFeatureTarget(
    dataObj: Data,
    targetCol: str = "Survived",
    catCols: list[str] | None = None,
    numCols: list[str] | None = None,
    missing: str = "keep",
) -> None:
    """
    分析特征与目标变量的关系

    Args:
        dataObj: Data 对象
        targetCol: 目标变量列名
        catCols: 类别特征列名列表, 如果 None 则自动识别
        numCols: 数值特征列名列表, 如果 None 则自动识别
        missing: 处理缺失值的方式, "keep"(当成一类) 或 "drop"(丢弃)
    """
    # 目标缺失直接删除
    data = dataObj.data
    data = data[data[targetCol].notnull()].copy()

    # 自动识别类别和数值特征
    if catCols is None:
        catCols = (
            data.select_dtypes(exclude=[np.number])
            .columns.drop(targetCol, errors="ignore")
            .tolist()
        )
    if numCols is None:
        numCols = (
            data.select_dtypes(include=[np.number])
            .columns.drop(targetCol, errors="ignore")
            .tolist()
        )

    # 分析类别特征
    for col in catCols:
        categorySeries = data[col].copy()

        # 处理缺失值
        if missing == "keep":
            # 把缺失值当成一个类别 "Missing"
            categorySeries = categorySeries.fillna("Missing")
            # 创建一个新的 DataFrame 用于分析和可视化
            categoryFrame = data.copy()
            # 替换原来的列为处理后的类别列
            categoryFrame[col] = categorySeries
        elif missing == "drop":
            # 只在当前分析里丢掉缺失值
            nonMissingMask = categorySeries.notna()
            # 更新类别列和分析用的 DataFrame
            categorySeries = categorySeries[nonMissingMask]
            # 创建一个新的 DataFrame 用于分析和可视化, 只包含非缺失值的行
            categoryFrame = data.loc[nonMissingMask].copy()
            # 替换原来的列为处理后的类别列
            categoryFrame[col] = categorySeries
        else:
            raise ValueError("missing 参数必须是 'keep' 或 'drop'")

        # 查看每一个列别的目标均值
        print(f"分析类别特征 '{col}' 与目标变量 '{targetCol}' 的关系:")
        categorySurvivalRate = (
            categoryFrame.groupby(col)[targetCol].mean().sort_values(ascending=False)
        )
        print(categorySurvivalRate)

        # 交叉表
        crosstab = pd.crosstab(
            categoryFrame[col],
            categoryFrame[targetCol],
            margins=True,
            normalize="index",
        )
        print(
            f"类别特征 '{col}' 与目标变量 '{targetCol}' 的交叉表 (行百分比):\n{crosstab}"
        )

        # 可视化
        plt.figure(figsize=(7, 4))
        sns.barplot(x=categorySurvivalRate.index, y=categorySurvivalRate.values)
        plt.title(f"{col} 各类别的 {targetCol} 均值", fontsize=14, weight="bold")
        plt.xlabel(col)
        plt.ylabel(f"{targetCol} 均值")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()

    # 分析数值特征
    for numCol in numCols:
        # 数值特征缺失处理：keep -> 保留缺失（统计里自然忽略）
        # drop -> 只对该列缺失行做删除
        numericFrame = data if missing == "keep" else data[data[numCol].notna()].copy()

        # 1) 不同目标类别下的描述统计
        print(f"\n[{numCol}] 不同目标类别的统计")
        groupedStats = numericFrame.groupby(targetCol)[numCol].describe()[
            ["mean", "std", "min", "max"]
        ]
        print(groupedStats)

        # 2) 箱线图：看中位数、离散程度、异常值差异
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=targetCol, y=numCol, data=numericFrame)
        plt.title(f"{numCol} 按 {targetCol} 分组箱线图", fontsize=14, weight="bold")
        plt.xlabel(targetCol)
        plt.ylabel(numCol)
        plt.tight_layout()
        plt.show()

        # 3) 分布对比：直方图+KDE，看两类是否明显分开
        plt.figure(figsize=(8, 4))
        sns.histplot(
            data=numericFrame,
            x=numCol,
            hue=targetCol,
            kde=True,
            stat="density",
            common_norm=False,
        )
        plt.title(f"{numCol} 按 {targetCol} 分组分布", fontsize=14, weight="bold")
        plt.xlabel(numCol)
        plt.ylabel("密度")
        plt.tight_layout()
        plt.show()


def analyzeFeatureRelations(
    dataObj: Data,
    numCols: list[str] | None = None,
    targetCol: str | None = "Survived",
    threshold: float = 0.7,
) -> pd.DataFrame:
    """
    分析特征之间的关系（相关性、共线性）

    Args:
        dataObj: Data 对象
        numCols: 数值特征列名列表, 如果 None 则自动识别
        targetCol: 目标变量列名, 自动识别时会排除该列
        threshold: 高相关性阈值, 默认 0.7

    Returns:
        相关性矩阵 DataFrame
    """
    data = dataObj.data

    # 自动识别数值特征，排除目标变量
    if numCols is None:
        numCols = data.select_dtypes(include=[np.number]).columns.tolist()
        if targetCol and targetCol in numCols:
            numCols.remove(targetCol)

    if len(numCols) < 2:
        print("⚠️ 数值特征少于2个，无法进行相关性分析")
        return pd.DataFrame()

    # 计算相关性矩阵
    corrMatrix = data[numCols].corr()

    print("=" * 60)
    print("📊 特征相关性分析")
    print("=" * 60)
    print("\n相关性矩阵:")
    print(corrMatrix.round(3))

    # 找出高相关性特征对
    print(f"\n高相关性特征对 (|r| > {threshold}):")
    highCorrPairs = []
    for i in range(len(numCols)):
        for j in range(i + 1, len(numCols)):
            corr = corrMatrix.iloc[i, j]
            if abs(corr) > threshold:
                highCorrPairs.append(
                    {
                        "特征1": numCols[i],
                        "特征2": numCols[j],
                        "相关系数": round(corr, 3),
                    }
                )
    if highCorrPairs:
        highCorrDf = pd.DataFrame(highCorrPairs)
        print(highCorrDf)
        print("\n⚠️ 存在高度相关的特征，建议考虑删除或合并")
    else:
        print("✅ 未发现高度相关的特征对")

    # 可视化：热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corrMatrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
    )
    plt.title("特征相关性热力图", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.show()

    return corrMatrix


def generatePreprocessSuggestions(dataObj: Data, targetCol: str = "Survived") -> None:
    """
    生成初步数据预处理建议

    Args:
        dataObj: Data 对象
        targetCol: 目标变量列名
    """
    data = dataObj.data

    print("=" * 60)
    print("📋 初步数据预处理建议")
    print("=" * 60)

    # 1. 缺失值处理建议
    print("\n【1. 缺失值处理建议】")
    missingCols = data.isnull().sum()
    missingCols = missingCols[missingCols > 0]
    if len(missingCols) > 0:
        for col in missingCols.index:
            missingRatio = missingCols[col] / len(data) * 100
            dtype = data[col].dtype

            # 使用 str(dtype) 或 .name 属性进行比较，确保兼容 pandas dtype 对象
            dtypeName = str(dtype)

            if missingRatio > 50:
                print(
                    f"  - {col}: 缺失率 {missingRatio:.1f}%，建议删除该列或使用指示变量"
                )
            elif dtypeName in ["object", "string", "category"] or dtypeName.startswith(
                "string"
            ):
                print(
                    f"  - {col}: 缺失率 {missingRatio:.1f}%，建议用众数填充或新建 'Missing' 类别"
                )
            else:
                print(
                    f"  - {col}: 缺失率 {missingRatio:.1f}%，建议用中位数/均值填充或使用模型插补"
                )
    else:
        print("  ✅ 无缺失值")

    # 2. 类别特征编码建议
    print("\n【2. 类别特征编码建议】")
    catCols = data.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    if targetCol in catCols:
        catCols.remove(targetCol)

    if catCols:
        for col in catCols:
            nunique = data[col].nunique()
            if nunique == 2:
                print(f"  - {col}: 二分类，建议使用 Label Encoding 或二值化")
            elif nunique <= 10:
                print(f"  - {col}: {nunique} 个类别，建议使用 One-Hot Encoding")
            else:
                print(
                    f"  - {col}: {nunique} 个类别，建议使用 Target Encoding 或频率编码"
                )
    else:
        print("  ✅ 无需编码的类别特征")

    # 3. 数值特征处理建议
    print("\n【3. 数值特征处理建议】")
    numCols = data.select_dtypes(include=[np.number]).columns.tolist()
    if targetCol in numCols:
        numCols.remove(targetCol)

    if numCols:
        for col in numCols:
            skewness = data[col].skew()
            if abs(skewness) > 1:
                print(f"  - {col}: 偏度 {skewness:.2f}，建议进行 log/sqrt 变换")

            # 检查异常值
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outlierCount = (
                (data[col] < Q1 - 1.5 * IQR) | (data[col] > Q3 + 1.5 * IQR)
            ).sum()
            if outlierCount > 0:
                print(
                    f"  - {col}: 有 {outlierCount} 个异常值，建议截断或 Winsorize 处理"
                )
    else:
        print("  ✅ 无数值特征")

    # 4. 特征工程建议
    print("\n【4. 特征工程建议】")
    print("  - 考虑从 Name 提取称谓 (Mr, Mrs, Miss 等)")
    print("  - 考虑合并 SibSp 和 Parch 为 FamilySize")
    print("  - 考虑从 Cabin 提取舱位等级 (A, B, C 等)")
    print("  - 考虑对 Fare 进行分箱处理")
    print("  - 考虑对 Age 进行分箱处理")


def main(filename: str, targetCol: str = "Survived") -> None:
    """
    执行完整的 EDA 流程

    Args:
        filename: 数据文件名（位于 datasets 目录下）
        targetCol: 目标变量列名，默认为 "Survived"
    """
    filepath = os.path.join("datasets", filename)

    data = Data(filepath)

    # 检查目标列是否存在
    hasTarget = targetCol in data.data.columns

    print("数据集基本信息:")
    data.getAllInfo()

    print("\n数据质量检查:")
    data.runAllChecks()

    print("\n可视化:")
    plotter = Plotter(data)

    # 目标变量可视化（仅当目标列存在时）
    if hasTarget:
        plotter.plotCount(targetCol)

    # 其他特征可视化
    plotter.plotCount("Pclass")
    plotter.plotCount("Sex")
    plotter.plotCount("SibSp")
    plotter.plotCount("Parch")
    plotter.plotCount("Embarked")

    # Cabin 首字母可视化（使用副本，避免修改原数据）
    cabinFirstLetter = data.data["Cabin"].str[0].fillna("Unknown")
    plotter.plotCount("Cabin", data=pd.DataFrame({"Cabin": cabinFirstLetter}))

    # 直方图(可选择是否剔除异常值)
    plotter.plotHist("Age", dropOutliers=True)
    plotter.plotHist("Fare", dropOutliers=True)

    # 以下分析仅当目标列存在时执行
    if hasTarget:
        # 目标变量分析
        analyzeTarget(data, targetCol=targetCol)

        # 特征与目标关系分析
        analyzeFeatureTarget(
            data,
            targetCol=targetCol,
            missing="keep",
            catCols=["Pclass", "Sex", "SibSp", "Parch", "Embarked"],
            numCols=["Age", "Fare"],
        )

        # 特征之间关系分析
        analyzeFeatureRelations(data, targetCol=targetCol)

        # 生成预处理建议
        generatePreprocessSuggestions(data, targetCol=targetCol)
    else:
        print(f"\n⚠️ 目标列 '{targetCol}' 不存在，跳过目标相关分析")

        # 仅执行特征之间关系分析（无需目标列）
        analyzeFeatureRelations(data, targetCol=None)

        # 生成预处理建议（无目标列版本）
        generatePreprocessSuggestions(data, targetCol=targetCol)


if __name__ == "__main__":
    main("train.csv")
    # main("test.csv")
