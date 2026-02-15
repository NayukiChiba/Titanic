"""
探索性数据分析
1. 读入与概览(已完成):    形状、前几行/后几行、数据类型、info、describe
2. 质量检查(已完成):     缺失值、重复值、异常值、离群点
3. 单变量分析(已完成):    数值列分布（直方图/箱线图）、类别列分布（计数图）
4. 目标变量分析(已完成):   目标分布、类别不平衡情况
5. 特征与目标关系(已完成): 类别-目标均值、数值-目标箱线/分布对比
6. 特征之间关系:   相关性、共线性、交互关系
7. 初步处理建议:   缺失填补/删除、异常处理、编码方式、特征工程
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


# 数据类，负责加载数据并提供基本的概览功能
class Data:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        return self.data

    # 获取数据的形状
    def shape(self):
        return self.data.shape

    # 获取数据的列名
    def columns(self):
        return self.data.columns

    # 获取数据的前几行
    def head(self, n=5):
        return self.data.head(n)

    # 获取数据的后几行
    def tail(self, n=5):
        return self.data.tail(n)

    # 获取数据的基本类型
    def info(self):
        return self.data.info()

    # 获取数据的描述性统计信息
    def describe(self):
        return self.data.describe()

    # 每一列的取值数量
    def value_counts(self):
        value_counts = {}
        for col in self.data.columns:
            value_counts[col] = self.data[col].value_counts()
        return value_counts

    def getallinfo(self):
        print("数据集形状:")
        print(self.data.shape)
        # 训练集的形状是(891, 12)
        # 测试集的形状是(418, 11)

        print("数据集的列名:\n", self.data.columns)
        """
        一共12个列
        columns:
            'PassengerId', 
            'Survived', 
            'Pclass', 
            'Name', 
            'Sex', 
            'Age', 
            'SibSp',
            'Parch', 
            'Ticket',
            'Fare',
            'Cabin',
            'Embarked'
        """

        print("数据集的前5行:\n", self.data.head())
        # 前5行数据

        print("数据集后5行:\n", self.data.tail())

        print("数据集信息:")
        print(self.data.info())
        # 基本信息，包括数据类型和非空值数量

        print("数据集描述性统计信息:\n", self.data.describe())
        # 描述性统计信息

        print("数据集每一列的取值数量:\n", self.data.value_counts())
        # 每一列的取值数量

    # 检查缺失值
    def missingCheck(self):
        missing_values = self.data.isnull().sum()
        if missing_values.sum() > 0:
            print("有缺失值! 缺失的列和数量为: ")
            print(missing_values[missing_values > 0])
        else:
            print("没有缺失值")
        return missing_values[missing_values > 0]

    # 检查重复值
    def duplicateCheck(self):
        duplicate_count = self.data.duplicated().sum()
        print("Duplicate values:", duplicate_count)
        return duplicate_count

    # 检查异常值
    def outlierCheck(self):
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        results = []

        for col in numeric_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            IQRoutliers = (self.data[col] < lower) | (self.data[col] > upper)

            z = (self.data[col] - self.data[col].mean()) / self.data[col].std()
            zscoreoutliers = (z < -3) | (z > 3)

            results.append(
                {
                    "column": col,
                    "IQR异常值": int(IQRoutliers.sum()),
                    "Z-score异常值": int(zscoreoutliers.sum()),
                    "IQR上界": upper,
                    "IQR下界": lower,
                }
            )

        print("数值列的异常值检查结果: ")
        report = pd.DataFrame(results)
        print(report)
        return report

    # 一次性跑完所有质量检查
    def allCheck(self):
        self.missingCheck()
        self.duplicateCheck()
        self.outlierCheck()

    def non_outlier(self, col, method="IQR"):
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

    def summary(self):
        summary_dict = {}
        for col in self.data.columns:
            summary_dict[col] = {
                "数据类型": self.data[col].dtype,
                "非空值数量": self.data[col].notnull().sum(),
                "缺失值数量": self.data[col].isnull().sum(),
                "唯一值数量": self.data[col].nunique(),
            }
        return pd.DataFrame(summary_dict).T

    def quality_report(self):
        missing = self.missingCheck()
        duplicates = self.duplicateCheck()
        outliers = self.outlierCheck()
        report = {
            "缺失值": missing,
            "重复值数量": duplicates,
            "异常值": outliers,
        }
        return report


"""
12个列
'PassengerId': 不需要plot 
'Survived': 只有0和1两类，可以用计数图展示
'Pclass': 只有1、2、3三类，可以用计数图展示
'Name': 不需要plot
'Sex': 只有male和female两类，可以用计数图展示
'Age': 从0到74, 可以用直方图和箱线图展示
'SibSp': 从0到8, 可以用计数图展示
'Parch': 从0到6, 可以用计数图展示
'Ticket': 不需要plot
'Fare': 从0到512, 可以用直方图和箱线图展示
'Cabin': 都是ABCDEFGT开头的字符串，可以用计数图展示
'Embarked': 只有C、Q、S三类，可以用计数图展示
"""


class Plotter:
    def __init__(self, data: Data):
        # 保存 Data 对象与原始 DataFrame
        self.source = data
        self.data = data.data

    def hist(self, col, drop_outliers=False):
        plot_data = self.data
        if drop_outliers:
            plot_data = self.source.non_outlier(col)
        plt.figure(figsize=(8, 6))
        ax = sns.histplot(plot_data[col].dropna(), bins=30, kde=True)
        for container in ax.containers:
            ax.bar_label(container, fmt="%d", padding=2)
        plt.title(f"{col} 分布", fontsize=14, weight="bold")
        plt.xlabel(col)
        plt.ylabel("频数")
        plt.tight_layout()
        plt.show()

    def count(self, col, data=None):
        plt.figure(figsize=(8, 6))
        plot_data = data if data is not None else self.data
        ax = sns.countplot(x=col, data=plot_data)
        for container in ax.containers:
            ax.bar_label(container, fmt="%d", padding=3)

        # 美化细节
        ax.set_title(f"{col} 频数分布", fontsize=14, weight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("数量")

        plt.tight_layout()
        plt.show()


# 目标变量分析, 看看有没有问题
def target_analysis(Data: Data, target_col="Survived"):
    # 获取targe
    data = Data.data
    target = data[target_col]

    # 看看有没有缺失
    missing = target.isnull().sum()
    if missing > 0:
        print(f"目标变量 {target_col} 有 {missing} 个缺失值")
    else:
        print(f"目标变量 {target_col} 没有缺失值")

    # 看看target有哪些类型
    counts = target.value_counts().sort_index()
    # 看看类别的比例
    percent = (counts / counts.sum()) * 100
    report = pd.DataFrame({target_col: counts, "百分比": percent})
    print(f"目标变量 {target_col} 的分布:\n{report}")

    # 不平衡的程度
    # 如果类别大于1
    if len(counts) > 1:
        imbalance_ratio = counts.max() / counts.min()
        print(f"类别不平衡程度 (最大类别数量 / 最小类别数量): {imbalance_ratio:.2f}")

        # 最大的类别占比
        marjority_class_percentage = percent.max()
        print(f"最大的类别占比: {marjority_class_percentage:.2f}%")

        # 熵
        p = counts / counts.sum()
        entropy = -np.sum(p * np.log2(p))
        print(f"目标变量的熵: {entropy:.4f}")
    else:
        print("目标变量只有一个类别, 无法计算不平衡程度和熵")

    # 可视化目标变量分布
    plt.figure(figsize=(8, 6))
    order = counts.index.tolist()
    ax = sns.countplot(x=target_col, data=data, order=order)
    for i, p in enumerate(ax.patches):
        ax.text(
            p.get_x() + p.get_width() / 2,
            p.get_height() + 5,
            f"{counts[i]} ({percent[i]:.1f}%)",
            ha="center",
        )
    ax.set_title(f"{target_col} 分布", fontsize=14, weight="bold")
    ax.set_xlabel(target_col)
    ax.set_ylabel("数量")
    plt.tight_layout()
    plt.show()


def featureTargetRelations(
    Data: Data,
    targetCol="Survived",
    catCols=None,
    numCols=None,
    missing="keep",  # "keep"=把缺失当成一类; "drop"=只在当前分析里丢掉缺失
):
    """
    分析特征与目标变量的关系
    - 类别特征: 计算每个类别的目标均值, 可视化柱状图
    - 数值特征: 计算数值特征在不同目标类别下的分布, 可视化箱线图或小提琴图
    Args:
    - dataFrame: 包含特征和目标的 DataFrame
    - targetCol: 目标变量列名
    - catCols: 类别特征列名列表, 如果 None 则自动识别
    - numCols: 数值特征列名列表, 如果 None 则自动识别
    - maxCat: 类别特征的最大类别数, 超过则不分析
    - missing: 处理缺失值的方式, "keep" 或 "drop"
    """
    # 目标缺失直接删除, 因为无法分析
    data = Data.data
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
        plt.title(f"{col} vs {targetCol} (mean)")
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
        plt.title(f"{numCol} by {targetCol}")
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
        plt.title(f"{numCol} distribution by {targetCol}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    train_filepath = "datasets/train.csv"
    test_filepath = "datasets/test.csv"

    train = Data(train_filepath)
    test = Data(test_filepath)

    # # 获取数据的基本信息
    # train.getallinfo()
    # # test.getallinfo()

    # # 质量检查（已归属到 Data 类）
    # train.allCheck()
    # test.allCheck()

    # 使用 Plotter 进行可视化
    # plotter = Plotter(train)
    # plotter.count("Survived")
    # plotter.count("Pclass")
    # plotter.count("Sex")
    # plotter.count("SibSp")
    # plotter.count("Parch")
    # plotter.count("Embarked")

    # # Cabin 首字母计数示例
    # train.data["Cabin"] = train.data["Cabin"].str[0].fillna("Unknown")
    # plotter.count("Cabin")

    # # 直方图（可选剔除异常值）
    # plotter.hist("Age", drop_outliers=True)
    # plotter.hist("Fare", drop_outliers=True)

    # 目标变量分析
    # target_analysis(train, target_col="Survived")

    # 特征与目标关系分析
    featureTargetRelations(
        train,
        targetCol="Survived",
        missing="keep",
        catCols=["Pclass", "Sex", "SibSp", "Parch", "Embarked"],
        numCols=["Age", "Fare"],
    )
