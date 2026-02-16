"""
特征工程模块

功能：
1. 缺失值处理（Age、Cabin、Embarked）
2. 特征创建（Title、FamilySize、IsAlone 等）
3. 特征编码（类别变量转数值）
4. 特征选择（删除无用列）

使用方法：
    python featureEngineering.py

处理流程：
    原始数据 → 缺失值填充 → 特征创建 → 特征编码 → 输出处理后数据
"""

import os
import sys
from dataclasses import dataclass, field

import pandas as pd


# =============================================================================
# 填充参数类（P2 修复：存储训练集统计量，测试集复用）
# =============================================================================
@dataclass
class FillParams:
    """存储从训练集计算的填充参数，供测试集复用"""

    # Age: 按 (Pclass, Sex) 分组的中位数
    ageMedianByGroup: dict = field(default_factory=dict)
    # Age: 全局中位数（兜底）
    ageMedianGlobal: float = 0.0
    # Embarked: 众数
    embarkedMode: str = "S"
    # Fare: 中位数
    fareMedian: float = 0.0


# =============================================================================
# 第一部分：数据加载
# =============================================================================


def loadData(filename: str) -> pd.DataFrame:
    """
    加载 CSV 数据文件

    Args:
        filename(str): 数据文件名(位于 datasets 目录下)

    Returns:
        加载后的 DataFrame
    """
    filepath = os.path.join("datasets", filename)
    # 如果没有文件夹或者文件
    if not os.path.exists(filepath):
        print(f"❌ 错误: 数据文件 '{filepath}' 不存在！")
        sys.exit(1)
    data = pd.read_csv(filepath)
    return data


# =============================================================================
# 第二部分：缺失值处理
# =============================================================================


def fillAge(df: pd.DataFrame, params: FillParams | None = None) -> pd.DataFrame:
    """
    填充 Age 缺失值

    Args:
        df: 原始 DataFrame
        params: 填充参数（测试集传入训练集的参数）

    Returns:
        填充后的 DataFrame

    策略选择（任选其一实现）：
        1. 简单策略：用整体中位数填充
        2. 分组策略：按 Pclass + Sex 分组，用组内中位数填充
        3. 模型策略：用其他特征预测 Age（进阶）

    提示：
        - df['Age'].median() 获取中位数
        - df['Age'].fillna(value) 填充缺失值
        - df.groupby(['Pclass', 'Sex'])['Age'].transform('median') 分组中位数
    """
    if params is None:
        # 训练集：按 Pclass 和 Sex 分组计算 Age 的中位数
        df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
            lambda x: x.fillna(x.median())
        )
        # P2 修复：分组中位数为 NaN 时（该组 Age 全缺失），用全局中位数兜底
        globalMedian = df["Age"].median()
        df["Age"] = df["Age"].fillna(globalMedian)
    else:
        # 测试集：使用训练集的分组中位数
        def fillWithParams(row):
            if pd.isna(row["Age"]):
                key = (row["Pclass"], row["Sex"])
                groupMedian = params.ageMedianByGroup.get(key)
                # P2 修复：分组中位数为 NaN 时回退到全局中位数
                if groupMedian is None or pd.isna(groupMedian):
                    return params.ageMedianGlobal
                return groupMedian
            return row["Age"]

        df["Age"] = df.apply(fillWithParams, axis=1)
    return df


def fillEmbarked(df: pd.DataFrame, params: FillParams | None = None) -> pd.DataFrame:
    """
    填充 Embarked 缺失值

    Args:
        df: 原始 DataFrame
        params: 填充参数（测试集传入训练集的参数）

    Returns:
        填充后的 DataFrame

    策略：用众数填充（S 港口最多）

    提示：
        - df['Embarked'].mode()[0] 获取众数
        - df['Embarked'].fillna(value) 填充
    """
    if params is None:
        # 训练集：用当前数据的众数
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    else:
        # 测试集：使用训练集的众数
        df["Embarked"] = df["Embarked"].fillna(params.embarkedMode)
    return df


def fillFare(df: pd.DataFrame, params: FillParams | None = None) -> pd.DataFrame:
    """
    填充 Fare 缺失值（测试集可能有缺失）

    Args:
        df: 原始 DataFrame
        params: 填充参数（测试集传入训练集的参数）

    Returns:
        填充后的 DataFrame

    策略：用中位数填充
    """
    if params is None:
        # 训练集：用当前数据的中位数
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    else:
        # 测试集：使用训练集的中位数
        df["Fare"] = df["Fare"].fillna(params.fareMedian)
    return df


# =============================================================================
# 第三部分：特征创建
# =============================================================================


def extractTitle(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 Name 中提取称谓 (Title)

    Args:
        df: 原始 DataFrame

    Returns:
        添加 Title 列后的 DataFrame

    逻辑：
        1. 用正则表达式从 Name 中提取称谓（如 Mr, Mrs, Miss）
        2. 将稀有称谓合并为 'Rare'
        3. 统一称谓映射

    常见称谓分类：
        - Mr: 成年男性
        - Miss: 未婚女性
        - Mrs: 已婚女性
        - Master: 男孩
        - Rare: 其他稀有称谓（Dr, Rev, Col 等）
    """
    # 正则提取称谓
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.")

    # 合并稀有称谓
    titleMapping = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Lady": "Rare",
        "Countess": "Rare",
        "Capt": "Rare",
        "Col": "Rare",
        "Don": "Rare",
        "Dr": "Rare",
        "Major": "Rare",
        "Rev": "Rare",
        "Sir": "Rare",
        "Jonkheer": "Rare",
        "Dona": "Rare",
    }

    # 替换称谓
    df["Title"] = df["Title"].replace(titleMapping)
    return df


def createFamilySize(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建家庭规模特征

    Args:
        df: 原始 DataFrame

    Returns:
        添加 FamilySize 和 IsAlone 列后的 DataFrame

    逻辑：
        FamilySize = SibSp + Parch + 1（加上自己）
        IsAlone = 1 if FamilySize == 1 else 0

    提示：
        - 直接用列运算：df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    """
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def extractCabinLetter(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 Cabin 提取舱位等级（首字母）

    Args:
        df: 原始 DataFrame

    Returns:
        添加 CabinLetter 列后的 DataFrame

    逻辑：
        - 提取 Cabin 首字母（A, B, C, D, E, F, G, T）
        - 缺失值标记为 'X' 或 'Unknown'

    提示：
        - df['Cabin'].str[0] 获取首字母
        - .fillna('X') 填充缺失值
    """
    # 提取首字母，缺失值填充为 'X'
    df["CabinLetter"] = df["Cabin"].str[0].fillna("X")
    return df


def createAgeBin(df: pd.DataFrame) -> pd.DataFrame:
    """
    对 Age 进行分箱处理

    Args:
        df: 原始 DataFrame

    Returns:
        添加 AgeBin 列后的 DataFrame

    分箱建议：
        - 儿童: 0-12
        - 青少年: 12-18
        - 青年: 18-35
        - 中年: 35-60
        - 老年: 60+

    提示：
        - pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=[...])
    """
    df["AgeBin"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=["Child", "Teenager", "YoungAdult", "MiddleAged", "Senior"],
    )
    return df


def createFareBin(df: pd.DataFrame) -> pd.DataFrame:
    """
    对 Fare 进行分箱处理

    Args:
        df: 原始 DataFrame

    Returns:
        添加 FareBin 列后的 DataFrame

    策略选择：
        1. 等宽分箱：pd.cut()
        2. 等频分箱：pd.qcut()（推荐，每组样本数相近）

    提示：
        - pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
    """
    df["FareBin"] = pd.qcut(
        df["Fare"], q=4, labels=["Low", "Medium", "High", "VeryHigh"]
    )
    return df


# =============================================================================
# 第四部分：特征编码
# =============================================================================


def encodeSex(df: pd.DataFrame) -> pd.DataFrame:
    """
    对 Sex 进行二值编码

    Args:
        df: 原始 DataFrame

    Returns:
        编码后的 DataFrame

    映射：
        female → 0
        male → 1
        （或反过来，看你的偏好）

    提示：
        - df['Sex'].map({'female': 0, 'male': 1})
    """
    sexMapping = {"female": 0, "male": 1}
    df["Sex"] = df["Sex"].map(sexMapping)
    return df


def encodeEmbarked(df: pd.DataFrame) -> pd.DataFrame:
    """
    对 Embarked 进行 One-Hot 编码

    Args:
        df: 原始 DataFrame

    Returns:
        添加 Embarked_C, Embarked_Q, Embarked_S 列后的 DataFrame

    提示：
        - pd.get_dummies(df, columns=['Embarked'], prefix='Embarked')
        - 或手动: df['Embarked_C'] = (df['Embarked'] == 'C').astype(int)
    """
    df = pd.get_dummies(df, columns=["Embarked"], prefix="Embarked")
    return df


def encodeTitle(df: pd.DataFrame) -> pd.DataFrame:
    """
    对 Title 进行编码

    Args:
        df: 原始 DataFrame

    Returns:
        编码后的 DataFrame

    策略选择：
        1. Label Encoding: Mr=0, Miss=1, Mrs=2, Master=3, Rare=4
        2. One-Hot Encoding: 创建 Title_Mr, Title_Miss 等列

    提示：
        - Label: df['Title'].map({'Mr': 0, 'Miss': 1, ...})
        - One-Hot: pd.get_dummies(df, columns=['Title'])
    """
    # 上面已经把Title合并了稀有称谓，所以直接Label Encoding
    titleMapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
    df["Title"] = df["Title"].map(titleMapping)
    # 或者使用One-Hot编码
    # df = pd.get_dummies(df, columns=['Title'], prefix='Title')
    return df


def encodeCategorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    统一编码所有类别特征

    Args:
        df: 原始 DataFrame

    Returns:
        所有类别特征编码后的 DataFrame

    调用顺序：
        1. encodeSex()
        2. encodeEmbarked()
        3. encodeTitle()
        4. 其他需要编码的特征...
    """
    df = encodeSex(df)
    df = encodeEmbarked(df)
    df = encodeTitle(df)
    # 其他编码函数调用...
    return df


# =============================================================================
# 第五部分：特征选择
# =============================================================================


def dropUselessColumns(df: pd.DataFrame) -> pd.DataFrame:
    """
    删除无用的原始列

    Args:
        df: 处理后的 DataFrame

    Returns:
        删除无用列后的 DataFrame

    建议删除的列：
        - PassengerId: 仅用于标识，无预测意义
        - Name: 已提取 Title
        - Ticket: 格式不规则，难以利用
        - Cabin: 已提取 CabinLetter

    提示：
        - df.drop(columns=['PassengerId', 'Name', ...], inplace=True)
        - 注意：测试集需要保留 PassengerId 用于提交！
    """
    df = df.drop(columns=["Name", "Ticket", "Cabin"])
    return df


def selectFeatures(df: pd.DataFrame, isTest: bool = False) -> pd.DataFrame:
    """
    选择最终用于建模的特征

    Args:
        df: 处理后的 DataFrame
        isTest: 是否为测试集（测试集需保留 PassengerId）

    Returns:
        仅包含建模特征的 DataFrame

    最终特征列表（参考）：
        - Pclass
        - Sex (编码后)
        - Age (或 AgeBin)
        - Fare (或 FareBin)
        - FamilySize
        - IsAlone
        - Title (编码后)
        - Embarked (One-Hot 后)
        - CabinLetter (编码后，可选)
    """
    # 定义最终特征列表
    features = [
        "Pclass",
        "Sex",
        "Age",
        "Fare",
        "FamilySize",
        "IsAlone",
        "Title",
        "Embarked_C",
        "Embarked_Q",
        "Embarked_S",
    ]

    # P1 修复：补齐可能缺失的 Embarked 哑变量列
    for col in ["Embarked_C", "Embarked_Q", "Embarked_S"]:
        if col not in df.columns:
            df[col] = 0

    return df[features + ["PassengerId"]] if isTest else df[features]


# =============================================================================
# 第六部分：完整流水线
# =============================================================================


def fitFillParams(df: pd.DataFrame) -> FillParams:
    """
    从训练集计算填充参数

    Args:
        df: 训练集 DataFrame

    Returns:
        FillParams 对象，包含所有填充参数
    """
    params = FillParams()

    # Age: 按 (Pclass, Sex) 分组的中位数
    ageGrouped = df.groupby(["Pclass", "Sex"])["Age"].median()
    params.ageMedianByGroup = ageGrouped.to_dict()
    params.ageMedianGlobal = df["Age"].median()

    # Embarked: 众数
    params.embarkedMode = df["Embarked"].mode()[0]

    # Fare: 中位数
    params.fareMedian = df["Fare"].median()

    return params


def preprocessData(
    df: pd.DataFrame, isTest: bool = False, params: FillParams | None = None
) -> pd.DataFrame:
    """
    完整的数据预处理流水线

    Args:
        df: 原始 DataFrame
        isTest: 是否为测试集
        params: 填充参数（测试集需传入训练集的参数）

    Returns:
        处理完成的 DataFrame

    处理顺序：
        1. 缺失值处理
        2. 特征创建
        3. 特征编码
        4. 特征选择

    注意：
        - 训练集和测试集要用相同的处理逻辑
        - 某些参数（如 Age 中位数）应从训练集计算，应用到测试集
    """
    # 缺失值处理
    df = fillAge(df, params)
    df = fillEmbarked(df, params)
    df = fillFare(df, params)

    # 特征创建
    df = extractTitle(df)
    df = createFamilySize(df)
    # df = extractCabinLetter(df)
    df = createAgeBin(df)
    # df = createFareBin(df)  # P2 修复：FareBin 未被 selectFeatures 使用，移除避免 qcut 报错

    # 特征编码
    df = encodeCategorical(df)

    # 删除无用列
    df = dropUselessColumns(df)

    # 特征选择
    df = selectFeatures(df, isTest=isTest)

    return df


# =============================================================================
# 第七部分：主函数
# =============================================================================
def processAndSave(
    filename: str, isTest: bool = False, params: FillParams | None = None
) -> tuple[pd.DataFrame, FillParams | None]:
    """
    加载、处理并保存单个数据文件

    Args:
        filename: 数据文件名（如 "train.csv"）
        isTest: 是否为测试集
        params: 填充参数（测试集需传入训练集的参数）

    Returns:
        (处理完成的 DataFrame, 填充参数)
    """
    # 1. 加载数据
    print(f"\n📂 加载 {filename}...")
    df = loadData(filename)
    print(f"   原始形状: {df.shape}")

    # 2. 如果是训练集，先计算填充参数
    if not isTest:
        params = fitFillParams(df)
        print("   ✓ 已计算填充参数")

    # 3. 预处理
    print("⚙️  处理中...")
    processedDf = preprocessData(df, isTest=isTest, params=params)
    print(f"   处理后形状: {processedDf.shape}")

    # 4. 保存结果
    outputName = filename.replace(".csv", "_processed.csv")
    outputPath = os.path.join("datasets", outputName)
    processedDf.to_csv(outputPath, index=False)
    print(f"💾 已保存: {outputPath}")

    return processedDf, params


def main():
    """
    主函数：执行完整的特征工程流程
    """
    print("=" * 60)
    print("🔧 Titanic 特征工程")
    print("=" * 60)

    # 处理训练集（同时计算填充参数）
    trainProcessed, fillParams = processAndSave("train.csv", isTest=False)

    # 处理测试集（复用训练集的填充参数）
    testProcessed, _ = processAndSave("test.csv", isTest=True, params=fillParams)

    # 打印摘要
    print("\n" + "=" * 60)
    print("📊 处理结果摘要")
    print("=" * 60)
    print(f"训练集: {trainProcessed.shape}")
    print(f"测试集: {testProcessed.shape}")
    print(f"特征列表: {list(trainProcessed.columns)}")
    print("\n✅ 特征工程完成！")


if __name__ == "__main__":
    main()
