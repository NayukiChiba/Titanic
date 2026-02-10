"""
探索性数据分析
1. 读入与概览:    形状、前几行/后几行、数据类型、info、describe
2. 质量检查:     缺失值、重复值、异常值、离群点
3. 单变量分析:    数值列分布（直方图/箱线图）、类别列分布（计数图）
4. 目标变量分析:   目标分布、类别不平衡情况
5. 特征与目标关系: 类别-目标均值、数值-目标箱线/分布对比
6. 特征之间关系:   相关性、共线性、交互关系
7. 初步处理建议:   缺失填补/删除、异常处理、编码方式、特征工程
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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
    
    def getallinfo(self):
        print("Train shape:", train.shape())
        # 训练集的形状是(891, 12)
        print("Test shape:", test.shape())
        # 测试集的形状是(418, 11)

        print("Train columns:\n", train.columns())
        print("Test columns:\n", test.columns())
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
            'Embarked']
        """

        print("Train head:\n", train.head())
        # 训练集的前5行数据
        print("Test head:\n", test.head())
        # 测试集的前5行数据

        print("Train tail:\n", train.tail())
        # 训练集的后5行数据
        print("Test tail:\n", test.tail())
        # 测试集的后5行数据

        print("Train info:")
        print(train.info())
        # 训练集的基本信息，包括数据类型和非空值数量
        print("Test info:")
        print(test.info())
        # 测试集的基本信息，包括数据类型和非空值数量

        print("Train describe:\n", train.describe())
        # 训练集的描述性统计信息
        print("Test describe:\n", test.describe())
        # 测试集的描述性统计信息


# 质量检查类，负责检查数据的质量问题，如缺失值、重复值、异常值、离群点等
class QualityCheck:
    def __init__(self):
        pass

    # 检查缺失值
    def missingCheck(self, data):
        missing_values = data.isnull().sum()
        if missing_values.sum() > 0:
            print("有缺失值! 缺失的列和数量为: ")
            print(missing_values[missing_values > 0])
        else:
            print("没有缺失值")
        
        # 返回缺失值的列和数量
        return missing_values[missing_values > 0]
        

    # 检查重复值
    def duplicateCheck(self, data):
        duplicate_count = data.duplicated().sum()
        print("Duplicate values:", duplicate_count)

    # 检查异常值
    def outlierCheck(self, data):
        # 使用IQR方法筛选数值列的异常值
        # IQR方法只对数值列有效，因此需要先选择数值列
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        results = []

        for col in numeric_cols:
            # 四分位数
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            # IQR
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            # 异常值判断: 小于lower或大于upper的值被认为是异常值
            IQRoutliers = (data[col] < lower) | (data[col] > upper)
        
            # 使用Z-score方法筛选数值列的异常值
            # Z-score方法也只对数值列有效，因此需要先选择数值列
            z = (data[col] - data[col].mean()) / data[col].std()
            zscoreoutliers = (z < -3) | (z > 3)
        
            results.append({
                "column": col,
                "IQR异常值": int(IQRoutliers.sum()),
                "Z-score异常值": int(zscoreoutliers.sum()),
                "IQR上界": upper,
                "IQR下界": lower,
            })
        print("数值列的异常值检查结果: ")
        print(pd.DataFrame(results))
        return pd.DataFrame(results)


    def allCheck(self, data):
        self.missingCheck(data)
        self.duplicateCheck(data)
        self.outlierCheck(data)


if __name__ == "__main__":
    train_filepath = "datasets/train.csv"
    test_filepath = "datasets/test.csv"
    
    train = Data(train_filepath)
    test = Data(test_filepath)

    # 获取数据的基本信息
    train.getallinfo()
    # test.getallinfo()


    # 质量检查
    # quality_check = QualityCheck()
    # quality_check.allCheck(train.data)
    # quality_check.allCheck(test.data)
    