# 📊 电信客户流失预测分析

**Telco Customer Churn Prediction** — 基于 Logistic Regression 的客户流失预测，面向银行数据分析/金融科技岗位求职。

[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-0.24+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)

## 📖 项目简介

使用 Logistic Regression 构建客户流失预测模型。与黑盒模型不同，Logistic Regression 的**系数具有天然可解释性**，每个特征的系数直接反映对流失概率的影响方向与强度——这对银行/金融行业尤为重要，因为监管要求模型决策必须可解释。

**核心目标：** 识别高风险流失客户，量化各因素影响，输出可落地的业务建议。

## 📊 数据集

| 指标 | 数值 |
|------|------|
| 总客户数 | 7,043 |
| 流失客户 | 1,869 (26.54%) |
| 留存客户 | 5,174 (73.46%) |
| 原始特征 | 20（含目标） |
| 含"无网络服务"三值特征 | 6 个 |

| 特征 | 说明 | 处理 |
|------|------|------|
| customerID | 客户ID | **删除** |
| gender | 性别 | Label Encode |
| SeniorCitizen | 是否老年人 | 保持 |
| Partner | 是否有伴侣 | Label Encode |
| Dependents | 是否有家属 | Label Encode |
| 	enure | 在网月数 | StandardScaler |
| PhoneService | 电话服务 | Label Encode |
| MultipleLines | 多线服务 | Label Encode (No phone service → 0) |
| InternetService | 互联网服务 | **OneHot (DSL/Fiber/No)** |
| OnlineSecurity | 在线安全 | Label Encode |
| OnlineBackup | 在线备份 | Label Encode |
| DeviceProtection | 设备保护 | Label Encode |
| TechSupport | 技术支持 | Label Encode |
| StreamingTV | 电视流媒体 | Label Encode |
| StreamingMovies | 电影流媒体 | Label Encode |
| Contract | 合同类型 | **OneHot (Month/1yr/2yr)** |
| PaperlessBilling | 电子账单 | Label Encode |
| PaymentMethod | 支付方式 | **OneHot (4类)** |
| MonthlyCharges | 月消费(USD) | StandardScaler |
| TotalCharges | 总消费 | object→float→StandardScaler |
| **Churn** | **目标标签** | **Yes/No → 1/0** |

> ⚠️ TotalCharges 为 object 类型，tenure=0 的 11 条记录为空字符串，已转为 0。

## 🏗️ 项目结构

`
bank-customer-churn-prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   ← 原始数据集
├── notebooks/
│   └── churn_analysis.ipynb                    ← 核心分析 Notebook（29 个单元格）
├── src/
│   ├── __init__.py
│   ├── data_loader.py                          ← 数据加载 + EDA 图表生成
│   ├── preprocessing.py                        ← 特征工程（编码/标准化/分割）
│   ├── model.py                                ← LR 训练 + 系数分析
│   └── evaluation.py                           ← 评估 + 混淆矩阵 + ROC
├── outputs/
│   └── figures/                                ← 8 张分析图表
├── main.py                                     ← 一键运行管线
├── README.md
├── requirements.txt
└── .gitignore
`

## 🚀 快速开始

`ash
cd bank-customer-churn-prediction

# 安装依赖
pip install -r requirements.txt

# 一键运行全部管线
python main.py

# 或打开 Notebook 交互式分析
jupyter notebook notebooks/churn_analysis.ipynb
`

## 🧠 方法

### 特征工程
- **11 个二分类特征** → Label Encode（Yes→1, No→0, No service→0）
- **3 个多分类特征**（InternetService, Contract, PaymentMethod）→ One-Hot Encode
- **数值特征**（tenure, MonthlyCharges, TotalCharges, SeniorCitizen）→ StandardScaler
- **TotalCharges 清洗**：空字符串→NaN→0（tenure=0 新客户）
- **Train/Test Split**：80/20 分层抽样（stratify）

### 模型
- **Logistic Regression**（sklearn）
- GridSearchCV 5 折交叉验证调参：C ∈ {0.01, 0.1, 1.0, 10.0}
- 评分指标：ROC-AUC
- 正则化：L2（默认）

### 评估指标
- ROC-AUC、Accuracy、Precision、Recall、F1-Score
- 混淆矩阵、分类报告
- 特征系数排序（Odds Ratio）

## 📈 结果解读

### 模型性能

| 指标 | 数值 | 说明 |
|------|------|------|
| **ROC-AUC** | **0.8411** | 良好区分能力 |
| Accuracy | 80.48% | 整体正确率 |
| Precision | 65.62% | 预测流失中真正流失的比例 |
| Recall | 55.61% | 真正流失中被识别的比例 |
| F1-Score | 0.6020 | 精确率与召回率的调和平均 |

### 混淆矩阵

`
                预测留存  预测流失
实际留存          926      109
实际流失          166      208
`

- **TP=208, FP=109, FN=166, TN=926**
- 假阴性 FN=166：漏判的流失客户（最需要关注的业务损失）

### 特征重要性排名（Logistic Regression 系数）

| 排名 | 特征 | 系数 | 胜率比 (Odds Ratio) | 方向 |
|------|------|------|---------------------|------|
| 1 | **MonthlyCharges** | -2.158 | 0.116 | ⬇ 降低流失 |
| 2 | **tenure** | -1.300 | 0.273 | ⬇ 降低流失 |
| 3 | InternetService_No | -1.198 | 0.302 | ⬇ 降低流失 |
| 4 | **InternetService_Fiber optic** | +1.132 | 3.103 | ⬆ 增加流失 |
| 5 | TotalCharges | +0.575 | 1.777 | ⬆ 增加流失 |
| 6 | StreamingMovies | +0.455 | 1.577 | ⬆ 增加流失 |
| 7 | StreamingTV | +0.454 | 1.575 | ⬆ 增加流失 |
| 8 | **Contract_Two year** | -0.324 | 0.723 | ⬇ 降低流失 |
| 9 | MultipleLines | +0.317 | 1.373 | ⬆ 增加流失 |
| 10 | **Contract_Month-to-month** | +0.306 | 1.358 | ⬆ 增加流失 |

> **Odds Ratio 解读：**
> - InternetService_Fiber optic Odds Ratio=3.10 → 相比非光纤用户，流失胜率（odds）是 3.1 倍
> - 	enure Odds Ratio=0.27 → 在网时长每增加 1 个标准差，流失胜率降低 73%

### EDA 关键发现

| 维度 | 高风险特征 | 流失率 | 低风险特征 | 流失率 |
|------|-----------|--------|-----------|--------|
| **合同类型** | Month-to-month | **42.7%** | Two year | 2.8% |
| **互联网服务** | Fiber optic | **41.9%** | DSL | 19.0% |
| **支付方式** | Electronic check | **45.3%** | Bank transfer (auto) | ~15% |
| **技术支持** | 未开通 | **41.6%** | 已开通 | 15.2% |
| **在网时长** | < 12 个月 | ~40% | > 36 个月 | ~10% |

### 业务建议

1. **短期合同客户**（Month-to-month, 流失率 42.7%）→ 推出"签约一年享 85 折"升级优惠，ROI 极高
2. **光纤高价客户**（流失率 41.9%）→ 主动检测网络质量，推送免费技术支持
3. **新客户（tenure < 6 个月）** → 设立专职 onboarding 团队，入职第一周电话回访
4. **Electronic check 支付用户**（流失率 45.3%）→ 推广自动扣款，赠送首月  抵扣
5. **未开通增值服务用户** → 赠送 OnlineSecurity / TechSupport 1 个月免费试用

### 高风险客户画像

> **典型流失客户：** 月付合同 + 光纤上网 + Electronic check 支付 + 无技术支持 + 在网不足 6 个月 + 月费 –100 + 单身无家属

## 🔧 依赖

`
pandas>=0.23.0
numpy>=1.14.0
matplotlib>=2.2.0
seaborn>=0.8.0
scikit-learn>=0.24.0
scipy>=1.1.0
jupyter>=1.0.0
`

## 📝 简历描述参考

> 独立完成电信客户流失预测项目，对 7,043 条客户数据进行数据清洗（TotalCharges 空值处理）、EDA 分析（10+ 张可视化图表）和特征工程（Label Encode / OneHot Encode / StandardScaler），使用 Logistic Regression + GridSearchCV 建模，ROC-AUC 达 0.84。通过模型系数可解释性，识别出合同类型、互联网服务、支付方式为流失核心驱动因素，并给出 5 条可落地的客户挽留策略。

## 📄 License

MIT
