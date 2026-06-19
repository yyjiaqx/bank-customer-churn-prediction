# 📊 电信客户流失预测分析

**Telco Customer Churn Prediction** — 多模型客户流失预测，面向银行数据分析/金融科技岗位求职。

[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-0.24+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-green.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.44+-red.svg)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 项目简介

对比 **Logistic Regression / Random Forest / XGBoost** 三种模型进行客户流失预测，通过 SMOTE 处理类别不平衡，GridSearchCV 调参，SHAP 可解释性分析。LR 提供系数解读，树模型提供特征重要性——兼顾**预测精度**与**业务可解释性**。

**核心目标：** 识别高风险流失客户，量化各因素影响，输出可落地的业务建议。

## 📊 数据集

| 指标 | 数值 |
|------|------|
| 总客户数 | 7,043 |
| 流失客户 | 1,869 (26.54%) |
| 留存客户 | 5,174 (73.46%) |
| 原始特征 | 20（含目标） |

| 特征 | 说明 | 处理 |
|------|------|------|
| customerID | 客户ID | **删除** |
| gender | 性别 | Label Encode |
| SeniorCitizen | 是否老年人 | 保持 |
| Partner | 是否有伴侣 | Label Encode |
| Dependents | 是否有家属 | Label Encode |
| tenure | 在网月数 | StandardScaler |
| PhoneService | 电话服务 | Label Encode |
| MultipleLines | 多线服务 | Label Encode |
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

## 🏗️ 项目结构

```
bank-customer-churn-prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── churn_analysis.ipynb
├── src/
│   ├── data_loader.py           ← EDA 图表生成
│   ├── preprocessing.py         ← 特征工程 + SMOTE
│   ├── model.py                 ← LR/RF/XGBoost + SHAP
│   └── evaluation.py            ← 多模型评估对比
├── outputs/figures/             ← 14+ 张分析图表
├── main.py                      ← 一键运行全管线
├── README.md
├── requirements.txt
└── .gitignore
```

## 🚀 快速开始

```bash
cd bank-customer-churn-prediction
pip install -r requirements.txt
python main.py
```

## 🧠 方法

### 特征工程
- 11 个二分类特征 → Label Encode
- 3 个多分类特征 → One-Hot Encode
- 数值特征 → StandardScaler
- TotalCharges 空值 → 0（tenure=0 新客户）

### 类别平衡
- **SMOTE** 过采样，将流失:未流失从 ~1:3 平衡至 ~1:1

### 模型
- **Logistic Regression** — GridSearchCV 调 C，5 折 CV，评分 ROC-AUC
- **Random Forest** — 调 n_estimators/max_depth/min_samples_split/class_weight
- **XGBoost** — 调 n_estimators/max_depth/learning_rate/subsample，自动 scale_pos_weight

### 可解释性
- **LR 系数分析** — 系数 + 胜率比 (Odds Ratio)，业务可解读
- **特征重要性** — RF/XGBoost 的 feature_importances_
- **SHAP** — Summary/Bar/Waterfall/Dependence 四图，单样本预测拆解

### 评估
- ROC-AUC / Accuracy / Precision / Recall / F1-Score
- 多模型混淆矩阵并排对比、ROC/PR 曲线叠加
- 模型指标柱状图对比

## 📈 结果解读

### 模型性能对比（SMOTE 平衡后，测试集 1,409 样本）

| 模型 | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------|----------|-----------|--------|----------|---------|
| Logistic Regression 🏆 | 73.46% | 50.00% | **79.14%** | 0.6128 | **0.8403** |
| Random Forest | 76.79% | 55.75% | 60.96% | 0.5824 | 0.8287 |
| XGBoost | 78.14% | 58.55% | 60.43% | 0.5947 | 0.8275 |

> **LR 召回率最优**（79.14%），适合"宁可多预警、不能漏客户"的流失挽留场景。对比无 SMOTE 的旧版 LR（Recall 55.61%），SMOTE 提升了 23.5 个百分点。

### 混淆矩阵（Logistic Regression）

```
              预测留存  预测流失
实际留存        739      296
实际流失         78      296
```

- TP=296, FP=296, FN=78, TN=739
- 374 个真实流失客户中，成功识别 296 个（79.14%），漏判 78 个

### 特征重要性（Random Forest）

| 排名 | 特征 | 重要性 |
|------|------|--------|
| 1 | tenure | 0.1297 |
| 2 | MonthlyCharges | 0.1211 |
| 3 | Contract_Month-to-month | 0.1192 |
| 4 | TotalCharges | 0.1181 |
| 5 | Contract_Two year | 0.0507 |

### 特征重要性（XGBoost）

| 排名 | 特征 | 重要性 |
|------|------|--------|
| 1 | Contract_Month-to-month | 0.4792 |
| 2 | InternetService_Fiber optic | 0.1700 |
| 3 | InternetService_No | 0.0316 |
| 4 | Contract_One year | 0.0299 |
| 5 | StreamingMovies | 0.0239 |

> XGBoost 将 48% 的重要性集中在「月付合同」上，与 EDA 发现的月付流失率 42.7% 高度吻合。

### EDA 关键发现

| 维度 | 高风险特征 | 流失率 | 低风险特征 | 流失率 |
|------|-----------|--------|-----------|--------|
| **合同类型** | Month-to-month | **42.7%** | Two year | 2.8% |
| **互联网服务** | Fiber optic | **41.9%** | DSL | 19.0% |
| **支付方式** | Electronic check | **45.3%** | Bank transfer (auto) | ~15% |
| **技术支持** | 未开通 | **41.6%** | 已开通 | 15.2% |
| **在网时长** | < 12 个月 | ~40% | > 36 个月 | ~10% |

### 业务建议

1. **短期合同客户**（Month-to-month, 流失率 42.7%）→ 推出"签约一年享 85 折"升级优惠
2. **光纤高价客户**（流失率 41.9%）→ 主动检测网络质量，推送免费技术支持
3. **新客户（tenure < 6 个月）** → 设立专职 onboarding 团队，入职第一周电话回访
4. **Electronic check 支付用户**（流失率 45.3%）→ 推广自动扣款，赠送首月抵扣
5. **未开通增值服务用户** → 赠送 OnlineSecurity / TechSupport 1 个月免费试用

### 高风险客户画像

> **典型流失客户：** 月付合同 + 光纤上网 + Electronic check 支付 + 无技术支持 + 在网不足 6 个月 + 单身无家属

## 🔧 依赖

```
pandas>=0.23.0
numpy>=1.14.0
matplotlib>=2.2.0
seaborn>=0.8.0
scikit-learn>=0.24.0
scipy>=1.1.0
xgboost>=1.5.0
imbalanced-learn>=0.9.0
shap>=0.44.0
jupyter>=1.0.0
```

## 📝 简历描述参考

> 独立完成电信客户流失预测项目，对 7,043 条客户数据进行清洗与 EDA 分析，通过 SMOTE 处理类别不平衡，对比 LR / RF / XGBoost 三种模型 + 5 折 GridSearchCV 调优，LR + SMOTE 召回率提升至 79.14%（较基线 +23.5pp），结合特征重要性与 SHAP 进行可解释性分析，输出分层挽留策略。

## 📄 License

MIT
