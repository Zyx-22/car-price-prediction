# car-price-prediction

二手车价格预测系统 - 基于机器学习的完整解决方案

## 项目简介

利用 15 万条真实二手车交易数据训练机器学习模型，对 5 万条测试数据进行价格预测。项目涵盖从数据清洗、特征工程、多模型训练到模型融合的完整流程。

## 核心能力

- 数据清洗与异常值处理
- 多维度特征工程（35+特征）
- 三模型集成融合（LightGBM + XGBoost + CatBoost）
- 5 折交叉验证

## 技术栈

| 组件 | 版本 | 用途 |
|------|------|------|
| Python | 3.9 | 编程语言 |
| Streamlit | 1.12.0 | Web 展示框架 |
| pandas | 1.x | 数据处理 |
| numpy | 1.x | 数值计算 |
| scikit-learn | 1.x | 机器学习工具 |
| LightGBM | 3.x | 梯度提升树 |
| XGBoost | 1.x | 梯度提升树 |
| CatBoost | 1.x | 梯度提升树 |
| plotly | - | 交互式图表 |

## 快速开始

### 本地运行

```bash
# 安装依赖
pip install -r requirements.txt

# 运行 Streamlit 应用
streamlit run app.py

# 或直接运行训练脚本
python solution.py
```

### 云端部署

本项目已配置好 Streamlit Cloud 部署文件，部署步骤：

1. 访问 [https://share.streamlit.io](https://share.streamlit.io)
2. 登录 GitHub 账号
3. 选择仓库 `user/car-price-prediction`
4. 主文件路径设为 `app.py`
5. 点击 Deploy

## 项目结构

```
├── app.py                      # Streamlit 前端展示页面
├── solution.py                 # 完整训练代码（推荐）
├── car_price_prediction.py     # 第一版训练代码
├── requirements.txt            # Python 依赖
├── .streamlit/
│   └── config.toml             # Streamlit 配置
├── project_data.json           # 项目数据统计
├── used_car_sample_submit.csv  # 提交格式示例
└── test_report.md              # 测试报告
```

## 模型效果

| 指标 | 数值 |
|------|------|
| 训练样本 | 150,000 |
| 测试样本 | 50,000 |
| 特征数量 | 35+ |
| 预测均价 | ¥5,810 |
| 融合策略 | LightGBM 40% + XGBoost 30% + CatBoost 30% |

## License

This project is for educational purposes.
