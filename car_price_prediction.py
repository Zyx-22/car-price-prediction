"""
二手车价格预测 - 完整机器学习流程
包括：数据加载、预处理、特征工程、模型训练、预测和结果输出
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import os

print("=" * 80)
print("二手车价格预测系统")
print("=" * 80)

# ==================== 1. 数据加载 ====================
print("\n[1/6] 加载数据...")

# 加载训练集
train_data = pd.read_csv('train_data/used_car_train_20200313.csv', sep=' ')
print(f"训练集形状: {train_data.shape}")
print(f"训练集列名: {train_data.columns.tolist()}")

# 加载测试集A
testA_data = pd.read_csv('testA_data/used_car_testA_20200313.csv', sep=' ')
print(f"测试集A形状: {testA_data.shape}")

# 加载测试集B
testB_data = pd.read_csv('testB_data/used_car_testB_20200421.csv', sep=' ')
print(f"测试集B形状: {testB_data.shape}")

# 保存SaleID用于后续提交
testA_ids = testA_data['SaleID']
testB_ids = testB_data['SaleID']

# ==================== 2. 数据探索与预处理 ====================
print("\n[2/6] 数据预处理...")

def preprocess_data(train, testA, testB):
    """数据预处理函数"""
    
    # 合并训练集和测试集进行统一的特征工程
    train['is_train'] = 1
    testA['is_train'] = 0
    testB['is_train'] = 0
    
    # 为测试集添加price列（填充0）
    if 'price' not in testA.columns:
        testA['price'] = 0
    if 'price' not in testB.columns:
        testB['price'] = 0
    
    combined = pd.concat([train, testA, testB], axis=0, ignore_index=True)
    print(f"合并后数据形状: {combined.shape}")
    
    # --- 处理缺失值 ---
    # notRepairedDamage: '-'表示缺失，替换为NaN
    combined['notRepairedDamage'] = combined['notRepairedDamage'].replace('-', np.nan)
    combined['notRepairedDamage'] = combined['notRepairedDamage'].astype(float)
    
    # 填充数值型特征的缺失值
    numeric_cols = ['bodyType', 'fuelType', 'gearbox', 'power', 'kilometer', 
                    'notRepairedDamage', 'model', 'brand', 'regionCode']
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = combined[col].fillna(combined[col].median())
    
    # --- 特征工程 ---
    
    # 1. 从regDate计算车龄
    combined['regDate_year'] = combined['regDate'].astype(str).str[:4].astype(int)
    combined['regDate_month'] = combined['regDate'].astype(str).str[4:6].astype(int)
    combined['creatDate_year'] = combined['creatDate'].astype(str).str[:4].astype(int)
    combined['creatDate_month'] = combined['creatDate'].astype(str).str[4:6].astype(int)
    
    # 计算车龄（年）
    combined['car_age'] = combined['creatDate_year'] - combined['regDate_year']
    # 修正负数车龄（异常数据）
    combined.loc[combined['car_age'] < 0, 'car_age'] = combined.loc[combined['car_age'] < 0, 'car_age'].abs()
    
    # 计算注册到创建的月份差
    combined['month_diff'] = (combined['creatDate_year'] - combined['regDate_year']) * 12 + \
                            (combined['creatDate_month'] - combined['regDate_month'])
    combined.loc[combined['month_diff'] < 0, 'month_diff'] = combined.loc[combined['month_diff'] < 0, 'month_diff'].abs()
    
    # 2. power特征处理
    # power为0或异常大的值可能是异常值
    combined['power_bin'] = pd.cut(combined['power'], bins=[-1, 50, 100, 150, 200, 300, 500, 1000, 10000], labels=False)
    
    # 3. kilometer特征处理
    combined['kilometer_bin'] = pd.cut(combined['kilometer'], bins=[0, 5, 10, 15, 20, 30], labels=False)
    
    # 4. 创建交互特征
    combined['power_per_age'] = combined['power'] / (combined['car_age'] + 1)
    combined['km_per_age'] = combined['kilometer'] / (combined['car_age'] + 1)
    combined['power_x_km'] = combined['power'] * combined['kilometer']
    
    # 5. 品牌相关特征
    brand_stats = combined.groupby('brand')['price'].agg(['mean', 'std', 'count']).reset_index()
    brand_stats.columns = ['brand', 'brand_price_mean', 'brand_price_std', 'brand_count']
    combined = combined.merge(brand_stats, on='brand', how='left')
    
    # 6. 车型相关特征
    model_stats = combined.groupby('model')['price'].agg(['mean', 'std']).reset_index()
    model_stats.columns = ['model', 'model_price_mean', 'model_price_std']
    combined = combined.merge(model_stats, on='model', how='left')
    
    # 7. 地区编码统计特征
    region_stats = combined.groupby('regionCode')['price'].agg(['mean', 'count']).reset_index()
    region_stats.columns = ['regionCode', 'region_price_mean', 'region_count']
    combined = combined.merge(region_stats, on='regionCode', how='left')
    
    # 8. v_0到v_14的统计特征
    v_cols = [f'v_{i}' for i in range(15)]
    combined['v_mean'] = combined[v_cols].mean(axis=1)
    combined['v_std'] = combined[v_cols].std(axis=1)
    combined['v_max'] = combined[v_cols].max(axis=1)
    combined['v_min'] = combined[v_cols].min(axis=1)
    combined['v_range'] = combined['v_max'] - combined['v_min']
    
    # 9. 对数变换目标变量（减少偏态）
    combined['price_log'] = np.log1p(combined['price'])
    
    print("特征工程完成！")
    print(f"总特征数: {len(combined.columns)}")
    
    return combined

combined_data = preprocess_data(train_data, testA_data, testB_data)

# ==================== 3. 准备训练和测试数据 ====================
print("\n[3/6] 准备训练和测试数据...")

# 分离训练集和测试集
train = combined_data[combined_data['is_train'] == 1].copy()
testA = combined_data[(combined_data['is_train'] == 0) & (combined_data['SaleID'].isin(testA_ids))].copy()
testB = combined_data[(combined_data['is_train'] == 0) & (combined_data['SaleID'].isin(testB_ids))].copy()

print(f"训练集样本数: {len(train)}")
print(f"测试集A样本数: {len(testA)}")
print(f"测试集B样本数: {len(testB)}")

# 定义特征列（排除不需要的列）
exclude_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'price', 'is_train', 'price_log']
feature_cols = [col for col in combined_data.columns if col not in exclude_cols]

print(f"使用的特征数: {len(feature_cols)}")
print(f"特征列表: {feature_cols[:10]}...")  # 打印前10个特征

# 准备X和y
X_train = train[feature_cols].values
y_train = train['price'].values
y_train_log = train['price_log'].values

X_testA = testA[feature_cols].values
X_testB = testB[feature_cols].values

# 处理无穷大和NaN值
X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
X_testA = np.nan_to_num(X_testA, nan=0, posinf=0, neginf=0)
X_testB = np.nan_to_num(X_testB, nan=0, posinf=0, neginf=0)

# ==================== 4. 模型训练 ====================
print("\n[4/6] 训练模型...")

# 定义交叉验证策略
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# --- 模型1: LightGBM ---
print("\n训练 LightGBM 模型...")
lgb_params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_estimators': 3000,
    'random_state': 42
}

lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_scores = cross_val_score(lgb_model, X_train, y_train_log, cv=kf, 
                             scoring='neg_mean_absolute_error')
print(f"LightGBM CV MAE: {-lgb_scores.mean():.4f} (+/- {lgb_scores.std() * 2:.4f})")

# 在整个训练集上训练
lgb_model.fit(X_train, y_train_log)
lgb_pred_A_log = lgb_model.predict(X_testA)
lgb_pred_B_log = lgb_model.predict(X_testB)

# 转换回原始尺度
lgb_pred_A = np.expm1(lgb_pred_A_log)
lgb_pred_B = np.expm1(lgb_pred_B_log)

# --- 模型2: XGBoost ---
print("\n训练 XGBoost 模型...")
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 3000,
    'random_state': 42,
    'verbosity': 0
}

xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_scores = cross_val_score(xgb_model, X_train, y_train_log, cv=kf,
                             scoring='neg_mean_absolute_error')
print(f"XGBoost CV MAE: {-xgb_scores.mean():.4f} (+/- {xgb_scores.std() * 2:.4f})")

xgb_model.fit(X_train, y_train_log)
xgb_pred_A_log = xgb_model.predict(X_testA)
xgb_pred_B_log = xgb_model.predict(X_testB)

xgb_pred_A = np.expm1(xgb_pred_A_log)
xgb_pred_B = np.expm1(xgb_pred_B_log)

# --- 模型3: CatBoost ---
print("\n训练 CatBoost 模型...")
cat_model = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.05,
    depth=8,
    loss_function='MAE',
    verbose=False,
    random_seed=42
)

cat_scores = cross_val_score(cat_model, X_train, y_train_log, cv=kf,
                             scoring='neg_mean_absolute_error')
print(f"CatBoost CV MAE: {-cat_scores.mean():.4f} (+/- {cat_scores.std() * 2:.4f})")

cat_model.fit(X_train, y_train_log)
cat_pred_A_log = cat_model.predict(X_testA)
cat_pred_B_log = cat_model.predict(X_testB)

cat_pred_A = np.expm1(cat_pred_A_log)
cat_pred_B = np.expm1(cat_pred_B_log)

# ==================== 5. 模型融合 ====================
print("\n[5/6] 模型融合...")

# 简单加权平均（根据CV表现调整权重）
# 通常三个模型表现接近时，可以使用等权重
weights = [0.4, 0.3, 0.3]  # LightGBM, XGBoost, CatBoost

final_pred_A = weights[0] * lgb_pred_A + weights[1] * xgb_pred_A + weights[2] * cat_pred_A
final_pred_B = weights[0] * lgb_pred_B + weights[1] * xgb_pred_B + weights[2] * cat_pred_B

# 确保预测值为非负数
final_pred_A = np.maximum(final_pred_A, 0)
final_pred_B = np.maximum(final_pred_B, 0)

print(f"测试集A预测价格统计:")
print(f"  均值: {final_pred_A.mean():.2f}")
print(f"  中位数: {np.median(final_pred_A):.2f}")
print(f"  最小值: {final_pred_A.min():.2f}")
print(f"  最大值: {final_pred_A.max():.2f}")

print(f"\n测试集B预测价格统计:")
print(f"  均值: {final_pred_B.mean():.2f}")
print(f"  中位数: {np.median(final_pred_B):.2f}")
print(f"  最小值: {final_pred_B.min():.2f}")
print(f"  最大值: {final_pred_B.max():.2f}")

# ==================== 6. 生成提交文件 ====================
print("\n[6/6] 生成提交文件...")

# 生成测试集A的提交文件
submit_A = pd.DataFrame({
    'SaleID': testA_ids.values,
    'price': final_pred_A
})
submit_A.to_csv('used_car_testA_predict.csv', index=False)
print(f"测试集A预测结果已保存: used_car_testA_predict.csv")
print(f"  行数: {len(submit_A)}")

# 生成测试集B的提交文件
submit_B = pd.DataFrame({
    'SaleID': testB_ids.values,
    'price': final_pred_B
})
submit_B.to_csv('used_car_testB_predict.csv', index=False)
print(f"测试集B预测结果已保存: used_car_testB_predict.csv")
print(f"  行数: {len(submit_B)}")

# 打印示例预测
print("\n测试集A前10条预测结果:")
print(submit_A.head(10).to_string(index=False))

print("\n" + "=" * 80)
print("预测完成！")
print("=" * 80)
print("\n生成的文件:")
print("  1. used_car_testA_predict.csv - 测试集A的预测结果")
print("  2. used_car_testB_predict.csv - 测试集B的预测结果")
print("\n提交文件格式与 sample_submit.csv 一致，可直接上传到竞赛平台。")
