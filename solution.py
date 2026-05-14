"""
二手车价格预测 - 完整解决方案
包含：数据加载、预处理、特征工程、模型训练、预测和结果输出
输出格式与 used_car_sample_submit.csv 完全一致
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
import os

print("="*80)
print("二手车价格预测系统 v2.0")
print("="*80)

# ==================== 1. 数据加载 ====================
print("\n[步骤1] 加载数据...")

# 解压并加载训练集
if not os.path.exists('train_data'):
    import zipfile
    with zipfile.ZipFile('used_car_train_20200313.zip', 'r') as zip_ref:
        zip_ref.extractall('train_data')
    print("训练集解压完成")

train = pd.read_csv('train_data/used_car_train_20200313.csv', sep=' ')
print(f"训练集形状: {train.shape}")
print(f"训练集列名: {list(train.columns)}")

# 解压并加载测试集B
if not os.path.exists('testB_data'):
    import zipfile
    with zipfile.ZipFile('used_car_testB_20200421.zip', 'r') as zip_ref:
        zip_ref.extractall('testB_data')
    print("测试集B解压完成")

test = pd.read_csv('testB_data/used_car_testB_20200421.csv', sep=' ')
print(f"测试集形状: {test.shape}")

# 保存SaleID用于提交
test_ids = test['SaleID'].copy()

# 查看目标变量分布
print(f"\n训练集价格统计:")
print(f"  均值: {train['price'].mean():.2f}")
print(f"  中位数: {train['price'].median():.2f}")
print(f"  标准差: {train['price'].std():.2f}")
print(f"  最小值: {train['price'].min():.2f}")
print(f"  最大值: {train['price'].max():.2f}")

# ==================== 2. 数据探索 ====================
print("\n[步骤2] 数据探索...")

# 检查缺失值
print("\n训练集缺失值统计:")
missing = train.isnull().sum()
missing_pct = (missing / len(train) * 100).round(2)
missing_df = pd.DataFrame({'缺失数': missing[missing > 0], '缺失率%': missing_pct[missing > 0]})
if len(missing_df) > 0:
    print(missing_df.sort_values('缺失率%', ascending=False))
else:
    print("无明显缺失值（'-'符号需要特殊处理）")

# 检查notRepairedDamage中的'-'符号
if 'notRepairedDamage' in train.columns:
    dash_count = (train['notRepairedDamage'] == '-').sum()
    print(f"\nnotRepairedDamage中'-'符号数量: {dash_count}")

# ==================== 3. 数据预处理 ====================
print("\n[步骤3] 数据预处理...")

def preprocess(train_df, test_df):
    """
    数据预处理和特征工程
    返回处理后的训练集和测试集
    """
    
    # 添加标识列
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    test_df['price'] = 0  # 测试集添加price列
    
    # 合并数据进行统一处理
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    print(f"合并后数据形状: {combined.shape}")
    
    # --- 3.1 处理特殊缺失值 ---
    # notRepairedDamage: '-'表示缺失
    combined['notRepairedDamage'] = combined['notRepairedDamage'].replace('-', np.nan)
    combined['notRepairedDamage'] = combined['notRepairedDamage'].astype(float)
    
    # --- 3.2 数值型特征缺失值填充 ---
    numeric_features = ['bodyType', 'fuelType', 'gearbox', 'power', 'kilometer', 
                       'notRepairedDamage', 'model', 'brand', 'regionCode']
    
    for col in numeric_features:
        if col in combined.columns:
            median_val = combined[col].median()
            combined[col] = combined[col].fillna(median_val)
            print(f"  {col}: 填充中位数 {median_val:.2f}")
    
    # --- 3.3 特征工程 ---
    print("\n开始特征工程...")
    
    # 3.3.1 时间特征 - 从regDate和creatDate提取
    combined['regDate_str'] = combined['regDate'].astype(str)
    combined['creatDate_str'] = combined['creatDate'].astype(str)
    
    # 提取年份和月份
    combined['reg_year'] = combined['regDate_str'].str[:4].astype(int)
    combined['reg_month'] = combined['regDate_str'].str[4:6].astype(int)
    combined['reg_day'] = combined['regDate_str'].str[6:8].astype(int)
    
    combined['creat_year'] = combined['creatDate_str'].str[:4].astype(int)
    combined['creat_month'] = combined['creatDate_str'].str[4:6].astype(int)
    combined['creat_day'] = combined['creatDate_str'].str[6:8].astype(int)
    
    # 计算车龄（年）
    combined['car_age'] = combined['creat_year'] - combined['reg_year']
    # 处理异常车龄（负数或过大）
    combined.loc[combined['car_age'] < 0, 'car_age'] = combined.loc[combined['car_age'] < 0, 'car_age'].abs()
    combined.loc[combined['car_age'] > 50, 'car_age'] = 50  # 上限设为50年
    
    # 计算总月数
    combined['total_months'] = combined['car_age'] * 12 + (combined['creat_month'] - combined['reg_month'])
    combined.loc[combined['total_months'] < 0, 'total_months'] = combined.loc[combined['total_months'] < 0, 'total_months'].abs()
    
    # 3.3.2 Power特征处理
    # power为0可能是异常值，用中位数填充
    power_median = combined.loc[combined['power'] > 0, 'power'].median()
    combined.loc[combined['power'] == 0, 'power'] = power_median
    # power过大的异常值处理
    combined['power'] = combined['power'].clip(upper=600)  # 设置上限
    
    # 创建power分箱特征
    combined['power_bin'] = pd.cut(combined['power'], 
                                    bins=[0, 50, 100, 150, 200, 300, 600], 
                                    labels=False).fillna(0).astype(int)
    
    # 3.3.3 Kilometer特征处理
    combined['kilometer_bin'] = pd.cut(combined['kilometer'], 
                                        bins=[0, 5, 10, 15, 20, 30], 
                                        labels=False).fillna(0).astype(int)
    
    # 3.3.4 交互特征
    combined['power_per_age'] = combined['power'] / (combined['car_age'] + 1)
    combined['km_per_age'] = combined['kilometer'] / (combined['car_age'] + 1)
    combined['power_x_km'] = combined['power'] * combined['kilometer']
    combined['power_x_age'] = combined['power'] * combined['car_age']
    
    # 3.3.5 品牌统计特征（使用训练集统计）
    train_mask = combined['is_train'] == 1
    brand_stats = combined[train_mask].groupby('brand')['price'].agg(['mean', 'std', 'count']).reset_index()
    brand_stats.columns = ['brand', 'brand_price_mean', 'brand_price_std', 'brand_count']
    brand_stats['brand_price_std'] = brand_stats['brand_price_std'].fillna(0)
    combined = combined.merge(brand_stats, on='brand', how='left')
    
    # 填充测试集的品牌统计特征
    combined['brand_price_mean'] = combined['brand_price_mean'].fillna(combined['brand_price_mean'].median())
    combined['brand_price_std'] = combined['brand_price_std'].fillna(0)
    combined['brand_count'] = combined['brand_count'].fillna(1)
    
    # 3.3.6 车型统计特征
    model_stats = combined[train_mask].groupby('model')['price'].agg(['mean', 'std']).reset_index()
    model_stats.columns = ['model', 'model_price_mean', 'model_price_std']
    model_stats['model_price_std'] = model_stats['model_price_std'].fillna(0)
    combined = combined.merge(model_stats, on='model', how='left')
    
    combined['model_price_mean'] = combined['model_price_mean'].fillna(combined['model_price_mean'].median())
    combined['model_price_std'] = combined['model_price_std'].fillna(0)
    
    # 3.3.7 地区统计特征
    region_stats = combined[train_mask].groupby('regionCode')['price'].agg(['mean', 'count']).reset_index()
    region_stats.columns = ['regionCode', 'region_price_mean', 'region_count']
    combined = combined.merge(region_stats, on='regionCode', how='left')
    
    combined['region_price_mean'] = combined['region_price_mean'].fillna(combined['region_price_mean'].median())
    combined['region_count'] = combined['region_count'].fillna(1)
    
    # 3.3.8 V特征统计
    v_cols = [f'v_{i}' for i in range(15)]
    combined['v_mean'] = combined[v_cols].mean(axis=1)
    combined['v_std'] = combined[v_cols].std(axis=1)
    combined['v_max'] = combined[v_cols].max(axis=1)
    combined['v_min'] = combined[v_cols].min(axis=1)
    combined['v_range'] = combined['v_max'] - combined['v_min']
    combined['v_skew'] = combined[v_cols].skew(axis=1)
    
    # 3.3.9 对数变换目标变量（减少偏态分布影响）
    combined['price_log'] = np.log1p(combined['price'])
    
    print(f"特征工程完成！总特征数: {len(combined.columns)}")
    
    return combined

# 执行预处理
combined = preprocess(train, test)

# ==================== 4. 准备训练数据 ====================
print("\n[步骤4] 准备训练和测试数据...")

# 分离训练集和测试集
train_processed = combined[combined['is_train'] == 1].copy()
test_processed = combined[combined['is_train'] == 0].copy()

print(f"训练集样本数: {len(train_processed)}")
print(f"测试集样本数: {len(test_processed)}")

# 定义特征列
exclude_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'price', 'is_train', 
                'price_log', 'regDate_str', 'creatDate_str']
feature_cols = [col for col in combined.columns if col not in exclude_cols]

print(f"使用的特征数量: {len(feature_cols)}")
print(f"特征列表前20个: {feature_cols[:20]}")

# 准备X和y
X_train = train_processed[feature_cols].values
y_train = train_processed['price'].values
y_train_log = train_processed['price_log'].values

X_test = test_processed[feature_cols].values

# 处理无穷大和NaN
X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

# 替换y中的异常值
y_train_log = np.nan_to_num(y_train_log, nan=0, posinf=0, neginf=0)

# ==================== 5. 模型训练 ====================
print("\n[步骤5] 训练模型...")

# 交叉验证设置
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
    'random_state': 42,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1
}

lgb_model = lgb.LGBMRegressor(**lgb_params)

# 交叉验证
lgb_scores = cross_val_score(lgb_model, X_train, y_train_log, cv=kf, 
                             scoring='neg_mean_absolute_error')
print(f"LightGBM 5折CV MAE: {-lgb_scores.mean():.4f} (+/- {lgb_scores.std() * 2:.4f})")

# 在全部训练集上训练
lgb_model.fit(X_train, y_train_log)
lgb_pred_log = lgb_model.predict(X_test)
lgb_pred = np.expm1(lgb_pred_log)

# 特征重要性
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 重要特征:")
print(importance.head(20).to_string(index=False))

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
    'verbosity': 0,
    'min_child_weight': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1
}

xgb_model = xgb.XGBRegressor(**xgb_params)

xgb_scores = cross_val_score(xgb_model, X_train, y_train_log, cv=kf,
                             scoring='neg_mean_absolute_error')
print(f"XGBoost 5折CV MAE: {-xgb_scores.mean():.4f} (+/- {xgb_scores.std() * 2:.4f})")

xgb_model.fit(X_train, y_train_log)
xgb_pred_log = xgb_model.predict(X_test)
xgb_pred = np.expm1(xgb_pred_log)

# --- 模型3: CatBoost (如果安装成功) ---
try:
    from catboost import CatBoostRegressor
    print("\n训练 CatBoost 模型...")
    
    cat_model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.05,
        depth=8,
        loss_function='MAE',
        verbose=False,
        random_seed=42,
        l2_leaf_reg=3
    )
    
    cat_scores = cross_val_score(cat_model, X_train, y_train_log, cv=kf,
                                 scoring='neg_mean_absolute_error')
    print(f"CatBoost 5折CV MAE: {-cat_scores.mean():.4f} (+/- {cat_scores.std() * 2:.4f})")
    
    cat_model.fit(X_train, y_train_log)
    cat_pred_log = cat_model.predict(X_test)
    cat_pred = np.expm1(cat_pred_log)
    
    use_catboost = True
    print("CatBoost 训练完成")
except Exception as e:
    print(f"\nCatBoost 训练跳过: {str(e)}")
    use_catboost = False

# ==================== 6. 模型融合 ====================
print("\n[步骤6] 模型融合...")

if use_catboost:
    # 三个模型加权平均
    weights = [0.4, 0.3, 0.3]  # LightGBM, XGBoost, CatBoost
    final_pred = weights[0] * lgb_pred + weights[1] * xgb_pred + weights[2] * cat_pred
    print(f"使用3个模型融合: LightGBM(40%) + XGBoost(30%) + CatBoost(30%)")
else:
    # 两个模型加权平均
    weights = [0.5, 0.5]  # LightGBM, XGBoost
    final_pred = weights[0] * lgb_pred + weights[1] * xgb_pred
    print(f"使用2个模型融合: LightGBM(50%) + XGBoost(50%)")

# 确保预测值为非负
final_pred = np.maximum(final_pred, 0)

# 打印预测统计
print(f"\n预测结果统计:")
print(f"  均值: {final_pred.mean():.2f}")
print(f"  中位数: {np.median(final_pred):.2f}")
print(f"  标准差: {final_pred.std():.2f}")
print(f"  最小值: {final_pred.min():.2f}")
print(f"  最大值: {final_pred.max():.2f}")
print(f"  小于0的数量: {(final_pred < 0).sum()}")

# ==================== 7. 生成提交文件 ====================
print("\n[步骤7] 生成提交文件...")

# 创建提交DataFrame，格式与sample_submit.csv完全一致
submit = pd.DataFrame({
    'SaleID': test_ids.values,
    'price': final_pred
})

# 保存为CSV
output_file = 'used_car_submit.csv'
submit.to_csv(output_file, index=False)

print(f"\n✓ 提交文件已生成: {output_file}")
print(f"  行数: {len(submit)}")
print(f"  列名: {list(submit.columns)}")

# 显示前10行
print(f"\n前10行预览:")
print(submit.head(10).to_string(index=False))

# 验证格式是否与sample一致
sample = pd.read_csv('used_car_sample_submit.csv')
print(f"\n格式验证:")
print(f"  sample_submit列名: {list(sample.columns)}")
print(f"  submit列名: {list(submit.columns)}")
print(f"  列名一致: {list(sample.columns) == list(submit.columns)}")
print(f"  sample行数: {len(sample)}")
print(f"  submit行数: {len(submit)}")

print("\n" + "="*80)
print("预测完成！")
print("="*80)
print(f"\n生成的文件: {output_file}")
print("可以直接上传到竞赛平台提交。")
