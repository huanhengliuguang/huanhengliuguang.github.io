
# Q2 第二问：生存分析代码重写与功能解释报告

本部分详细阐述我们如何将原始的 Databricks 生存分析项目迁移至本地 Jupyter Notebook，并逐模块解释每段代码的功能和逻辑。

---

## 模块 01：数据加载与清洗（01_intro.py）

📌 **功能概述：**
- 加载原始客户数据（CSV 文件）
- 清理缺失值，转换字段格式
- 生成“银层数据” `silver_customers_csv/part-00000.csv`

### 🧩 核心代码与解释：

```python
df = spark.read.csv("customer_churn.csv", header=True, inferSchema=True)
df = df.dropna(how="any")
```

加载数据并删除包含空值的行，确保后续建模不被错误值影响。

```python
df = df.withColumn("TotalCharges", col("TotalCharges").cast("double"))
df = df.withColumn("SeniorCitizen", col("SeniorCitizen").cast("integer"))
df = df.withColumn("Churn", when(col("Churn") == "Yes", 1).otherwise(0))
```

字段类型转换：
- 将 TotalCharges 转为 double；
- 将 SeniorCitizen 转为整型；
- 将 Churn 转换为二值型（1 表示流失）。

```python
df.coalesce(1).write.option("header", True).csv("silver_customers_csv", mode="overwrite")
```

输出为单个 CSV 文件，作为本地持久化中间层数据。

---

## 模块 02：Kaplan-Meier 生存函数估计（02_kaplan_meier.py）

📌 **功能概述：**
- 估计客户的生存函数
- 绘制不同合同类型下的 KM 曲线
- 使用 log-rank 检验检验合同组之间的生存差异

### 🧩 核心代码与解释：

```python
kmf = KaplanMeierFitter()
kmf.fit(df["tenure"], event_observed=df["Churn"])
```

拟合整体生存函数，使用 tenure 作为持续时间，Churn 表示事件是否发生。

```python
for contract_type in df["Contract"].unique():
    subset = df[df["Contract"] == contract_type]
    kmf.fit(subset["tenure"], subset["Churn"], label=contract_type)
    kmf.plot_survival_function()
```

为每种合同类型绘制单独生存曲线，直观展示哪类合同流失率更高。

```python
logrank_test(group_a["tenure"], group_b["tenure"],
             event_observed_A=group_a["Churn"], event_observed_B=group_b["Churn"]).p_value
```

比较 Month-to-month 与 Two year 客户的流失率差异，得出是否具有统计显著性。

---

## 模块 03：Cox 比例风险模型（03_cox_proportional_hazards.py）

📌 **功能概述：**
- 编码分类变量
- 拟合 Cox 模型
- 输出变量风险比（Hazard Ratio）
- 检查比例风险假设是否成立

### 🧩 核心代码与解释：

```python
df_encoded = pd.get_dummies(df, columns=["dependents", "internetService", "techSupport"], drop_first=False)
```

对分类变量进行 One-Hot 编码，生成 0/1 二值特征用于建模。

```python
cph = CoxPHFitter()
cph.fit(model_df, duration_col="tenure", event_col="Churn")
cph.print_summary()
```

使用 lifelines 的 CoxPHFitter 拟合比例风险模型，输出每个变量对应的风险比（exp(coef)）与置信区间。

```python
cph.check_assumptions(model_df, show_plots=True)
```

自动检测模型是否满足比例风险假设（即风险比随时间恒定）。

---

## 模块 04：AFT 加速失效时间模型（04_accelerated_failure_time.py）

📌 **功能概述：**
- 拟合 Log-Logistic AFT 模型
- 替代 Cox 模型用于解释性更强的场景
- 检查 log(-log(S(t))) 曲线是否为直线

### 🧩 核心代码与解释：

```python
aft = LogLogisticAFTFitter()
aft.fit(model_df, duration_col="tenure", event_col="Churn")
aft.print_summary()
aft.plot()
```

拟合 AFT 模型，AFT 通过对数时间建模，更适合某些不满足比例风险假设的场景。可视化模型中变量对生存时间的加速/减缓效果。若各组线条为近似平行直线，表明使用 AFT 模型是合理的。

---

## 模块 05：客户生命周期价值估算（05_customer_lifetime_value.py）

📌 **功能概述：**
- 利用 Cox 模型预测单个客户的生存概率
- 每月折现利润，估算累计 NPV
- 绘制生存概率曲线与净现值曲线

### 🧩 核心代码与解释：

```python
surv = cph.predict_survival_function(customer_input, times=range(1, 61))
```

预测某个客户（或特定特征组）的生存函数。

```python
result["Expected Monthly Profit"] = result["Survival Probability"] * monthly_profit
result["NPV"] = result["Expected Monthly Profit"] / ((1 + irr_monthly) ** result["Contract Month"])
result["Cumulative NPV"] = result["NPV"].cumsum()
```

每月利润乘以生存概率后折现为净现值（NPV），再计算累计值（CLV）。

```python
sns.lineplot(x="Contract Month", y="Cumulative NPV", data=result)
```

绘制客户价值增长趋势，用于回本周期分析和客户分层。

---

## 总结

本次任务复现了从数据清洗、KM 分析、Cox/AFT 建模，到生命周期价值预测的完整生存分析流程。我们将原始的 Databricks 模块成功转换为本地 Jupyter 环境下可执行的 Python 脚本，并增强了解释性和可视化。
