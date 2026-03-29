# fires_train.py - 단계1 + 단계2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# =============================================
# 단계 1-1: 데이터 불러오기
# =============================================
fires = pd.read_csv("./sanbul2district-divby100.csv", sep=",")

# =============================================
# 단계 1-2: 기본 정보 출력
# =============================================
print("### fires.head():")
print(fires.head())
print("\n### fires.info():")
print(fires.info())
print("\n### fires.describe():")
print(fires.describe())
print("\n### month value_counts:")
print(fires["month"].value_counts())
print("\n### day value_counts:")
print(fires["day"].value_counts())

# =============================================
# 단계 1-3: 데이터 시각화 (히스토그램)
# =============================================
fires.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

# =============================================
# 단계 1-4: 로그 변환
# =============================================
fires["burned_area"].hist()
plt.title("Before log transform")
plt.show()

fires["burned_area"] = np.log(fires["burned_area"] + 1)

fires["burned_area"].hist()
plt.title("After log transform")
plt.show()

# =============================================
# 단계 1-5: Train/Test 분리
# =============================================
train_set, test_set = train_test_split(fires, test_size=0.2, random_state=42)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(fires, fires["month"]):
    strat_train_set = fires.loc[train_index]
    strat_test_set  = fires.loc[test_index]

print("\nTest set 비율 (month):")
print(strat_test_set["month"].value_counts() / len(strat_test_set))
print("\n전체 비율 (month):")
print(fires["month"].value_counts() / len(fires))

# =============================================
# 단계 1-6: Scatter Matrix
# =============================================
from pandas.plotting import scatter_matrix
attrs = ["burned_area", "max_temp", "avg_temp", "max_wind_speed"]
scatter_matrix(fires[attrs], figsize=(10, 8))
plt.show()

# =============================================
# 단계 1-7: 지역별 Plot
# =============================================
fires.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
           s=fires["max_temp"], label="max_temp",
           c="burned_area", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()
plt.show()

# =============================================
# 단계 1-8: OneHotEncoding
# =============================================
fires_tr     = strat_train_set.drop(["burned_area"], axis=1)
fires_labels = strat_train_set["burned_area"].copy()
fires_num    = fires_tr.drop(["month", "day"], axis=1)

cat_encoder = OneHotEncoder()
month_encoded = cat_encoder.fit_transform(fires_tr[["month"]])
print("cat_month_encoder.categories_:", cat_encoder.categories_)

day_encoded = cat_encoder.fit_transform(fires_tr[["day"]])
print("cat_day_encoder.categories_:", cat_encoder.categories_)

# =============================================
# 단계 1-9: Pipeline + StandardScaler
# =============================================
num_attribs = ['longitude','latitude','avg_temp','max_temp','max_wind_speed','avg_wind']
cat_attribs = ['month','day']

num_pipeline  = Pipeline([('std_scaler', StandardScaler())])
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

fires_prepared = full_pipeline.fit_transform(fires_tr)

fires_test        = strat_test_set.drop(["burned_area"], axis=1)
fires_test_labels = strat_test_set["burned_area"].copy()
fires_test_prepared = full_pipeline.transform(fires_test)

print("\nfires_prepared shape:", fires_prepared.shape)

# =============================================
# 단계 2: Keras 모델 개발
# =============================================
X_train, X_valid, y_train, y_valid = train_test_split(
    fires_prepared, fires_labels, test_size=0.2, random_state=42)
X_test, y_test = fires_test_prepared, fires_test_labels

np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(1)
])

model.summary()
model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.SGD(learning_rate=1e-3))
history = model.fit(X_train, y_train, epochs=200,
                    validation_data=(X_valid, y_valid))

# 모델 저장
model.save('fires_model.keras')
print("모델 저장 완료: fires_model.keras")

# 평가
X_new = X_test[:3]
print("예측:", np.round(model.predict(X_new), 2))
print("실제:", np.round(y_test[:3].values, 2))