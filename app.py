# sanbul-pwa-flask.py
import os
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from flask import Flask, render_template
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

np.random.seed(42)

# ── 데이터 & 파이프라인 재구성 (모델 로드 전 필요) ──
fires = pd.read_csv("./sanbul2district-divby100.csv", sep=",")
fires["burned_area"] = np.log(fires["burned_area"] + 1)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(fires, fires["month"]):
    strat_train_set = fires.loc[train_index]
    strat_test_set  = fires.loc[test_index]

fires_tr     = strat_train_set.drop(["burned_area"], axis=1)
fires_labels = strat_train_set["burned_area"].copy()

num_attribs = ['longitude','latitude','avg_temp','max_temp','max_wind_speed','avg_wind']
cat_attribs = ['month','day']

num_pipeline  = Pipeline([('std_scaler', StandardScaler())])
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])
full_pipeline.fit(fires_tr)

# ── 모델 로드 ──
model = keras.models.load_model('fires_model.keras')

# --- Dummy Predict (웜업: 첫 연결 전 가짜 데이터로 한 번 구동) ---
print("모델 웜업 시작...")
dummy_data = pd.DataFrame(
    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, '01-Jan', '00-sun']],
    columns=['longitude','latitude','avg_temp','max_temp','max_wind_speed','avg_wind','month','day']
)
dummy_prepared = full_pipeline.transform(dummy_data)
model(dummy_prepared, training=False) # 직접 호출 방식으로 변경하여 메모리와 속도 최적화
print("모델 웜업 완료!")

# ── Flask 앱 설정 ──
app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap5 = Bootstrap5(app)

# ── 폼 정의 ──
class LabForm(FlaskForm):
    longitude     = StringField('longitude(1-7)',              validators=[DataRequired()])
    latitude      = StringField('latitude(1-7)',               validators=[DataRequired()])
    month         = StringField('month(01-Jan ~ Dec-12)',      validators=[DataRequired()])
    day           = StringField('day(00-sun ~ 06-sat, 07-hol)',validators=[DataRequired()])
    avg_temp      = StringField('avg_temp',                    validators=[DataRequired()])
    max_temp      = StringField('max_temp',                    validators=[DataRequired()])
    max_wind_speed= StringField('max_wind_speed',              validators=[DataRequired()])
    avg_wind      = StringField('avg_wind',                    validators=[DataRequired()])
    submit        = SubmitField('Submit')

# ── 라우트 ──
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        longitude      = float(form.longitude.data)
        latitude       = float(form.latitude.data)
        month          = form.month.data
        day            = form.day.data
        avg_temp       = float(form.avg_temp.data)
        max_temp       = float(form.max_temp.data)
        max_wind_speed = float(form.max_wind_speed.data)
        avg_wind       = float(form.avg_wind.data)

        input_data = pd.DataFrame(
    [[longitude, latitude, avg_temp, max_temp, max_wind_speed, avg_wind, month, day]],
    columns=['longitude','latitude','avg_temp','max_temp','max_wind_speed','avg_wind','month','day']
)

        input_prepared = full_pipeline.transform(input_data)
        prediction     = model(input_prepared, training=False)
        result         = np.expm1(prediction[0][0])

        return render_template('result.html', result=round(float(result), 2))
    return render_template('prediction.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)

    