import os
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

import tensorflow as tf
from tensorflow import keras

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

import numpy as np
import pandas as pd
from flask import Flask, render_template

from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit # 과제에서는 train_test_split 등

np.random.seed(42)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap5 = Bootstrap5(app)

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

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        # 1. 폼에서 입력받은 변수 저장
        longitude      = float(form.longitude.data)
        latitude       = float(form.latitude.data)
        month          = form.month.data
        day            = form.day.data
        avg_temp       = float(form.avg_temp.data)
        max_temp       = float(form.max_temp.data)
        max_wind_speed = float(form.max_wind_speed.data)
        avg_wind       = float(form.avg_wind.data)
        
        # 2. 파형 전처리를 위한 기존 데이터세트 로드 및 파이프라인 구성 (가이드라인 ... 구역)
        fires = pd.read_csv("./sanbul2district-divby100.csv", sep=",")
        fires["burned_area"] = np.log(fires["burned_area"] + 1)
        
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(fires, fires["month"]):
            strat_train_set = fires.loc[train_index]
            
        fires_tr = strat_train_set.drop(["burned_area"], axis=1)
        
        num_attribs = ['longitude','latitude','avg_temp','max_temp','max_wind_speed','avg_wind']
        cat_attribs = ['month','day']
        
        num_pipeline  = Pipeline([('std_scaler', StandardScaler())])
        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])
        full_pipeline.fit(fires_tr)
        
        # 3. 모델 로드 
        model = keras.models.load_model('fires_model.keras')
        
        # 4. 입력 데이터 생성 및 변환
        input_data = pd.DataFrame(
            [[longitude, latitude, avg_temp, max_temp, max_wind_speed, avg_wind, month, day]],
            columns=['longitude','latitude','avg_temp','max_temp','max_wind_speed','avg_wind','month','day']
        )
        input_prepared = full_pipeline.transform(input_data)
        
        # 5. 모델 예측 (메모리 초과 502 에러 방지를 위해 직접 호출 방식 유지)
        prediction = model(input_prepared, training=False)
        result = np.expm1(prediction[0][0])
        
        return render_template('result.html', result=round(float(result), 2))
        
    return render_template('prediction.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)