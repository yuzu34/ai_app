import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. アプリのタイトル
st.title('🚢 海難事故の生存確率予測アプリ')
st.write('もしあなたが歴史的な客船に乗って遭難したら、生存できた確率はどれくらいでしょうか？AIが予測します。')

# 2. データの読み込みとAIモデルの学習
@st.cache_data
def load_and_train():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    df = df[['Survived', 'Pclass', 'Sex', 'Age']]
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    X = df[['Pclass', 'Sex', 'Age']]
    y = df['Survived']
    clf = RandomForestClassifier(random_state=42, max_depth=5)
    clf.fit(X, y)
    return clf

clf = load_and_train()

# 3. ユーザー入力（サイドバー）
st.sidebar.header('あなたの情報を入力')
pclass = st.sidebar.selectbox('客室クラス（1がVIP、3が一般）', [1, 2, 3])
sex = st.sidebar.radio('性別', ['男性', '女性'])
age = st.sidebar.slider('年齢', 0, 100, 30)

sex_num = 0 if sex == '男性' else 1
input_data = pd.DataFrame([[pclass, sex_num, age]], columns=['Pclass', 'Sex', 'Age'])

survival_prob = clf.predict_proba(input_data)[0][1] * 100

st.subheader('AIの予測結果')
st.write(f'あなたの生存確率は... **{survival_prob:.1f} %** です！')

if survival_prob > 50:
    st.success('生き残る可能性が高いです！優先的に救命ボートに乗れるかもしれません。')
else:
    st.error('危険な状態です...冷たい海に投げ出されるかもしれません。')