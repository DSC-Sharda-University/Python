import seaborn as sns
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import plotly.express as px


st.header("Machine learning web app to predict if a person will get daibetes\
 or not")
df = pd.read_csv("Admission_Predict.csv")
diabete = pd.read_csv("diabetes.csv")
if st.checkbox('Show dataframe'):
    st.write(df)
st.header("Is there Score usefulfor classifying the admission")
bar = px.scatter(df, x='GRE Score', y='TOEFL Score', color='Research',
                 marginal_x='box', marginal_y='violin')
bar

st.header("how many University rating do we have")
count_rate = px.bar(df, x='University Rating')
count_rate

st.header("How is the GRE distributed over Research")
bar = px.histogram(df, x='GRE Score', color='Research', marginal='box')
bar

st.header("Is there SOP for each CGPA of students")
box = px.box(df, x="SOP", y="CGPA")
box

st.header("")
box_ = px.scatter(df, x='CGPA', y='LOR ', marginal_x='violin',
                  marginal_y='box', color='Research')
box_

count_p = sns.countplot(df['University Rating'])

count_p

st.header("Statistical analysis of the dataset")
st.write(df.describe().T)

fig = px.density_heatmap(df, x="University Rating", y='SOP')
fig

fig = px.funnel(df, x='SOP', y='LOR ', color='Research')
fig

bar = px.histogram(diabete, x='Pregnancies', color='Outcome', marginal='box')
bar

scat = px.scatter(diabete, x='SkinThickness', y='Age', color='Outcome',
                  marginal_x='box', marginal_y='violin')
scat

if st.checkbox('Show columns'):
    st.write(diabete.columns.to_list())

if st.checkbox("Show number of classes to predict dataset"):
    st.write(diabete.Outcome.value_counts())

X = diabete.drop('Outcome', axis=1)
y = diabete["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.25,
                                                    random_state=1)
print(df.columns.to_list())
alg = ['Decision Tree', 'Support Vector Machine', "Random Forest",
       'Gaussian', 'Adaboost']
classifier = st.selectbox('Which algorithm?', alg)

if classifier == 'Decision Tree':
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    acc = dtc.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_dtc = dtc.predict(X_test)
    cm_dtc = confusion_matrix(y_test, pred_dtc)
    st.write('Confusion matrix: ', cm_dtc)


elif classifier == 'Support Vector Machine':
    svm = SVC()
    svm.fit(X_train, y_train)
    acc = svm.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_svm = svm.predict(X_test)
    cm = confusion_matrix(y_test, pred_svm)
    st.write('Confusion matrix: ', cm)


elif classifier == 'Random Forest':
    rfc = RandomForestClassifier(n_estimators=500)
    rfc.fit(X_train, y_train)
    acc = rfc.score(X_test, y_test)
    st.write("Accuracy: ", acc)
    y_pred = rfc.predict(X_test)
    cm_rfc = confusion_matrix(y_test, y_pred)
    st.write("Confusion matrix: ", cm_rfc)

elif classifier == 'Gaussian':
    bayes = GaussianNB()
    bayes.fit(X_train, y_train)
    acc_nb = bayes.score(X_test, y_test)
    st.write('Accuracy is:', acc_nb)
    y_pred = bayes.predict(X_test)
    cm_nb = confusion_matrix(y_test, y_pred)
    st.write("confusion matrix of Gaussian")

elif classifier == 'Adaboost':
    adb = AdaBoostClassifier(learning_rate=0.1, n_estimators=200,
                             random_state=0, algorithm='SAMME.R')
    adb.fit(X_train, y_train)
    acc_adb = adb.score(X_test, y_test)
    st.write("Accuracy of adaboost is: ", acc_adb)
    y_pred = adb.predict(X_test)
    cm_adb = confusion_matrix(y_test, y_pred)
    st.write("confusion matrix of Adaboost is: ", cm_adb)
