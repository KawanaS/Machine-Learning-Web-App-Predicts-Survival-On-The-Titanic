import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

labelencoder=LabelEncoder()

st.title('PREDICT YOUR SURVIVAL ON THE TITANIC')

df=pd.read_csv('titanic.csv')

def visualize_data():
    st.subheader('Data Visualization')
    df = pd.read_csv('titanic.csv')
    if df is not None:
        df = df.drop(columns=['Name'])
        st.dataframe(df.head())

    visuals=st.sidebar.selectbox('Select Visualization',('Shape','Describe','Survival Value Count','Visualize Survival Count',
                                                             'Survival Rate By Class,Sex & Age'))
    if visuals=='Shape':
        st.subheader('Shape Of The Data')
        st.write(df.shape)
    if visuals=='Describe':
        st.subheader('Description')
        st.write(df.describe())
    if visuals == 'Survival Value Count':
        st.subheader('Survival Value Count')
        st.write(df.iloc[:, 0].value_counts())
    if visuals == 'Visualize Survival Count':
        sns.countplot(df['Survived'])
        st.pyplot()
    if visuals == 'Survival Rate By Class,Sex & Age':
        st.subheader('Survival Rate By Class,Sex & Age')
        fig, axs = plt.subplots(1, 3)
        sns.barplot(x='Pclass', y='Survived', data=df, ax=axs[2])
        sns.barplot(x='Sex', y='Survived', data=df, ax=axs[1])
        age = pd.cut(df['Age'], [0, 18, 30, 45, 60, 80])
        b = sns.barplot(x=age, y='Survived', data=df, ax=axs[0])
        b.tick_params(labelsize=6.5)
        st.pyplot(fig)

visualize_data()

model=st.sidebar.selectbox('Select Model', ('SVC', 'KNN', 'RFC'))

def classification(model_name):
    if model=='SVC':
        C = st.sidebar.slider('C', 0.01, 10.0)
        gamma = st.sidebar.slider('Gamma', 0.01, 10.0)
        clf = SVC(C=C, gamma=gamma, kernel='rbf', random_state=0)
    if model=='KNN':
        K = st.sidebar.slider('K', 1, 15)
        clf = KNeighborsClassifier(n_neighbors=K, metric='minkowski', p=2)
    if model=='RFC':
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion='entropy',
                                     random_state=0)
    return clf

clf=classification(model)

df= df.drop(columns=['Name'])
#encode the sex column
df.iloc[:,2]=labelencoder.fit_transform(df.iloc[:,2].values)
st.dataframe(df.head())
#split the data into independent x and dependent y variables
x=df.iloc[:,1:7].values
y=df.iloc[:,0].values
#split the data into 80% training and 20% testing
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
st.write(f"Classifier={model}")
st.write(f"Accuracy={accuracy}")

def survival():
    Pclass = st.sidebar.slider('Pclass', 1, 3)
    sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
    age = st.sidebar.slider('Age', 0, 80)
    n_sibling_spouse = st.sidebar.slider('Siblings/Spouses', 0, 5)
    n_parents_children = st.sidebar.slider('Parents/Children', 0, 3)
    fair = st.sidebar.slider('Fair', 0, 200)
    if sex == 'Male':
        sex = 1
    else:
        sex = 0

    data=[[Pclass, sex, age, n_sibling_spouse, n_parents_children, fair]]
    return data

survive_or_not=survival()
prediction=clf.predict(survive_or_not)

if prediction==0:
    st.write('You did not survive!')
else:
    st.write('You survived!')
    