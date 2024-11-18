#!/usr/bin/env python
# coding: utf-8

# Importing all the required modules:
# -

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import missingno as msno
#from ydata_profiling import ProfileReport
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from pyclustering.cluster.clarans import clarans;
from pyclustering.utils import timedcall;
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, concatenate
import xgboost as xgb
from xgboost import XGBClassifier


# Loading the Dataset:
# -

# In[3]:


telecom = pd.read_csv('telecom_customer_churn.csv')
telecom.head(3)


# In[4]:


originalData = telecom
originalData.head()


# In[5]:


telecom.info()


# Exploratory Data Analysis (EDA):
# -

# In[6]:


telecom.hist(figsize=(15,15), xrot=0);


# In[7]:


telecom.describe()


# In[8]:


telecom.pivot_table(values='Monthly Charge', index='Contract', columns='Customer Status', aggfunc='mean')


# In[9]:


telecom.pivot_table(values='Monthly Charge', index=['Contract', 'Multiple Lines'], columns='Customer Status', aggfunc='mean')


# In[10]:


telecom.pivot_table(values='Monthly Charge', index=['Contract', 'Internet Service'], columns='Customer Status', aggfunc='mean')


# In[11]:


telecom.pivot_table(values='Tenure in Months', index='Contract', columns='Customer Status', aggfunc='mean')


# In[12]:


Customer_Stayed=telecom[telecom['Customer Status']=='Stayed'].Age
Customer_Churned=telecom[telecom['Customer Status']=='Churned'].Age
Customer_Joined=telecom[telecom['Customer Status']=='Joined'].Age

plt.xlabel('Age')
plt.ylabel('Customers Numbers')
plt.hist([Customer_Stayed,Customer_Churned,Customer_Joined],label=['Stayed','Churned','Joined'])

plt.title('Customers Behavior ',fontweight ="bold")
plt.legend();


# In[13]:


sns.histplot(telecom['Age']);


# In[14]:


telecom_corr = telecom.select_dtypes(include='number')
corr_data  = telecom_corr.corr()
plt.figure(figsize = (20,10))
sns.heatmap(corr_data, annot = True);


# In[15]:


fig = px.histogram(telecom, x="Customer Status", template ='xgridoff',barmode = "group", title = "<b>Customer Status Distribution<b>")
fig.update_layout(width=400, height=400, bargap=0.2)
fig.show()


# In[16]:


pd.crosstab(telecom['Customer Status'], telecom['Married']).plot(kind='bar');


# In[17]:


pd.crosstab(telecom['Customer Status'], telecom['Gender']).plot(kind='bar')


# In[18]:


sns.countplot(data=telecom, x='Customer Status', hue='Internet Type');


# In[19]:


fig,axes=plt.subplots(7,2,figsize=(10,20))
sns.countplot(x="Gender",hue='Customer Status',data=telecom,ax=axes[0,0])
sns.countplot(x="Married",hue='Customer Status',data=telecom,ax=axes[0,1])
sns.countplot(x="Phone Service",hue='Customer Status',data=telecom,ax=axes[1,0])
sns.countplot(x="Multiple Lines",hue='Customer Status',data=telecom,ax=axes[1,1])
sns.countplot(x="Internet Service",hue='Customer Status',data=telecom,ax=axes[2,0])
sns.countplot(x="Online Security",hue='Customer Status',data=telecom,ax=axes[2,1])
sns.countplot(x="Online Backup",hue='Customer Status',data=telecom,ax=axes[3,0])
sns.countplot(x="Device Protection Plan",hue='Customer Status',data=telecom,ax=axes[3,1])
sns.countplot(x="Premium Tech Support",hue='Customer Status',data=telecom,ax=axes[4,0])
sns.countplot(x="Streaming TV",hue='Customer Status',data=telecom,ax=axes[4,1])
sns.countplot(x="Streaming Movies",hue='Customer Status',data=telecom,ax=axes[5,0])
sns.countplot(x="Streaming Music",hue='Customer Status',data=telecom,ax=axes[5,1])
sns.countplot(x="Unlimited Data",hue='Customer Status',data=telecom,ax=axes[6,0])
sns.countplot(x="Paperless Billing",hue='Customer Status',data=telecom,ax=axes[6,1])
plt.tight_layout()
plt.show()


# In[20]:


fig, ax = plt.subplots(figsize=(12, 8))
order = telecom[telecom['Customer Status'] == 'Churned']['Churn Reason'].value_counts().index
sns.countplot(data=telecom[telecom['Customer Status'] == 'Churned'], y='Churn Reason', hue='Customer Status', ax=ax, order=order)

plt.xlabel('Count')
plt.ylabel('Churn Reason')
plt.title('Count of Customer by Churn Reasons')

plt.show()


# Data Cleaning:
# -

# In[21]:


msno.matrix(telecom)


# In[22]:


#Dropping the whole columns
telecom = telecom.drop('Churn Category',axis=1)
telecom = telecom.drop('Churn Reason',axis=1)
telecom = telecom.drop('Customer ID',axis=1)


# In[23]:


#dropping the data values of "Joined" in the customer status
telecom = telecom.loc[telecom['Customer Status'] != 'Joined']


# In[24]:


telecom.columns


# In[25]:


a=telecom['Internet Service']
a.value_counts()


# In[26]:


b=telecom['Internet Type']
b.value_counts()


# Handling Null and Missing Values:

# In[27]:


telecom['Internet Type'] = telecom['Internet Type'].fillna('No Internet')
telecom['Avg Monthly GB Download'] = telecom['Avg Monthly GB Download'].fillna(0)
telecom['Online Security'] = telecom['Online Security'].fillna('No Internet')
telecom['Online Backup'] = telecom['Online Backup'].fillna('No Internet')
telecom['Device Protection Plan'] = telecom['Device Protection Plan'].fillna('No Internet')
telecom['Premium Tech Support'] = telecom['Premium Tech Support'].fillna('No Internet')
telecom['Streaming TV'] = telecom['Streaming TV'].fillna('No Internet')
telecom['Streaming Movies'] = telecom['Streaming Movies'].fillna('No Internet')
telecom['Streaming Music'] = telecom['Streaming Music'].fillna('No Internet')
telecom['Unlimited Data'] = telecom['Unlimited Data'].fillna('No Internet')


# In[28]:


telecom.isna().sum()


# In[29]:


telecom['Phone Service'].value_counts()


# In[30]:


telecom['Multiple Lines'] = telecom['Multiple Lines'].fillna('No Phone Service')
telecom['Avg Monthly Long Distance Charges'] = telecom['Avg Monthly Long Distance Charges'].fillna(0)


# Dropping Duplicates:

# In[31]:


#Dropping the duplicate values (if any)
telecom = telecom.drop_duplicates()


# In[32]:


msno.matrix(telecom)


# In[33]:


telecom.info()


# In[ ]:


profile = ProfileReport(telecom)


# Outlier Analysis:

# In[34]:


colnames=['Age','Number of Dependents','Number of Referrals','Tenure in Months','Avg Monthly Long Distance Charges',
                'Avg Monthly GB Download','Monthly Charge','Total Charges','Total Extra Data Charges','Total Long Distance Charges','Total Revenue']

fig, ax = plt.subplots(4,3, figsize = (15,15))
for i, subplot in zip(colnames, ax.flatten()):
    sns.boxplot(x = 'Customer Status', y = i , data = telecom, ax = subplot)


# In[35]:


sns.histplot(telecom['Number of Dependents']);


# In[36]:


sns.histplot(telecom['Number of Referrals']);


# In[37]:


sns.histplot(telecom['Total Extra Data Charges']);


# In[38]:


S=telecom['Total Extra Data Charges'].value_counts()
S.head()


# Data Preprocessing and Transformation:

# In[39]:


categorical_columns = ['Gender', 'City', 'Zip Code', 'Married', 'Offer', 'Phone Service', 'Multiple Lines', 'Internet Service',
                       'Internet Type', 'Online Security', 'Online Backup', 'Device Protection Plan',
                       'Premium Tech Support', 'Streaming TV', 'Streaming Movies', 'Streaming Music',
                       'Unlimited Data', 'Contract', 'Paperless Billing', 'Payment Method', 'Customer Status']

for column in categorical_columns:
    telecom[column] = pd.Categorical(telecom[column])


# In[40]:


telecom.dtypes


# In[41]:


data_telecom = telecom.copy()


# In[42]:


df_X = telecom.drop('Customer Status', axis=1)
df_y = telecom['Customer Status']


# In[43]:


set_config(display='diagram') # shows the pipeline graphically when printed


# In[44]:


num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('cat_encoder', OneHotEncoder(sparse=False)) # returns a
        # regular matrix that can be combined easily with the data we get from
        # the numeric pipeline
    ])

SimpleImputer.get_feature_names_out = StandardScaler.get_feature_names_out  #

prep_pipeline = ColumnTransformer([
    ('num', num_pipeline, make_column_selector(dtype_include=np.number)),
    ('cat', cat_pipeline, make_column_selector(dtype_include='category'))
])

prep_pipeline


# In[45]:


processed_X = prep_pipeline.fit_transform(df_X, df_y)
df_processed_X = pd.DataFrame(processed_X,
                              columns = prep_pipeline.get_feature_names_out(),
                              index = df_X.index)
print(df_processed_X.shape)
df_processed_X.head()


# Feature Selection:
# -

# Univariate Feature Selection:

# In[46]:


selector = SelectKBest(f_classif, k=50)
selector.fit_transform(df_processed_X, df_y)

cols = selector.get_support(indices=True)
features_df_new = df_processed_X.iloc[:,cols]

features_df_new.head()


# Principal Component Analysis:

# In[47]:


pca = PCA(n_components=30)
pca_features = pca.fit_transform(features_df_new, df_y)
new_df = pca.transform(features_df_new)


# Recurrent Feature Elimination:

# In[48]:


random_forest = RandomForestClassifier()

rfe = RFE(random_forest, n_features_to_select=20)

rfe.fit(pca_features, df_y)

selected_features = rfe.support_

cols1 = rfe.get_support(indices=True)
df_3 = features_df_new.iloc[:,cols1]
df_3.columns


# In[49]:


print(df_3.head())


# In[50]:


data = df_3.values


# Clustering:
# -

# DBSCAN Clustering:

# In[51]:


# Perform DBSCAN clustering
dbscan = DBSCAN(eps=1, min_samples=3)
dbscan_labels = dbscan.fit_predict(data)

# Get the number of unique labels
num_unique_labels = len(np.unique(dbscan_labels))

if num_unique_labels > 1:
    # Calculate silhouette score only if there are multiple labels
    dbscan_score = silhouette_score(data, dbscan_labels)
    print("DBSCAN Silhouette Score:", dbscan_score)
else:
    print("DBSCAN: Only one label found. Unable to calculate silhouette score.")


# KMeans Clustering:

# In[52]:


# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=24, n_init=10)
kmeans_labels = kmeans.fit_predict(data)

# Calculate silhouette score for KMeans clustering
kmeans_score = silhouette_score(data, kmeans_labels)
print("KMeans Silhouette Score:", kmeans_score)


# CLARANS Clustering:

# In[53]:


num_clusters = 5
num_local = 3
max_neighborhoods = 2


# In[54]:


# Initializing CLARANS clustering object
clarans_instance = clarans(data.tolist(), num_clusters, num_local, max_neighborhoods)


# In[55]:


# Perform CLARANS clustering
(ticks, result) = timedcall(clarans_instance.process);
print("Execution time : ", ticks, "\n");


# In[56]:


#Get the cluster labels
clst = clarans_instance.get_clusters();


# In[57]:


#Get the best mediods of the clusters 
med = clarans_instance.get_medoids();


# In[58]:


print("Index of clusters' points :\n",clst)
print("\nLabel class of each point :\n ",df_y)
print("\nIndex of the best medoids : ",med)


# Clustering Visualizations:

# _DBSCAN Clustering:_

# In[59]:


# Applying PCA for dimensionality reduction
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

# Creating a scatter plot for DBSCAN clustering results
plt.figure(figsize=(8, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=dbscan_labels, cmap='rainbow', s=50)
plt.title("DBSCAN Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cluster Labels")
plt.show()


# _KMeans Clustering:_

# In[60]:


# Applying PCA for dimensionality reduction
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

# Creating a scatter plot for KMeans clustering results
plt.figure(figsize=(8, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=kmeans_labels, cmap='rainbow', s=50)
plt.title("KMeans Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cluster Labels")
plt.show()


# _CLARNS Clustering:_

# In[61]:


# Retrieving the CLARANS cluster assignments
clarans_clusters = clarans_instance.get_clusters()

# Creating a scatter plot for visualizing the clusters
plt.figure(figsize=(6, 4))

for cluster_idx, cluster in enumerate(clarans_clusters):
    cluster_points = data[cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_idx + 1}')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('CLARANS Clustering Results')
plt.legend()
plt.show()


# Neural Networks Modeling:
# -

# In[62]:


#Gathering CLARANS Clustering labels for modeling data
num_clusters = len(np.unique(np.concatenate(clarans_clusters)))
cluster_feature = np.zeros((len(data), num_clusters))


# In[63]:


for cluster_idx, cluster in enumerate(clarans_clusters):
    cluster_feature[cluster, cluster_idx] = 1


# In[64]:


data_with_cluster = np.hstack((data, cluster_feature))


# In[65]:


X = data_with_cluster
y = df_y


# In[73]:


# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[75]:


df_X_test = pd.DataFrame(X_test)
X_test_indices = df_X_test.index

customerID = originalData.loc[X_test_indices, 'Customer ID']


# In[76]:


# Encoding target labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
num_classes = len(label_encoder.classes_)


# In[77]:


#Reshaping the data to be compatible with neural network models
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# Defining Activation Functions:

# In[78]:


# SWISH activation function Definition
def swish(x):
    return x * tf.sigmoid(x)


# In[79]:


# MISH activation function Definition
def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


# In[80]:


# APTx activation function Definition
def aptx(x):
    return x * tf.math.tanh(tf.math.softplus(x) + 1)


# Feed Forward Neural Network Definition:

# In[81]:


def build_ffnn_model(activation_function):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation=activation_function, input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation=activation_function),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model


# Convolutional Neural Network Defintion:

# In[82]:


def build_cnn_model(activation_function):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(16, 3, activation=activation_function, padding='same', input_shape=(X_train.shape[1], 1)),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(32, 3, activation=activation_function, padding='same'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=activation_function),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    return model


# Function to Compile, Train and Evaluate Neural Network Models:

# In[83]:


def compileandtrainmodel(model_name, model_type, epochs, batch_size):
    print(f"Building Model: {model_name}")
    model_type.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_type.fit(X_train, y_train_encoded, validation_data=(X_test, y_test_encoded), epochs=epochs, batch_size=batch_size)
    loss, accuracy = model_type.evaluate(X_test, y_test_encoded)
    print(f"Model Name: {model_name} ==> Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")


# Function to calculate metrics like Confusion Matrix, Precision, Recall, F1 Score:

# In[84]:


def metricscalculation(modelname, model, X_val, y_val):
    # Predicted labels for the model
    model_predictions = model.predict(X_val)
    model_preditcion_classes = np.argmax(model_predictions, axis=1)

    # Confusion matrix for the model
    model_confusion_matrix = confusion_matrix(y_val, model_preditcion_classes)
    print(f"{modelname} Confusion Matrix:")
    print(model_confusion_matrix)

    # Extracting values from Confusion Matrix
    TP = model_confusion_matrix[1, 1]
    FP = model_confusion_matrix[0, 1]
    TN = model_confusion_matrix[0, 0]
    FN = model_confusion_matrix[1, 0]
    
    # Calculation various metrics for the model
    model_precision = precision_score(y_val, model_preditcion_classes)
    model_recall = recall_score(y_val, model_preditcion_classes)
    model_f1 = f1_score(y_val, model_preditcion_classes)
    
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    print("True Positives:", TP)
    print("False Positives:", FP)
    print("True Negatives:", TN)
    print("False Negatives:", FN)
    
    modelmetrics_df = pd.DataFrame({
    "Metric": ["Precision", "Recall", "F1 Score", "Sensitivity", "Specificity"],
    modelname: [model_precision, model_recall, model_f1, sensitivity, specificity]
    })

    # Print the metrics table
    print(modelmetrics_df)


# In[126]:


import plotly.graph_objects as go
import pandas as pd

def create_table_with_formatting(actual_values, predicted_values, customer_ids):
    # Create DataFrame
    data = {'Customer ID': customer_ids, 'True Values': actual_values, 'Predicted Values': predicted_values}
    df = pd.DataFrame(data)
    df.insert(0, 'S.No', range(1, 1 + len(df)))

    # Map 1 to 'Not Churn' and 0 to 'Churn' in 'True Values' and 'Predicted Values' columns
    df['True Values'] = df['True Values'].map({1: 'Not Churn', 0: 'Churn'})
    df['Predicted Values'] = df['Predicted Values'].map({1: 'Not Churn', 0: 'Churn'})

    # Add a correctness column
    df['Correct Prediction'] = df['True Values'] == df['Predicted Values']

    # Create a table with Plotly Graph Objects
    fig = go.Figure(data=[go.Table(
        header=dict(values=df.columns),
        cells=dict(values=df.transpose().values),
    )])

    # Conditional formatting for false predictions
    color_map = {True: 'palegreen', False: 'lightsalmon'}
    cell_colors = [[color_map[correct] for correct in df['Correct Prediction']]]

    # Update table layout with conditional formatting
    fig.update_traces(cells=dict(fill=dict(color=cell_colors)))

    # Customize layout
    fig.update_layout(
        title_text='Table: Predictions vs True Values',
        autosize=True,
        width=700,  # Total width of the table
        margin=dict(l=0, r=0, b=0, t=40),  # Adjust margins as needed
    )

    return fig


# Evaluation of Feed Forward Neural Networks with Various Activation Functions:

# _Feed Forward Neural Network with ReLU Activation Function_

# In[86]:


model_name = "Feed Forward Neural Network with ReLU Activation Function"
ffnn_model_relu = build_ffnn_model('relu')

compileandtrainmodel(model_name, ffnn_model_relu, 10, 40)


# In[87]:


metricscalculation(model_name, ffnn_model_relu, X_test, y_test_encoded)


# In[127]:


## FFNN ReLU
fig = create_table_with_formatting(y_test_encoded, np.argmax(ffnn_model_relu.predict(X_test), axis=1), customerID)
fig.show()


# _Feed Forward Neural Network with SWISH Activation Function_

# In[89]:


model_name = "Feed Forward Neural Network with SWISH Activation Function"
ffnn_model_swish = build_ffnn_model(swish)

compileandtrainmodel(model_name, ffnn_model_swish, 10, 40)


# In[90]:


metricscalculation(model_name, ffnn_model_swish, X_test, y_test_encoded)


# In[128]:


## FFNN SWISH
fig = create_table_with_formatting(y_test_encoded, np.argmax(ffnn_model_swish.predict(X_test), axis=1), customerID)
fig.show()


# _Feed Forward Neural Network with MISH Activation Function_

# In[92]:


model_name = "Feed Forward Neural Network with MISH Activation Function"
ffnn_model_mish = build_ffnn_model(mish)

compileandtrainmodel(model_name, ffnn_model_mish, 10, 40)


# In[93]:


metricscalculation(model_name, ffnn_model_mish, X_test, y_test_encoded)


# In[129]:


## FFNN MISH
fig = create_table_with_formatting(y_test_encoded, np.argmax(ffnn_model_mish.predict(X_test), axis=1), customerID)
fig.show()


# _Feed Forward Neural Network with APTx Activation Function_

# In[95]:


model_name = "Feed Forward Neural Network with APTx Activation Function"
ffnn_model_aptx = build_ffnn_model(aptx)

compileandtrainmodel(model_name, ffnn_model_aptx, 10, 40)


# In[96]:


metricscalculation(model_name, ffnn_model_aptx, X_test, y_test_encoded)


# In[130]:


## FFNN APTx
fig = create_table_with_formatting(y_test_encoded, np.argmax(ffnn_model_aptx.predict(X_test), axis=1), customerID)
fig.show()


# Evaluation of Convolutional Neural Networks with Various Activation Functions:

# _Convolutional Neural Network with ReLU Activation Function:_

# In[98]:


model_name = "Convolutional Neural Network with ReLU Activation Function"
cnn_model_relu = build_ffnn_model('relu')

compileandtrainmodel(model_name, cnn_model_relu, 10, 32)


# In[99]:


metricscalculation(model_name, cnn_model_relu, X_test, y_test_encoded)


# In[131]:


## CNN ReLU
fig = create_table_with_formatting(y_test_encoded, np.argmax(cnn_model_relu.predict(X_test), axis=1), customerID)
fig.show()


# _Convolutional Neural Network with SWISH Activation Function:_

# In[101]:


model_name = "Convolutional Neural Network with SWISH Activation Function"
cnn_model_swish = build_ffnn_model(swish)

compileandtrainmodel(model_name, cnn_model_swish, 10, 32)


# In[102]:


metricscalculation(model_name, cnn_model_swish, X_test, y_test_encoded)


# In[132]:


## CNN SWISH
fig = create_table_with_formatting(y_test_encoded, np.argmax(cnn_model_swish.predict(X_test), axis=1), customerID)
fig.show()


# _Convolutional Neural Network with MISH Activation Function:_

# In[104]:


model_name = "Convolutional Neural Network with MISH Activation Function"
cnn_model_mish = build_ffnn_model(mish)

compileandtrainmodel(model_name, cnn_model_mish, 10, 32)


# In[105]:


metricscalculation(model_name, cnn_model_mish, X_test, y_test_encoded)


# In[133]:


## CNN MISH
fig = create_table_with_formatting(y_test_encoded, np.argmax(cnn_model_mish.predict(X_test), axis=1), customerID)
fig.show()


# _Convolutional Neural Network with APTx Activation Function:_

# In[107]:


model_name = "Convolutional Neural Network with APTx Activation Function"
cnn_model_aptx = build_ffnn_model(aptx)

compileandtrainmodel(model_name, cnn_model_aptx, 10, 32)


# In[108]:


metricscalculation(model_name, cnn_model_aptx, X_test, y_test_encoded)


# In[134]:


## CNN APTx
fig = create_table_with_formatting(y_test_encoded, np.argmax(cnn_model_aptx.predict(X_test), axis=1), customerID)
fig.show()


# Ensemble Model of CNN and FFNN:

# In[110]:


# Combining the outputs of CNN and FFNN Models
combinedInput = concatenate([cnn_model_swish.output, ffnn_model_swish.output])

# Add Fully Connected layers for Ensemble Model
x = Dense(4, activation=mish)(combinedInput)
x = Dense(1, activation="sigmoid")(x)

# Creating the Ensemble Model
ensemble_model = Model(inputs=[cnn_model_swish.input, ffnn_model_swish.input], outputs=x)

# Compiling the Ensemble Model
ensemble_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the Ensemble Model
ensemble_model.fit([X_train, X_train], y_train_encoded, validation_data = ([X_test, X_test], y_test_encoded), epochs=10, batch_size=32, verbose=1)

# Making predictions with the Ensemble Model
ensemble_predictions = ensemble_model.predict([X_test, X_test])
ensemble_classes = np.round(ensemble_predictions)

# Calculating the accuracy of Ensemble Model
accuracy = accuracy_score(y_test_encoded, ensemble_classes)
print(f"Ensemble Accuracy: {accuracy:.4f}")


# In[135]:


## Ensemble Predictions
fig = create_table_with_formatting(y_test_encoded, ensemble_classes[0][0], customerID)
fig.show()


# Hyper Parameter Tuning and Cross Validation to Boost the Ensemble Model with XGBoost:

# In[118]:


# Reshaping ensemble_predictions to be a 2-dimensional matrix
ensemble_predictions_reshaped = np.reshape(ensemble_predictions, (len(ensemble_predictions), -1))

# Defining the hyperparameter grid
param_grid = {
    'n_estimators': [450, 460, 470, 480],
    'max_depth': [10, 11, 12],
    'learning_rate': [0.3, 0.4, 0.5]
}

# Initializing the XGBoost classifier
xgb_classifier = XGBClassifier()

# Initializing GridSearchCV with cross-validation
grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, scoring='accuracy', cv=3)

# Fitting the grid search to the data
grid_search.fit(ensemble_predictions_reshaped, y_test_encoded)

# Obtaining the best hyperparameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Printing the best parameters
print("Best Hyperparameters:", best_params)

# Storing the best parameters
xgb_n_estimators = best_params['n_estimators']
xgb_max_depth = best_params['max_depth']
xgb_learning_rate = best_params['learning_rate']

# Making predictions using the best model
best_predictions = best_model.predict(ensemble_predictions_reshaped)

# Calculating accuracy using the true labels
accuracy = accuracy_score(y_test_encoded, best_predictions)
print(f"Accuracy Score on Test Set: {accuracy:.4f}")


# Performing XGBoost on Ensemble Model:

# In[119]:


# Converting ensemble predictions into binary classes
ensemble_classes = np.round(ensemble_predictions)

# Calculating ensemble accuracy
ensemble_accuracy = accuracy_score(y_test_encoded, ensemble_classes)
print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")

# Training XGBoost Model on the ensemble predictions
xgb_model = xgb.XGBClassifier(n_estimators= xgb_n_estimators, max_depth = xgb_max_depth, learning_rate = xgb_learning_rate)
xgb_model.fit(ensemble_predictions, y_test_encoded)

# Predicting using the XGBoost Model
xgb_predictions = xgb_model.predict(ensemble_predictions)

# Calculating the XGBoost Model accuracy
xgb_accuracy = accuracy_score(y_test_encoded, xgb_predictions)
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")


# In[136]:


## XGBoost Predictions
fig = create_table_with_formatting(y_test_encoded, xgb_predictions, customerID)
fig.show()


# Generating Confusion Matrix and Calculating Metrics for Ensemble Model and XGBoost Model:

# In[120]:


# Calculating confusion matrix for ensemble predictions
ensemble_confusion_matrix = confusion_matrix(y_test_encoded, ensemble_classes)
print("Ensemble Confusion Matrix:")
print(ensemble_confusion_matrix)

# Extracting values from Ensemble Confusion Matrix
ensemble_TP = ensemble_confusion_matrix[1, 1]
ensemble_FP = ensemble_confusion_matrix[0, 1]
ensemble_TN = ensemble_confusion_matrix[0, 0]
ensemble_FN = ensemble_confusion_matrix[1, 0]

# Calculating precision, recall, F1 score, Sensitivity and Specificity for Ensemble model
ensemble_precision = precision_score(y_test_encoded, ensemble_classes)
ensemble_recall = recall_score(y_test_encoded, ensemble_classes)
ensemble_f1 = f1_score(y_test_encoded, ensemble_classes)

ensemble_sensitivity = ensemble_TP / (ensemble_TP + ensemble_FN)
ensemble_specificity = ensemble_TN / (ensemble_TN + ensemble_FP)


# In[121]:


# Calculating confusion matrix for XGBoost predictions
xgb_confusion_matrix = confusion_matrix(y_test_encoded, xgb_predictions)
print("XGBoost Confusion Matrix:")
print(xgb_confusion_matrix)

# Extracting values from Ensemble Confusion Matrix
xgb_TP = xgb_confusion_matrix[1, 1]
xgb_FP = xgb_confusion_matrix[0, 1]
xgb_TN = xgb_confusion_matrix[0, 0]
xgb_FN = xgb_confusion_matrix[1, 0]

# Calculating precision, recall, F1 score, Sensitivity and Specificity for XGBoost model
xgb_precision = precision_score(y_test_encoded, xgb_predictions)
xgb_recall = recall_score(y_test_encoded, xgb_predictions)
xgb_f1 = f1_score(y_test_encoded, xgb_predictions)

xgb_sensitivity = xgb_TP / (xgb_TP + xgb_FN)
xgb_specificity = xgb_TN / (xgb_TN + xgb_FP)


# In[122]:


# Creating a DataFrame to store the metrics
metrics_df = pd.DataFrame({
    "Metric": ["Precision", "Recall", "F1 Score", "Sensitivity", "Specificity"],
    "Ensemble": [ensemble_precision, ensemble_recall, ensemble_f1, ensemble_sensitivity, ensemble_specificity],
    "XGBoost": [xgb_precision, xgb_recall, xgb_f1, xgb_sensitivity, xgb_specificity]
})

# Print the metrics table
print(metrics_df)


# Visualizing the Confusion Matrices of Ensemble Model and XGBoost Model:

# In[123]:


# Defining class labels
class_labels = ["Negative", "Positive"]

# Setting the figure size
plt.figure(figsize=(10, 5))

# Plotting Ensemble Model Confusion Matrix 
plt.subplot(1, 2, 1)
sns.heatmap(ensemble_confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix - Ensemble Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Plotting XGBoost Model Confusion Matrix
plt.subplot(1, 2, 2)
sns.heatmap(xgb_confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix - XGBoost Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()


# Calculating ROC Curve and AUC for XGBoost Model:

# In[124]:


# Obtaining the predicted probabilities from the XGBoost model
xgb_probabilities = xgb_model.predict_proba(ensemble_predictions)[:, 1]

# Computing ROC curve and AUC for the XGBoost model
fpr_xgb, tpr_xgb, _ = roc_curve(y_test_encoded, xgb_probabilities)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

# Plotting the ROC curve for the XGBoost model
plt.figure()
plt.plot(fpr_xgb, tpr_xgb, color='darkorange', lw=2, label='XGBoost (AUC = %0.2f)' % roc_auc_xgb)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




