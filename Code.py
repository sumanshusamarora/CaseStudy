# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sqlite3 as sql
import pandas as pd
import numpy as np
#Initializing SQLITE connection
conn = sql.connect(r'C:\Users\suman\Downloads\test_data.db\test_data.db')
#Task Stage 1 - Answer 1
c1 = conn.cursor()
c1.execute("Select ROUND(SUM(revenue)) from customers where cc_payments = 1;")
c1.fetchall()
#Task Stage 1 - Answer 2
c2 = conn.cursor()
c2.execute("""SELECT Round(((1.0*count(distinct case when cc_payments = 1 then customer_id end))/
                      (1.0*COUNT(customer_id)))*100,2) as TOTAL_FEM_ITEMS from customers where female_items > 0;""")
c2.fetchall()
#Task Stage 1 - Answer 3
c3 = conn.cursor()
c3.execute("""SELECT ROUND(AVG(Revenue)) FROM customers where desktop_orders > 0 or android_orders > 0 or ios_orders > 0;""")
c3.fetchall()
#Task Stage 1 - Answer 4
#While there could be many checks that we could put to target customer for a luxury brand but below is most simplest targeting criteria with an assumption that customer has a spending limit more than 500 in last 360 days
c4 = conn.cursor()
c4.execute("""SELECT Customer_id from customers where orders > (cancels+returns) and male_items > 0 and revenue/orders > 500;""")
c4.fetchall()

#Task Stage 2
#Duplicates, check revenue vs orders vs cancels, first order vs last order, segment wise items vs total items
#Getting all data in dataframe to check variables and other stats
data = pd.read_json(r'C:\Users\suman\Downloads\test_data\data.json')
#Adding variables names
data["customer_id"].count()
# Step 1 - Removing duplicates
data.drop_duplicates(inplace=True)
data.reset_index(drop=True, inplace=True)
data["customer_id"].count()

data_desc = data.describe()
data.dtypes
#Streamlining column order
data = data[['customer_id', 'days_since_first_order', 'days_since_last_order', 'is_newsletter_subscriber', 'orders', 'items', 'cancels', 'returns', 'female_items', 'wapp_items', 'wftw_items', 'wacc_items',  'wspt_items', 'male_items', 'mapp_items', 'macc_items', 'mftw_items', 'mspt_items', 'unisex_items', 'curvy_items', 'sacc_items', 'vouchers', 'cc_payments', 'paypal_payments', 'afterpay_payments', 'apple_payments', 'different_addresses', 'shipping_addresses', 'devices', 'msite_orders', 'desktop_orders', 'android_orders', 'ios_orders', 'other_device_orders', 'work_orders', 'home_orders', 'parcelpoint_orders', 'other_collection_orders', 'redpen_discount_used', 'coupon_discount_applied', 'average_discount_onoffer', 'average_discount_used', 'revenue']]
#Updaing newsletter_subscriber clumn flag as 1 and 0
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["is_newsletter_subscriber"] = le.fit_transform(data["is_newsletter_subscriber"])

#It seems like days since last order is actaully hours since last order to dividing by 24 to bring it at day level
data["days_since_last_order"] = data["days_since_last_order"]/24


#Only coupon_discount_applied column has some msising values and since there are some outliers in this column so imputing them by median value
data=data.fillna(data.median())

#Finding onservations where total items not equal to female + male + unisex
issue = 0
for x in range(len(data)):
    if data["items"][x] != data["female_items"][x] + data["male_items"][x] + data["unisex_items"][x]:
        issue+=1
print(str(issue) + " number of rows have mismatch in item count")

#Finding onservations where total orders are not equal to sum by shipping types
issue = 0
for x in range(len(data)):
    if data["orders"][x] != data["work_orders"][x] + data["home_orders"][x] + data["parcelpoint_orders"][x] + data["other_collection_orders"][x]:
        issue+=1
print(str(issue) + " number of rows does not sum up w.r.t home + office + other shipping types")

#Finding onservations where total orders are not equal to total device orders
issue = 0
lst_customer = []
for x in range(len(data)):
    if data["orders"][x] != data["msite_orders"][x] + data["desktop_orders"][x] + data["android_orders"][x] + data["ios_orders"][x] + data["other_device_orders"][x]:
        issue+=1
        lst_customer.append(data["customer_id"][x])
print(str(issue) + " number of orders does not sum up w.r.t mobile site + web + mobile application")

#Going by definition, it looks like the returns and cancellation are full order returned and cancelled, not just items so they should not be more than number of orders placed
data[data["orders"]<(data["cancels"] + data["returns"])] #1908 rows
data[data["items"]<(data["cancels"] + data["returns"])] #3 rows


#Check if items less than orders
data[data["orders"] > data ["items"]]

#Check if female itms total matches with total female item types
data[data["female_items"] > data["wapp_items"] + data["wftw_items"] + data["wacc_items"] + data["wspt_items"]] #2859
data[data["male_items"] > data["mapp_items"] + data["macc_items"] + data["mftw_items"] + data["mspt_items"]] #5480

#Since we do not know if the order got returned or cancel so updating number of orders to be equal to cancel + returns


#Stage 3 - Developing the model
#Basis my understanding there are many variables which do not contribute to the Gender prediction so removing them from data
#data_new = data[['female_items',  'male_items',  'unisex_items', 'curvy_items', 'sacc_items', 'redpen_discount_used', 'coupon_discount_applied', 'average_discount_onoffer', 'average_discount_used', 'revenue']]
data_new = data.copy()

data_new["FemaleItemsRatio"] = 0.1
data_new["TotalMaleFemaleItems"] = data_new["female_items"] + data_new["male_items"]
#data_new["AvgRevenue"] =data_new["revenue"]/ (data_new["TotalMaleFemaleItems"] + data_new["unisex_items"] + data_new["curvy_items"] + data_new["sacc_items"])
for x in range(len(data_new)):
    if data_new['female_items'][x]+data_new['male_items'][x] != 0:
        data_new.at[x,"FemaleItemsRatio"] = data_new['female_items'][x]/(data_new['female_items'][x]+data_new['male_items'][x])        
data_new2=data_new.drop(columns=['female_items',  'male_items'])
data_new2.drop(columns=['different_addresses',  'shipping_addresses', 'customer_id'], inplace=True)

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")
# Compute the correlation matrix
corr = data_new2.corr()
corr = abs(corr)

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

corr_FIR = corr["FemaleItemsRatio"][:]
important_features = list(corr_FIR[corr_FIR > 0.2].index)

#Finding the most important features that impact female_male_item_Ratio
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
x=np.array(data_new2[important_features])
ss = StandardScaler()
x = ss.fit_transform(x)
wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(x)

y_pred_kmeans = kmeans.fit_predict(x)
# Evaluate the K-Means clustering results
data_new2["Is_Male"] = y_pred_kmeans

test = data_new2[important_features]
test["Is_Male"] = y_pred_kmeans


#Building ANN now
data_ANN = data.copy()
data_ANN["Is_Male"] = y_pred_kmeans
data_ANN.drop(columns="customer_id", inplace=True)
X=data_ANN.iloc[:,:42]
y=data_ANN.iloc[:,42]
#Step 1 - Feature selection
from feature_selector import FeatureSelector
fts=FeatureSelector(X,y)
fts.identify_missing(missing_threshold=0.9)

fts.identify_collinear(correlation_threshold=0.7)
fts.plot_collinear()
collinear_features = fts.ops['collinear']

fts.identify_zero_importance(task='classification', eval_metric='auc', n_iterations=30, early_stopping=True)
zero_importance_features = fts.ops['zero_importance']

fts.plot_feature_importances(threshold=0.99, plot_n=12)
Most_important_Features = list(fts.feature_importances["feature"].head(28))

Data_ANN_2 = data_ANN[Most_important_Features]
X=Data_ANN_2.iloc[:,:]
y=data_ANN.iloc[:,42]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu', input_dim = 28))
classifier.add(Dropout(0.2))
# Adding the second hidden layer
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 500, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score, f1_score
accuracy_score(y_test, y_pred)
f1_score(y_test, y_pred)
