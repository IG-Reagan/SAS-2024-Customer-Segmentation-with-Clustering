# Customer Segmentation with Clustering

**This project applies unsupervised machine learning to segment e-commerce customers using clustering methods.  The dataset (SAS, 2024) covers ~68k customers across 47 countries, with transactions from 2012–2016. Key techniques include feature engineering (Frequency, Recency, CLV, Avg. Unit Cost, Age), dimensionality reduction (PCA, t-SNE), and clustering (KMeans, Hierarchical). The final KMeans model identified 4 customer segments, providing insights for targeted marketing and retention strategies.**

**1. Import the required libraries and data set with the provided URL.**

# Install gdown to download the files from google drive

!pip install gdown

!gdown 'https://drive.google.com/uc?export=download&id=1S5wniOV5_5htDfUFeZhlCLibvtihNLKK'

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import downloaded dataset
download = 'CUSTOMERS_CLEAN.csv'

# data = pd.read_csv(url, parse_dates=['Order_Date', 'Delivery_Date', 'Customer_BirthDate'])
data = pd.read_csv(download)

# View the dataset
data.head(3)

# Check the data shape
data.shape

# Display the dataset info to have a general idea of the data
data.info()

# Check how many records vs unique Customer ID
print('number of records', len(data['Customer ID']))
print('number of unique customers', data['Customer ID'].nunique())

**2.1. Identify missing values**

# Check the sum of missing values per feature
data.isna().sum()

# Find out the proportion missing values per column
data.isna().mean() * 100

# Drop rows with null values in City and Postal_Code columns.
data_2 = data.dropna(subset=['City', 'Postal_Code'])
print('New shape of the dataset is: ', data_2.shape)

# Check percentage of data records left
percentage_rows_left = data_2.shape[0] / data.shape[0]*100
print('Percentage of rows left:', round(percentage_rows_left,2), '%')

# Filter out countries with missing values in the 'State_Province' column
countries_with_missing_states = data_2[data_2['State_Province'].isna()]['CustomerCountryLabel'].unique()
print(len(countries_with_missing_states), 'countries with missing values in State_Province column')
print(countries_with_missing_states)

# View all 35 countries with missing values in the 'State_Province' column
def count_missing_vals(series):
    missing_count = series.isna().sum()
    return missing_count

missing_df = data_2.groupby(['CustomerCountryLabel'])['State_Province'].agg(count='count', nulls=count_missing_vals)
missing_df = missing_df.sort_values(by=['nulls'], ascending=False)
print("List showing the 35 countries with missing values in the 'State_Province' column")
print(missing_df.head(40)) # View the top 40 countries with the highest number of missing values

# Drop State_Province column and view dataset info again
data_3 = data_2.drop(columns = ['State_Province'], axis=1)

#  Check percentage of null values by row
data_3.info()

**2.2. Check for duplicates**

# Check for duplicated rows
duplicated_rows = data_3.duplicated().sum()
print('Number of duplicated records: ', duplicated_rows)

# Drop duplicated rows
data_3 = data_3.drop_duplicates()


# Check duplicated rows again
duplicated_rows = data_3.duplicated().sum()
print('Number of duplicated records: ', duplicated_rows)
print('Shape of dataset:', data_3.shape)


**2.3. Determine if there are any outliers**

# Import the re library to convert currency to floats
import re

# Function to convert currency strings to float
def currency_to_float(currency_str):
    # Check if the currency string is negative and remove parentheses if present
    is_negative = '(' in currency_str and ')' in currency_str
    clean_str = re.sub(r'[^\d.]+', '', currency_str)

    try:
        value = float(clean_str)
        if is_negative:
            value = -value
        return value
    except ValueError:
        return float('nan')

# Apply the custom function to the 'Total Revenue' column
data_3['Total Revenue'] = data_3['Total Revenue'].apply(currency_to_float)
data_3['Unit Cost'] = data_3['Unit Cost'].apply(currency_to_float)
data_3['Profit'] = data_3['Profit'].apply(currency_to_float)

data_3.head(3)

# Check descriptive statistics of the numeric variables
num_columns = ['Total Revenue', 'Unit Cost', 'Profit', 'Quantity']
data_3[num_columns].describe().T

# Use a boxplot to check the statistical distribution of the numberic columns; Total Revenue, Unit Cost and Profit
fig, ax = plt.subplots(1, 4, figsize=(12, 3))

# Plot boxplots for different columns
sns.boxplot(y='Total Revenue', data=data_3, ax=ax[0])
ax[0].set_title('Total Revenue')

sns.boxplot(y='Unit Cost', data=data_3, ax=ax[1])
ax[1].set_title('Unit Cost')

sns.boxplot(y='Profit', data=data_3, ax=ax[2])
ax[2].set_title('Profit')

sns.boxplot(y='Quantity', data=data_3, ax=ax[3])
ax[3].set_title('Quantity')

plt.tight_layout()
plt.show()

data_3['Unit Price'] = data_3['Total Revenue'] / data_3['Quantity']
data_3['Profit_check'] = data_3['Total Revenue'] - (data_3['Unit Cost'] * data_3['Quantity'])

fig, ax = plt.subplots(1,4, figsize=(20,3))

sns.scatterplot(x='Unit Cost', y='Unit Price', data=data_3, ax=ax[0])
ax[0].set_title('Plot1: Unit Cost vs. Unit Price')

sns.scatterplot(x='Unit Cost', y='Profit', data=data_3, ax=ax[1])
ax[1].set_title('Plot2: Unit Cost vs. Profit')

sns.scatterplot(x='Unit Price', y='Profit', data=data_3, ax=ax[2])
ax[2].set_title('Plot3: Unit Price vs. Profit')

sns.scatterplot(x='Profit', y='Profit_check', data=data_3, ax=ax[3])
ax[3].set_title('Plot4: Profit vs. Profit_check')

plt.show()

# View record with profit < -1000
data_3[data_3['Profit'] < -1000]

**3. Perform feature engineering**

**3.1. Create new features for frequency, recency, CLV, average unit cost, and customer age.**

# Create new feature for frequency
data_4 = data_3
data_4['Frequency'] = data_4.groupby('Customer ID')['Order ID'].transform('size')
data_4.head()

# Calculate recency as the difference in days between each row order date from the very first order in the dataset.
# A higher number of days will indicate a more recent order.

# Convert Order date to datetime format
data_4['Order_Date'] = pd.to_datetime(data_4['Order_Date'])

# Earliest order date
first_order_date = data_4['Order_Date'].min()

# Calculate the number of days from the first order date
data_4['Recency'] = (data_4['Order_Date'] - first_order_date).dt.days

# Display first order date and dataframe showing the last 3 orders
print('First order was placed on: ', first_order_date)
data_4.sort_values('Order_Date', ascending = False).head(3)


# Create new feature named 'Customer_Age'
data_4['Customer_Age'] = (pd.to_datetime('today') - pd.to_datetime(data_4['Customer_BirthDate'])).dt.days // 365
data_4.head()

# View unique values in the Discount column
print(data_4['Discount'].unique())

# Create new column using a for loop
for i, row in data_4.iterrows():
  if row['Discount'] == '   .':
    data_4.at[i, 'Discount_Applied'] = 0
  else:
    data_4.at[i, 'Discount_Applied'] = 1

print(data_4['Discount_Applied'].unique())

print('Unique OrderTypeLabels - types of order placement channels: ', '\n', data_4['OrderTypeLabel'].unique())
print('')
print('Unique Customer_Groups - customer loyalty status: ', '\n', data_4['Customer_Group'].unique())

# Select features into an aggregate dataset with each customer per row by grouping each feature by Customer_ID

# Sort data by Recency in order to priorities information from the more recent orders over older ones
data_4_sorted = data_4.sort_values('Recency', ascending = False)

# Create the aggregate dataset grouped by Customer_ID with all features to be selected
data_aggr = data_4_sorted.groupby('Customer ID').agg({'Continent': 'first',
                                                      'CustomerCountryLabel': 'first',
                                                      'Customer_Type': 'first',
                                                      'City': 'first', # Takes the customers most recent city in case they have recently relocated
                                                      'Customer_Group': 'first', # Selects the most recent Customer_Group label, taking into account if the customer has recently changed status
                                                      'OrderTypeLabel': 'first', # Takes the first OrderLabel in the series which should be the most recent as sorted by Recency
                                                      'Total Revenue': 'sum',
                                                      'Unit Cost': 'mean',
                                                      'Frequency': 'max',
                                                      'Recency': 'max',
                                                      'Customer_Age': 'max',
                                                      'Discount_Applied': 'sum'}).reset_index()

# Rename total Revenue to CLV and Unit Cost to Avg_Unit_Cost
data_aggr = data_aggr.rename(columns = {'Total Revenue': 'CLV', 'Unit Cost': 'Avg_Unit_Cost'})

# View the first few rows of the aggregated dataset
data_aggr.head()

data_aggr.shape

**3.4. Perform feature scaling and encoding if needed**

# View descriptive statistics of the data
round(data_aggr.describe(), 2).T

# Scale the data using the standard scaler

# Extract the selected features
X = data_aggr[['CLV', 'Avg_Unit_Cost', 'Frequency', 'Recency', 'Customer_Age', 'Discount_Applied']].values

scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

scaled_X

# a. Check distributions of CLV, Frequency and Age

# Boxplots of CLV, Frequency and Age
fig, ax = plt.subplots(2,3, figsize=(18, 8))

sns.boxplot(data=data_aggr, y='CLV', ax=ax[0,0])
ax[0,0].set_title('Customer Lifetime Value (CLV)')
ax[0,0].set_ylabel('CLV', fontsize=12)

sns.boxplot(data=data_aggr, y='Frequency', ax=ax[0,1], color='green')
ax[0,1].set_title('Customer Order Frequency')
ax[0,1].set_ylabel('Frequency', fontsize=12)

sns.boxplot(data=data_aggr, y='Customer_Age', ax=ax[0,2], color='mediumslateblue')
ax[0,2].set_title('Customer Age Distribution')
ax[0,2].set_ylabel('Customer Age', fontsize=12)

# Histograms of CLV
sns.histplot(data=data_aggr, x='CLV', ax=ax[1,0], bins=40)
# Calculate mean and median lines
mean_value, median_value = data_aggr['CLV'].mean(), data_aggr['CLV'].median()
# Calculate mean and median
ax[1,0].axvline(mean_value, color='y', linestyle='--', label=f'Mean: {mean_value:.2f}')
ax[1,0].axvline(median_value, color='y', linestyle='-', label=f'Median: {median_value:.2f}')
ax[1,0].set_xlabel('CLV', fontsize=12)
ax[1,0].legend()

# Histograms of Frequency
sns.histplot(data=data_aggr, x='Frequency', ax=ax[1,1], bins=40, color='green')
# Calculate mean and median
mean_value, median_value = data_aggr['Frequency'].mean(), data_aggr['Frequency'].median()
# Add mean and median lines
ax[1,1].axvline(mean_value, color='y', linestyle='--', label=f'Mean: {mean_value:.2f}')
ax[1,1].axvline(median_value, color='y', linestyle='-', label=f'Median: {median_value:.2f}')
ax[1,1].set_xlabel('Frequency', fontsize=12)
ax[1,1].legend()

# Histograms of Age
sns.histplot(data=data_aggr, x='Customer_Age', ax=ax[1,2], bins=14, color='mediumslateblue')
ax[1,2].set_xlabel('Customer Age', fontsize=12)

plt.show()

# b. Distribution of orders by channel; countplot of OrderTypeLabel
plt.figure(figsize=(8,5))
sns.countplot(data=data_aggr, x='OrderTypeLabel')
plt.title('Distribution of Customers by Order Channels')
plt.show()


# c. Orders distributed by continent and channels; Countplot and barplots in grid, of CLV by Continent and OrderTypeLabel
fig, ax = plt.subplots(1,3, figsize=(24, 5))
colour_palette = {'Catalog Sale': 'green', 'Internet Sale': 'orange', 'Retail Sale': 'mediumblue'}

sns.countplot(data=data_aggr, x='Continent', hue='OrderTypeLabel', palette = colour_palette, ax=ax[0])
ax[0].set_title('Count of orders by Continent and Order Channel')
ax[0].set_xlabel('Continent', fontsize=12)

sns.barplot(data=data_aggr, x='Continent', y='CLV', hue='OrderTypeLabel', palette = colour_palette, ax=ax[1])
ax[1].set_title('Average CLV by Continent and Order Channel')
ax[1].set_xlabel('Continent', fontsize=12)

# Aggregate data to sum CLV
data_aggr_sum = data_aggr.groupby(['Continent', 'OrderTypeLabel']).agg({'CLV': 'sum'}).reset_index().sort_values('CLV', ascending = False)

sns.barplot(data=data_aggr_sum, x='Continent', y='CLV', hue='OrderTypeLabel', palette = colour_palette, ax=ax[2])
ax[2].set_title('Sum of CLV by Continent and Order Channel')
ax[2].set_xlabel('Continent', fontsize=12)
plt.show()


# d. Customers distributed into Customer Groups; Countplot of customer group
plt.figure(figsize=(8,5))
sns.countplot(data=data_aggr, x='Customer_Group')
plt.title('Distribution of Customers by Customer Groups')
plt.xlabel('Customer Groups (Loyalty Program)', fontsize=12)
plt.show()


# e. Customer Groups generating the highest CLV; Barplot of Customer Group vs CLV
fig, ax=plt.subplots(1,2, figsize=(15,5))

# Aggregate data to sum CLV
data_aggr_sum = data_aggr.groupby('Customer_Group').agg({'CLV': 'sum'}).reset_index().sort_values('CLV', ascending = False)

sns.barplot(data=data_aggr_sum, x='Customer_Group', y='CLV', ax=ax[0])
ax[0].set_title('Sum of CLV by Customer Group')
ax[0].set_xlabel('Customer Group', fontsize=12)

sns.barplot(data=data_aggr, x='Customer_Group', y='CLV', ax=ax[1])
ax[1].set_title('Average CLV by Customer Group')
ax[1].set_xlabel('Continent', fontsize=12)
plt.show()

# f. Are the customers contributing highest CLV also buying the most expensive products?; Scatterplot of CLV vs Unit Cost
fig, ax = plt.subplots(1,2, figsize=(15, 5))

sns.scatterplot(data=data_aggr, x='CLV', y='Avg_Unit_Cost', hue='Customer_Group', ax=ax[0])
ax[0].set_title('Customer CLV vs Average Unit Cost of Items they Bought', fontsize=13)

# g. Scatterplot of frequency vs CLV
sns.scatterplot(data=data_aggr, x='CLV', y='Frequency', hue='Customer_Group', ax=ax[1])
ax[1].set_title('Customer CLV vs Frequency of Order', fontsize=13)
plt.show()


# g. Relationship between Frequency and Recency; Scatter plot of frequency vs recency
plt.figure(figsize=(10,7))
sns.scatterplot(data=data_aggr, x='Recency', y='Frequency', hue='Customer_Group')
plt.title('Recency vs Frequency - Who are the most frequent customers? Did they order recently?')
plt.xlabel('Recency', fontsize=13)
plt.ylabel('Frequency', fontsize=13)
plt.show()


# h. Customers accessing more discounts; Barplot of frequency vs discount_applied
plt.figure(figsize=(8,5))
sns.barplot(data=data_aggr, x='Discount_Applied', y='Frequency')
plt.xlabel('Number of times shopped with discount', fontsize=12)
plt.show()

# i. Total Revenue evolution for the different ordering channels; Line plot of order date vs Total Revenue, hue - OrderTypeLabel
plt.figure(figsize=(15,6))
data_aggr_TR = data_4.groupby(['Order_Date', 'OrderTypeLabel']).agg({'Total Revenue':'sum'}).reset_index()
sns.lineplot(data=data_aggr_TR, x='Order_Date', y='Total Revenue', hue='OrderTypeLabel')
plt.title('Revenue Trend over the Years', fontsize=15)
plt.xlabel('Order Date', fontsize=13)
plt.ylabel('Total Revenue', fontsize=13)
plt.show()

# j. Visualise correlations between the variables using visualisation

# Use a custom function for reusability
def feature_corr(df, title):
    df_corr = df.corr()
    fig, ax = plt.subplots(figsize=(12, 9))

    mask = np.triu(np.ones_like(df_corr, dtype=bool))
    mask = mask[1:, :-1]
    corr = df_corr.iloc[1:,:-1].copy()

    cmap = sns.diverging_palette(0, 230, 90, 50, as_cmap=True)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
               linewidths=5, cmap=cmap, vmin=-1, vmax=1,
               cbar_kws={"shrink": .8}, square=True)

    yticks = [i for i in corr.index]
    xticks = [i for i in corr.columns]

    plt.yticks(plt.yticks()[0], labels=yticks, rotation=0)
    plt.xticks(plt.xticks()[0], labels=xticks)

    title = title
    plt.title(title, loc='left', fontsize=18)
    plt.subplots_adjust(bottom=0.25, left=0.25)

    plt.show()

columns = ['CLV', 'Avg_Unit_Cost', 'Frequency', 'Recency', 'Customer_Age', 'Discount_Applied']
feature_corr(df = data_aggr[columns], title = 'Feature Correlation')

**5. Incorporate Column Transformer and Pipeline**

# Create a new dataset suitable for clustering; remove column identifier ('Customer ID')
data_for_clustering = data_aggr.drop(columns='Customer ID')
print('Number of unique cities: ', data_aggr['City'].nunique())

# Remove 'City': There are 10,471 cities where customers are located which is too many to provide useful differentiation between customers for clustering
# Drop the City column
data_for_clustering = data_for_clustering.drop(columns=['City'])

# View the first few rows
data_for_clustering.head()

# Import libraries
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Set seed for reproducibility
np.random.seed(20)

# Incorporate column transformer and create a pipeline
umerical_features = data_for_clustering.select_dtypes('number').columns.tolist()
numeric_transformer = Pipeline(
    steps=[('scaler', StandardScaler())]
)

categorical_features = data_for_clustering.select_dtypes('object').columns.tolist()
categorical_transformer = Pipeline(
    steps=[('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))]
)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
    ]
)

# Fit the pipeline on the data
preprocessor.fit(data_for_clustering)

# Transform the data to see it's new shape
transformed_data = preprocessor.transform(data_for_clustering)

transformed_data.shape

# Dimension Reduction pipeline with PCA
dimreduc_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("dim_reduction", PCA(n_components=2))  # Reduce to 2 dimensions
])

# Fit the pipeline on the data
dimreduc_pipeline.fit(data_for_clustering)

# Transform the data
twodim_data = dimreduc_pipeline.transform(data_for_clustering)

# Check shape of 2D data after PCA
twodim_data.shape

**6. Select the optimum value of clusters (𝑘) with the Elbow and Silhouette score methods.**

**a. Elbow Method**

# Elbow method: try different k.
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++',random_state=0, n_init = 10)
    kmeans.fit(twodim_data)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters', fontsize=12)
plt.ylabel('WCSS', fontsize=12)
plt.show()

**b. Silhouette Method**

# Import library.
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

range_n_clusters = [2, 3, 4, 5, 6]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns.
    fig, ax = plt.subplots(1, 2, figsize=(18, 7))

    # The 1st subplot is the silhouette plot.
    # The silhouette coefficient can range from -1, 1 however the lowest here is -0.2 so we use [-0.2, 1].
    ax[0].set_xlim([-0.2, 1])

    # The (n_clusters+1)*10 inserts a blank space between silhouette plots of individual clusters to demarcate them clearly.
    ax[0].set_ylim([0, len(twodim_data) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator seed of 20 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=20, n_init = 10)
    cluster_labels = clusterer.fit_predict(twodim_data)

    # The silhouette_score gives the average value for all the samples, giving a perspective into the density and separation of formed clusters
    silhouette_avg = silhouette_score(twodim_data, cluster_labels)
    print("For n_clusters =", n_clusters,"; "
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(twodim_data, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them.
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax[0].fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle.
        ax[0].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot.
        y_lower = y_upper + 10  # 10 for the 0 samples.

    ax[0].set_title("The silhouette plot for the various clusters.")
    ax[0].set_xlabel("The silhouette coefficient values")
    ax[0].set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values.
    ax[0].axvline(x=silhouette_avg, color="red", linestyle="--")

    ax[0].set_yticks([])  # Clear the yaxis labels / ticks.
    ax[0].set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed.
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax[1].scatter(twodim_data[:, 0], twodim_data[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters.
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers.
    ax[1].scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax[1].scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax[1].set_title("The visualization of the clustered data.")
    ax[1].set_xlabel("Feature space for the 1st feature")
    ax[1].set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()

**7. Hierarchical Clustering - Dendogram**

# Import libraries

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Perform Hierarchical Agglomerative Clustering
agglo_cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='average')
cluster_labels = agglo_cluster.fit_predict(twodim_data)

# Plotting the clusters
for cluster_idx in range(3):  # Adjust range according to the number of clusters
    cluster_data = twodim_data[cluster_labels == cluster_idx]
    sample_indices = np.random.choice(len(cluster_data), size=min(1000, len(cluster_data)), replace=False)
    plt.scatter(cluster_data[sample_indices, 0], cluster_data[sample_indices, 1], label=f'Cluster {cluster_idx + 1}')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Creating the linkage matrix and plotting the dendrogram
sample_indices = np.random.choice(len(twodim_data), size=min(1000, len(twodim_data)), replace=False)
Z = linkage(twodim_data[sample_indices], method='average')
plt.figure(figsize=(15, 10))
dendrogram(Z)
plt.title('Dendrogram for the Data')
plt.xlabel('Sample Index or (Cluster Size)')
plt.ylabel('Distance')
plt.show()

# Perform k-means clustering with 4 clusters using scikit-learn.
kmeans = KMeans(n_clusters=4, random_state=42,  n_init=10)
cluster_labels_KM = kmeans.fit_predict(twodim_data)

# Extracting centroids.
centroids = kmeans.cluster_centers_

# Print final centroids.
print("Final Centroids:", centroids)

# Add the cluster labels to the original dataset
data_aggr['Cluster'] = cluster_labels_KM + 1

# Verify the result
data_aggr.head()

# Plot boxplots of cluster assignments vs Frequency, Recency, CLV and Average Unit Cost

# Set the plot axes
fig, ax=plt.subplots(2,2, figsize=(13,10))

# Creat boxplots
sns.boxplot(data=data_aggr, x='Cluster', y='Frequency', ax=ax[0,0])
ax[0,0].set_title('Cluster Distribution by Order Frequency')

sns.boxplot(data=data_aggr, x='Cluster', y='Recency', ax=ax[0,1])
ax[0,1].set_title('Cluster Distribution by Order Recency')

sns.boxplot(data=data_aggr, x='Cluster', y='CLV', ax=ax[1,0])
ax[1,0].set_title('Cluster Distribution by Customer Lifetime Value (CLV)')

sns.boxplot(data=data_aggr, x='Cluster', y='Avg_Unit_Cost', ax=ax[1,1])
ax[1,1].set_title('Cluster Distribution by Avg Unit Cost of Orders Purchased')

# Set a global title of the plots
plt.suptitle(("Clusters Plotted against Frequency, Recency, CLV and Avg Unist Cost"), fontsize=14)
plt.subplots_adjust(top=0.9, bottom=0.2)

# Show plots
plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.tight_layout()
plt.show()

# Set plot size
plt.figure(figsize=(10,7))

# Plotting the clusters and centroids.
for cluster_idx in range(4):
    # cluster_data = data[data['cluster'] == cluster_idx]
    cluster_data = twodim_data[cluster_labels_KM == cluster_idx]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_idx + 1}')
    plt.scatter(*centroids[cluster_idx], s=200, marker='X', edgecolor='black')

plt.title('KMeans Clustering with 4 Clusters Using the 2D PCA data')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
# plt.xlim(145, 200)
# plt.ylim(40, 145)
plt.legend()
plt.show()

### Reference:
**SAS, 2024. CUSTOMERS_CLEAN [Data set]. SAS. Last revised on 15 December 2021. [Accessed 20 February 2024].**
