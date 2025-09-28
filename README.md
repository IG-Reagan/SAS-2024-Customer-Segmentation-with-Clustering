# Customer Segmentation with Clustering

## Overview
This project applies unsupervised machine learning to segment customers of a global e-commerce company into distinct groups based on behavioural and value metrics. By identifying customer clusters, the business can optimise marketing strategies, improve retention, and allocate resources more effectively.

## Problem Statement
Retailers face the challenge of understanding diverse customer bases across multiple continents. Grouping customers into meaningful segments enables more effective resource allocation and tailored marketing strategies. The objective of this project was to explore, preprocess, and analyse a large transactional dataset to uncover meaningful customer clusters that could guide marketing and retention efforts.

## Data Source
The dataset was provided by **SAS (2024)** and represents an anonymised, real-world e-commerce organisation. It includes transactions from **five continents (Oceania, North America, Europe, Africa, Asia)** and **47 countries**, covering **951,668 rows of orders** made between **1 January 2012 and 30 December 2016**. After aggregation, the dataset contained approximately **68,300 unique customers**. 

Reference: SAS, 2024. CUSTOMERS_CLEAN [Data set]. SAS. Last revised on 15 December 2021. [Accessed 20 February 2024].

> Disclaimer: While the dataset originates from SAS, the business context for this project is illustrative.

## Variables & Explanations
The dataset contained 20 raw features. Key examples:  
- **Quantity** – Number of items ordered.  
- **City / Country / Continent** – Customer location.  
- **Order & Delivery Dates** – Used to derive recency and delivery metrics.  
- **Total Revenue** – Value of the order (USD).  
- **Unit Cost** – Cost per item (USD).  
- **Discount** – Discount percentage applied.  
- **Customer Group / Type** – Loyalty programme labels.  
- **Customer ID** – Unique customer identifier.  

**Engineered features for segmentation**:  
- **Frequency** – Number of purchases per customer.  
- **Recency** – Days since the last purchase.  
- **Customer Lifetime Value (CLV)** – Total revenue per customer.  
- **Average Unit Cost** – Mean price of purchased items.  
- **Customer Age** – Derived from date of birth.  

## Methods / Tools Used
- **Languages & Libraries**: Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, SciPy  
- **Preprocessing**: Handling missing values, duplicates, feature engineering, scaling, encoding  
- **Dimensionality Reduction**: PCA, t-SNE  
- **Clustering**:  
  - KMeans (final model, *k=4*)  
  - Hierarchical Agglomerative Clustering (trialled on samples)  
- **Evaluation**: Elbow Method, Silhouette Scores  

## Results / Outcomes
- Optimum segmentation identified with **4 clusters** using KMeans.  
- **Cluster profiles**:  
  - *Cluster 1*: Medium frequency, medium CLV, low-cost items, increasing loyalty  
  - *Cluster 2*: Low activity, low CLV, weak brand loyalty → require targeted promotions  
  - *Cluster 3*: High frequency, high CLV, loyal “Gold” customers (low-cost frequent purchases)  
  - *Cluster 4*: Low frequency, high-cost purchases → growth potential with delivery incentives  
- KMeans was computationally efficient and robust compared to hierarchical clustering on this dataset.  

## How to Run
1. Clone this repository.  
2. Install dependencies:  
   pip install -r requirements.txt

3. Open and run the notebook:
   customer_segmentation_with_clustering.ipynb


4. Note: Raw dataset is not included due to licensing. A synthetic sample may be provided for demonstration.

## Supplementary Materials

- Full project report with analysis and visualisations: [Available on request]

- Presentation slides: [Available on request]

---
