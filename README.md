# Customer Segmentation with Clustering

## Overview
This project applied unsupervised learning techniques to segment customers of a multinational e-commerce business. The objective was to identify distinct groups of customers based on purchasing behaviour and demographics, enabling more targeted marketing strategies and improved customer retention.

The analysis used clustering algorithms (KMeans and Hierarchical) alongside dimensionality reduction (PCA, t-SNE) to explore customer groups and generate actionable insights.

## Problem Statement
E-commerce companies need to understand their diverse customer base to improve marketing efficiency, customer satisfaction, and retention. This project addressed the challenge by using clustering methods to group customers based on frequency, recency, customer lifetime value (CLV), and average unit cost. The resulting segments provide a basis for targeted promotions, loyalty programmes, and product development.

## Variables & Explanations
**Original dataset variables (sample):**
- `Quantity`: Number of units ordered.  
- `City`, `State Province`, `Country`, `Continent`: Customer location data.  
- `Order Date`, `Delivery Date`: Dates of transaction and fulfilment.  
- `Total Revenue`: Total revenue of order.  
- `Unit Cost`: Cost per unit ordered.  
- `Discount`: Discount percentage applied.  
- `Order Type Label`: Sales channel (internet, retail, catalogue).  
- `Customer Group`: Loyalty group membership.  
- `Customer Type`: Loyalty membership tier.  
- `Profit`: Calculated as `(Total Revenue – Unit Cost) × Quantity`.  
- `Days to Delivery`: Days between order and delivery.  
- `Customer ID`: Unique identifier for each customer.  

**Engineered features:**
- **Frequency**: Number of purchases by customer over study period.  
- **Recency**: Days since last order.  
- **Customer Lifetime Value (CLV)**: Total income generated per customer.  
- **Average Unit Cost**: Average cost of items purchased.  
- **Customer Age**: Derived from birth date.  

## Methods / Tools Used
- **Python Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn, SciPy  
- **Feature Engineering**: Derived Frequency, Recency, CLV, Avg. Unit Cost, Age  
- **Pre-processing**: Handling missing values, removing duplicates, scaling, one-hot encoding  
- **Dimensionality Reduction**: PCA (Principal Component Analysis), t-SNE  
- **Clustering Techniques**:  
  - KMeans (Elbow and Silhouette methods for k selection)  
  - Hierarchical clustering (Agglomerative)  

## Results / Outcomes
- **Optimal number of clusters**: 4 (determined via Elbow and Silhouette methods).  
- **Cluster profiles**:  
  - **Cluster 1**: Medium frequency, medium CLV, lower unit cost items.  
  - **Cluster 2**: Low frequency, low CLV, low loyalty — candidates for targeted promotions.  
  - **Cluster 3**: High frequency, high CLV, mostly low-cost repeat purchases (loyal customers).  
  - **Cluster 4**: Low frequency, high unit cost — occasional big-ticket buyers.  

- KMeans provided the most stable segmentation compared with hierarchical clustering, which was resource-intensive and sensitive to outliers.  

## Source of Data
The dataset originates from an anonymised e-commerce database made available by **SAS (2024)** for educational and research purposes. It covers transactions from 2012 to 2016 across 47 countries and five continents, comprising 951,668 records.  

## How to Run
1. Clone this repository.  
2. Install dependencies listed in `requirements.txt`.  
3. Open the notebook `Customer_Segmentation.ipynb`.  
4. Run all cells to reproduce the feature engineering, dimensionality reduction, and clustering analysis.  
   - **Note**: Original dataset not included for confidentiality. A synthetic subset can be used for demonstration.  

## Links to Supplementary Material
- [Project Report (PDF)](link-to-your-drive-folder)  
- [Presentation / Visuals](link-to-your-drive-folder)  
