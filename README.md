# feature_selection
<img src = "https://designerapp.officeapps.live.com/designerapp/Media.ashx/?id=7be0e4ac-eabd-4d5f-8345-36f3016b2e1d.gif&fileToken=4b5a2bc0-68a9-43f7-a1c5-954eb13ac53a&dcHint=KoreaCentral"/>

<img src="https://github.com/vasanthgx/feature_selection/blob/main/images/Picture6.gif" width="300" align='center'>

<img src="https://github.com/vasanthgx/feature_selection/blob/main/images/Design.gif" width="300" align='center'>
 Trying out the different feature selection methods on the California housing dataset
 
 ![logo](https://github.com/vasanthgx/feature_selection/blob/main/images/resizedlogo1.png)
 <img src="https://github.com/vasanthgx/feature_selection/blob/main/images/resizedlogo1.png" width="300" align='center'>
 <img src="https://github.com/vasanthgx/feature_selection/blob/main/images/resizedlogo2.png" width="300" align='center'>
 
<img src="https://github.com/Anmol-Baranwal/Cool-GIFs-For-GitHub/assets/74038190/b3fef2db-e671-4610-bb84-1d65533dc5fb" width="300" align='center'>
<img src = "https://designerapp.officeapps.live.com/designerapp/Media.ashx/?id=7be0e4ac-eabd-4d5f-8345-36f3016b2e1d.gif&fileToken=4b5a2bc0-68a9-43f7-a1c5-954eb13ac53a&dcHint=KoreaCentral"/>
<br><br>

# Project Title

Exploring the different Feature Selection Methods on the California Housing Dataset.


## Implementation Details

- Dataset: California Housing Dataset (view below for more details)
- Model: [Linear Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- Input: 8 features - Median Houshold income, House Area, ...
- Output: House Price

## Dataset Details

This dataset was obtained from the StatLib repository ([Link](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html))

This dataset was derived from the 1990 U.S. census, using one row per census block group. A block group is the smallest geographical unit for which the U.S. Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people).

A household is a group of people residing within a home. Since the average number of rooms and bedrooms in this dataset are provided per household, these columns may take surprisingly large values for block groups with few households and many empty houses, such as vacation resorts.

It can be downloaded/loaded using the sklearn.datasets.fetch_california_housing function.

- [California Housing Dataset in Sklearn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- 20640 samples
- 8 Input Features: 
    - MedInc median income in block group
    - HouseAge median house age in block group
    - AveRooms average number of rooms per household
    - AveBedrms average number of bedrooms per household
    - Population block group population
    - AveOccup average number of household members
    - Latitude block group latitude
    - Longitude block group longitude
- Target: Median house value for California districts, expressed in hundreds of thousands of dollars ($100,000)

### What is univariate feature selection ?

Univariate feature selection is a type of feature selection method that evaluates each feature individually based on univariate statistical tests. In this approach, features are considered independently of each other, and their relevance to the target variable is assessed one at a time.

The process typically involves calculating a statistical metric (such as chi-square, F-test, or mutual information) to quantify the relationship between each feature and the target variable. Features are then ranked or scored based on their individual importance, and a predetermined number of top-ranked features are selected for inclusion in the final feature subset.

Univariate feature selection is relatively simple and computationally efficient, making it suitable for high-dimensional datasets. However, it may overlook interactions between features, which could affect model performance. It's often used as a preprocessing step before applying more sophisticated feature selection techniques or building machine learning models.

## Evaluation and Results

**Supervised**

### Filter Based Methods
**Mutual Information**
![alt text](https://github.com/vasanthgx/feature_selection/blob/main/images/mi.png)
Mutual information is a measure of the amount of information obtained about one random variable through the observation of another random variable. In the context of feature selection, mutual information is used to quantify the relationship between an input feature and the target variable.

In feature selection, mutual information is utilized to assess the relevance of each feature to the target variable. Higher mutual information between a feature and the target implies that the feature provides more information about the target variable and thus is more likely to be useful for prediction or classification tasks.

The mutual information between two variables, X and Y, can be calculated using various methods, such as entropy-based techniques. Essentially, it measures how much knowing the value of one variable reduces the uncertainty about the other variable.

In the context of feature selection, features with high mutual information with the target variable are usually selected, as they are expected to contribute more to the predictive power of the model. Conversely, features with low mutual information may be discarded as they are less informative for predicting the target variable.

![alt text](https://github.com/vasanthgx/feature_selection/blob/main/images/mi-bar-chart.png)

Overall, mutual information serves as a useful tool in feature selection by quantifying the relevance of features to the target variable, thereby aiding in the construction of more effective and efficient predictive models.

**Chi Squared**

The chi-squared (χ²) method is a statistical test used for feature selection in machine learning and statistics. It measures the dependence between two variables, typically the input features and the target variable in a classification task.

In feature selection, the chi-squared method assesses whether there is a significant association between each feature and the target variable. It is particularly useful when dealing with categorical variables.

**Pearson Correlation**

![alt text](https://github.com/vasanthgx/feature_selection/blob/main/images/pearson.png)

Pearson correlation is a measure of the linear correlation between two continuous variables. In feature selection for machine learning, Pearson correlation can be used to assess the relationship between each feature and the target variable.By examining the Pearson correlation coefficients, you can identify features that are strongly correlated (either positively or negatively) with the target variable, aiding in feature selection for machine learning models.
![alt text](https://github.com/vasanthgx/feature_selection/blob/main/images/heatmap.png)

Here we can clearly see the positive correlation between AveBedrms and AveRooms and negative correlation between Latitude and Longitude features.
### Wrapper Based Methods

**Recursive Feature Elimination ( R F E )**

Recursive Feature Elimination (RFE) is a feature selection technique commonly used in machine learning to select the most relevant features for a predictive model. It works by recursively removing the least important features until the specified number of features is reached or until a stopping criterion is met.
Recursive Feature Elimination helps in feature selection by iteratively removing the least important features, allowing the model to focus on the most relevant features. It can be particularly useful in scenarios where the number of features is large relative to the number of samples, as it helps in reducing overfitting and improving model interpretability.

**Select From Model**

SelectFrom Model is a feature selection technique in machine learning provided by scikit-learn. It allows you to select the most important features based on the importance weights provided by a base estimator (model). This technique is particularly useful when the base estimator has an attribute that ranks the importance of features, such as decision trees or linear models.
SelectFrom Model is a flexible and powerful feature selection technique that leverages the importance scores provided by a base estimator to select relevant features for your machine learning models.

**Sequential Feature Selection**

Sequential Feature Selection (SFS) is a type of feature selection method in machine learning where subsets of features are iteratively evaluated to find the best subset that maximizes the performance of the model. It sequentially adds or removes features from the feature set until a certain criterion is met.

There are two main types of Sequential Feature Selection:

**Forward Feature Selection:** In forward feature selection, the algorithm starts with an empty set of features and iteratively adds one feature at a time, selecting the one that improves the model's performance the most until a predefined criterion is met. This process continues until the desired number of features is reached or until further addition of features does not significantly improve the model's performance.

**Backward Feature Selection:** In backward feature selection, the algorithm starts with the entire set of features and iteratively removes one feature at a time, selecting the one whose removal improves the model's performance the most until a predefined criterion is met. This process continues until the desired number of features is reached or until further removal of features does not significantly improve the model's performance.

Sequential Feature Selection is typically performed using a performance metric such as accuracy, AUC, F1-score, or any other relevant metric for the specific task. The performance of the model is evaluated using cross-validation or another appropriate validation technique at each step of the feature selection process.

One of the advantages of Sequential Feature Selection is that it can reduce the dimensionality of the feature space while maintaining or even improving the performance of the model. It can also help in improving the interpretability of the model by selecting the most relevant features.

Scikit-learn provides an implementation of Sequential Feature Selection through the SequentialFeatureSelector class in the feature_selection module. This class supports both forward and backward feature selection strategies and can be used with any estimator that supports the fit and predict methods.

### Embedded Methods
Embedded methods of feature selection in machine learning are techniques where feature selection is integrated into the process of training the model itself. In other words, feature selection is performed "on the fly" during the model training process. These methods automatically select the most relevant features as part of the model training process, rather than as a separate preprocessing step.

Embedded feature selection methods are particularly common in algorithms where feature selection naturally occurs during the model training process. Some common examples of algorithms that incorporate embedded feature selection include:

**Lasso (L1 regularization):** Lasso regression adds a penalty term to the linear regression objective function, which forces some coefficients (and thus corresponding features) to be exactly zero. As a result, Lasso effectively performs feature selection by automatically setting less important features' coefficients to zero during training.

**Elastic Net:** Elastic Net is a linear regression model that combines L1 (Lasso) and L2 (Ridge) regularization penalties. Like Lasso, Elastic Net can perform feature selection by setting less important features' coefficients to zero.

**Tree-based algorithms (Random Forest, Gradient Boosting Machines):** Decision trees and ensemble methods based on decision trees (such as Random Forest and Gradient Boosting Machines) inherently perform feature selection during the training process. These algorithms select the most informative features at each split in the tree, thereby prioritizing the most relevant features in the final model.

**Linear SVM with L1 Regularization:** Support Vector Machines (SVMs) with L1 regularization can perform feature selection similarly to Lasso regression. By adding an L1 penalty term to the SVM objective function, less important features' coefficients are encouraged to be zero, effectively performing feature selection during the model training process.

**Neural Networks with Dropout:** Dropout is a regularization technique commonly used in neural networks. During training, random neurons (and thus corresponding features) are "dropped out" with a certain probability, effectively performing feature selection by ignoring less important features during training.

Embedded feature selection methods offer the advantage of integrating feature selection directly into the model training process, potentially leading to more efficient and effective feature selection. However, the choice of algorithm and its hyperparameters can significantly impact the effectiveness of embedded feature selection, and it may not always provide the best feature subset for a given problem. Therefore, it's essential to experiment with different algorithms and parameter settings to find the best-performing model.



### UnSupervised

**PCA**

Principal Component Analysis (PCA) is a widely used dimensionality reduction technique in machine learning and data analysis. It is primarily used for feature extraction and data visualization by reducing the dimensionality of the dataset while preserving most of the relevant information.

PCA works by transforming the original high-dimensional data into a new coordinate system (i.e., new set of variables) called principal components. These principal components are linear combinations of the original features and are orthogonal to each other. The first principal component captures the maximum variance in the data, the second principal component captures the maximum remaining variance orthogonal to the first component, and so on.

Here's how PCA works in more detail:

**Standardization:** If the features in the dataset have different scales, it's essential to standardize them (subtract the mean and divide by the standard deviation) to ensure that each feature contributes equally to the PCA.

**Covariance Matrix:** PCA computes the covariance matrix of the standardized data. The covariance matrix measures the pairwise covariances between the features, indicating how they vary together.

**Eigenvalue Decomposition:** PCA then performs eigenvalue decomposition (or singular value decomposition) on the covariance matrix to obtain the eigenvectors and eigenvalues. Eigenvectors represent the directions (principal components) of maximum variance in the data, while eigenvalues represent the magnitude of variance along each eigenvector.

**Selecting Principal Components:** The principal components are sorted in descending order of their corresponding eigenvalues. The top k eigenvectors (principal components) are selected to retain most of the variance in the data, where k is the desired number of dimensions in the reduced space.

**Projection:** Finally, PCA projects the original data onto the selected principal components to obtain the lower-dimensional representation of the data.

PCA is often used for:

Dimensionality reduction: By selecting only the top principal components, PCA reduces the dimensionality of the dataset while retaining most of the variance. This can help in reducing computational complexity and alleviating the curse of dimensionality.

Data visualization: PCA can be used to visualize high-dimensional data in lower-dimensional space (e.g., 2D or 3D), making it easier to explore and understand the structure of the data.

Noise reduction: PCA can help in removing noise and redundant information from the dataset by focusing on the principal components that capture the most significant variations in the data.

Overall, PCA is a powerful technique for reducing the dimensionality of data and extracting meaningful features for machine learning models. It is widely used across various domains, including image processing, signal processing, and natural language processing.



## Metrics of R-squared and Mean Squared error

| **Feature Selection Method**| R2 Score | MSE |
| -------------        | :-------------: |:------: |
| **Supervised - Filter Based Methods** |           |       |   
| Mutual Information   | 0.57   | 0.52  |
| Chi Squared   | xxx   | xxx  |
| Pearson Correlation - method 1(using the f_regression and SelectKBest class from sklearn)| 0.56 | 0.58  |
| Pearson Correlation - method 2 (using the corr() function)   | 0.52  | .60 |
| **Supervised - Wrapper Based Methods**   |    |  |
| Recursive Feature Elimination   | 0.59   | 0.55 |
| Select From Model | xxx   | xxx  |
| Sequential Feature Selection - forward selection method  | 0.54   | 0.61  |
| Sequential Feature Selection - backward selection method  | 0.61   | 0.52  |
| **Supervised - Embedded Mehtods**   |    |  |
| Lasso Regularization  | .58   | .55  |
| **UnSupervised - Dimensionality Reduction**   |    |   |
| PCA   | .01  | 1.31 |
                 

**The above quant results show that we have the best result with the backward selection method of the SequentialFeatureSelector, with a model using just 4 out of the 8 features.**

## Key Takeaways

After a feature selection process, key takeaways include:

- Enhanced model performance and generalization due to reduced dimensionality.
- Improved interpretability, making insights more actionable.
- Increased computational efficiency, particularly with large datasets.
- Valuable domain insights from selected features.
- Confidence in model predictions through rigorous validation.








## How to Run

The code is built on Jupyter notebook

```bash
Simply download the repository, upload the notebook and dataset on colab, and hit play!
```


## Roadmap
The next steps would be 

- Incorporate chosen features into model development.
- Train the model and assess its performance through rigorous evaluation.
- Fine-tune the model if necessary for optimization.
- Analyze model predictions for insights into the problem domain.
- Deploy the model and monitor its performance, iterating as needed for continuous improvement.

## Libraries 

**Language:** Python

**Packages:** Sklearn, Matplotlib, Pandas, Seaborn


## FAQ

### What is Feature Selection  and its relevance ?

Feature selection in machine learning refers to the process of selecting a subset of relevant features or variables from a larger set of features to use in model training. Features are the individual measurable properties or characteristics of the data that are used to make predictions or decisions.

Feature selection is important for several reasons:

**Curse of dimensionality:** When dealing with high-dimensional data, having too many features can lead to overfitting and increased computational complexity. Feature selection helps to mitigate this issue by reducing the number of dimensions in the data.

**Improving model performance:** By selecting only the most relevant features, we can potentially improve the performance of our machine learning models. Irrelevant or redundant features can introduce noise into the model and decrease its predictive accuracy.

**Interpretability:**  Models built with a smaller set of features are often easier to interpret and understand, both for practitioners and stakeholders. This is especially important in domains where interpretability is crucial, such as healthcare or finance.

Feature selection methods can be broadly categorized into three main types:

**Filter methods:** These methods evaluate the relevance of features based on statistical properties such as correlation, mutual information, or significance tests. Features are selected independently of the machine learning algorithm used.

**Wrapper methods:** Wrapper methods involve training multiple models with different subsets of features and selecting the subset that produces the best performance according to a chosen evaluation metric. This approach is computationally expensive but can lead to better feature subsets.

**Embedded methods:** Embedded methods incorporate feature selection directly into the model training process. Some machine learning algorithms, such as decision trees or LASSO regression, inherently perform feature selection as part of their training process.

The choice of feature selection method depends on factors such as the size and nature of the dataset, the computational resources available, and the specific goals of the machine learning project.

### How do we finally select the best set of features after the feature selection process ?

Selecting the best features from different feature selection methods involves a combination of experimentation, evaluation, and domain expertise. Here's a general process for selecting the best features:

**Understand the Problem:** Gain a thorough understanding of the problem you're trying to solve and the characteristics of your dataset. Consider factors such as the nature of the features, the size of the dataset, and the computational resources available.

**Choose Feature Selection Methods:** Select one or more feature selection methods based on the characteristics of your dataset and the requirements of your problem. Consider using a combination of filter, wrapper, and embedded methods to explore different approaches.

**Preprocessing:** Before applying feature selection methods, preprocess your data to handle missing values, normalize or standardize features, and encode categorical variables if necessary. Preprocessing ensures that the feature selection process is more effective and robust.

**Evaluate Feature Importance:** For filter methods, evaluate the importance or relevance of each feature using statistical measures such as correlation coefficients, mutual information, or significance tests. For wrapper and embedded methods, evaluate feature subsets using cross-validation or other evaluation metrics.

**Select Features:** Based on the evaluation results, select the subset of features that performs best according to your chosen evaluation metric. This subset may come from a single feature selection method or a combination of methods.

**Validate Selected Features:** Validate the selected features on a holdout dataset or using a different evaluation metric to ensure that they generalize well and improve the performance of your machine learning model.

**Iterate:** Iterate on the feature selection process by trying different combinations of feature selection methods, preprocessing techniques, and evaluation metrics. Experimentation is key to finding the best feature subset for your specific problem.

**Domain Expertise:** Finally, leverage domain expertise to interpret the selected features and understand their relevance to the problem domain. Domain knowledge can help identify meaningful patterns and relationships in the data that may not be captured by feature selection methods alone.

By following these steps and combining empirical evaluation with domain expertise, you can effectively select the best features for your machine learning models.

#### What is the California Housing Dataset?

The California Housing Dataset is a widely used dataset in machine learning and statistics. It contains data related to housing in California, particularly focusing on the state's census districts. The dataset typically includes features such as median house value, median income, housing median age, average number of rooms, average number of bedrooms, population, and geographical information like latitude and longitude.

The main objective of using this dataset is often to build predictive models, such as regression models, to predict the median house value based on other attributes present in the dataset. It's commonly used for practicing and learning regression techniques, particularly in the context of supervised learning.

This dataset has been used in various research studies, educational settings, and competitions due to its relevance to real-world problems and its accessibility for educational purposes.
## Acknowledgements


 - [pearson-correlation](https://articles.outlier.org/pearson-correlation-coefficient)
 - [github repo for handsonml-3](https://github.com/ageron/handson-ml3)
 - [EDA on the California housing dataset - kaggle notebook](https://www.kaggle.com/code/olanrewajurasheed/california-housing-dataset)
 


## Contact

If you have any feedback/are interested in collaborating, please reach out to me at vasanth1627@gmail.com


## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)