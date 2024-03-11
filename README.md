# feature_selection
 Trying out the different feature selection methods on the California housing dataset
<img src="https://github.com/Anmol-Baranwal/Cool-GIFs-For-GitHub/assets/74038190/b3fef2db-e671-4610-bb84-1d65533dc5fb" width="300" align='center'>

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

## What is univariate feature selection ?

Univariate feature selection is a type of feature selection method that evaluates each feature individually based on univariate statistical tests. In this approach, features are considered independently of each other, and their relevance to the target variable is assessed one at a time.

The process typically involves calculating a statistical metric (such as chi-square, F-test, or mutual information) to quantify the relationship between each feature and the target variable. Features are then ranked or scored based on their individual importance, and a predetermined number of top-ranked features are selected for inclusion in the final feature subset.

Univariate feature selection is relatively simple and computationally efficient, making it suitable for high-dimensional datasets. However, it may overlook interactions between features, which could affect model performance. It's often used as a preprocessing step before applying more sophisticated feature selection techniques or building machine learning models.

## Evaluation and Results

** Supervised **

#### Filter Based Methods
** Mutual Information **
![alt text](https://github.com/vasanthgx/house_prices/blob/main/images/hist1.png)
** Chi Squared **
![alt text](https://github.com/vasanthgx/house_prices/blob/main/images/hist1.png)
** Pearson Correlation **
![alt text](https://github.com/vasanthgx/house_prices/blob/main/images/hist1.png)


#### Wrapper Based Methods
** Recursive Feature Elimination ( R F E ) **
![alt text](https://github.com/vasanthgx/house_prices/blob/main/images/corr2.png)

** Select From Model **
![alt text](https://github.com/vasanthgx/house_prices/blob/main/images/hist1.png)

** Sequential Feature Selection **
![alt text](https://github.com/vasanthgx/house_prices/blob/main/images/hist1.png)

#### Embedded Methods
** Lasso Regularization **
![alt text](https://github.com/vasanthgx/house_prices/blob/main/images/corr1.png)

** Random Forest **
![alt text](https://github.com/vasanthgx/house_prices/blob/main/images/hist1.png)


### UnSupervised

** PCA **
![alt text](https://github.com/vasanthgx/house_prices/blob/main/images/hist1.png)



## Metrics of R-squared and Mean Squared error

| Feature Selection Method| R2 Score | MSE |
| -------------        | ------------- |------ |
| Supervised - Filter Based Methods |           |       |   
| Mutual Information   | xxx   | xxx  |
| Chi Squared   | xxx   | xxx  |
| Pearson Correlation   | xxx   | xxx  |
| Supervised - Wrapper Based Methods   |    |  |
| Recursive Feature Elimination   | xxx   | xxx  |
| Select From Model | xxx   | xxx  |
| Sequential Feature Selection  | xxx   | xxx  |
| Supervised - Embedded Mehtods   |    |  |
| Lasso Regularization  | xxx   | xxx  |
| Random Forest   | xxx   | xxx  |
| UnSupervised - Dimensionality Reduction   |    |   |
| PCA   | xxx   | xxx  |
                 

The above quant results show that we have an error of 54,000 USD betweeen the prediction and the actual house price.

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

#### What is Feature Selection  and its relevance ?

Feature selection in machine learning refers to the process of selecting a subset of relevant features or variables from a larger set of features to use in model training. Features are the individual measurable properties or characteristics of the data that are used to make predictions or decisions.

Feature selection is important for several reasons:

** Curse of dimensionality: ** When dealing with high-dimensional data, having too many features can lead to overfitting and increased computational complexity. Feature selection helps to mitigate this issue by reducing the number of dimensions in the data.

** Improving model performance: ** By selecting only the most relevant features, we can potentially improve the performance of our machine learning models. Irrelevant or redundant features can introduce noise into the model and decrease its predictive accuracy.

** Interpretability: **  Models built with a smaller set of features are often easier to interpret and understand, both for practitioners and stakeholders. This is especially important in domains where interpretability is crucial, such as healthcare or finance.

Feature selection methods can be broadly categorized into three main types:

** Filter methods:** These methods evaluate the relevance of features based on statistical properties such as correlation, mutual information, or significance tests. Features are selected independently of the machine learning algorithm used.

** Wrapper methods: ** Wrapper methods involve training multiple models with different subsets of features and selecting the subset that produces the best performance according to a chosen evaluation metric. This approach is computationally expensive but can lead to better feature subsets.

** Embedded methods: ** Embedded methods incorporate feature selection directly into the model training process. Some machine learning algorithms, such as decision trees or LASSO regression, inherently perform feature selection as part of their training process.

The choice of feature selection method depends on factors such as the size and nature of the dataset, the computational resources available, and the specific goals of the machine learning project.

#### How do we finally select the best set of features after the feature selection process ?

Selecting the best features from different feature selection methods involves a combination of experimentation, evaluation, and domain expertise. Here's a general process for selecting the best features:

** Understand the Problem:** Gain a thorough understanding of the problem you're trying to solve and the characteristics of your dataset. Consider factors such as the nature of the features, the size of the dataset, and the computational resources available.

** Choose Feature Selection Methods:** Select one or more feature selection methods based on the characteristics of your dataset and the requirements of your problem. Consider using a combination of filter, wrapper, and embedded methods to explore different approaches.

** Preprocessing: ** Before applying feature selection methods, preprocess your data to handle missing values, normalize or standardize features, and encode categorical variables if necessary. Preprocessing ensures that the feature selection process is more effective and robust.

** Evaluate Feature Importance: ** For filter methods, evaluate the importance or relevance of each feature using statistical measures such as correlation coefficients, mutual information, or significance tests. For wrapper and embedded methods, evaluate feature subsets using cross-validation or other evaluation metrics.

** Select Features: ** Based on the evaluation results, select the subset of features that performs best according to your chosen evaluation metric. This subset may come from a single feature selection method or a combination of methods.

** Validate Selected Features: ** Validate the selected features on a holdout dataset or using a different evaluation metric to ensure that they generalize well and improve the performance of your machine learning model.

** Iterate: ** Iterate on the feature selection process by trying different combinations of feature selection methods, preprocessing techniques, and evaluation metrics. Experimentation is key to finding the best feature subset for your specific problem.

** Domain Expertise: ** Finally, leverage domain expertise to interpret the selected features and understand their relevance to the problem domain. Domain knowledge can help identify meaningful patterns and relationships in the data that may not be captured by feature selection methods alone.

By following these steps and combining empirical evaluation with domain expertise, you can effectively select the best features for your machine learning models.

#### What is the California Housing Dataset?

The California Housing Dataset is a widely used dataset in machine learning and statistics. It contains data related to housing in California, particularly focusing on the state's census districts. The dataset typically includes features such as median house value, median income, housing median age, average number of rooms, average number of bedrooms, population, and geographical information like latitude and longitude.

The main objective of using this dataset is often to build predictive models, such as regression models, to predict the median house value based on other attributes present in the dataset. It's commonly used for practicing and learning regression techniques, particularly in the context of supervised learning.

This dataset has been used in various research studies, educational settings, and competitions due to its relevance to real-world problems and its accessibility for educational purposes.
## Acknowledgements


 - ![Hands on machine learning - by Geron](https://github.com/vasanthgx/house_prices/blob/main/images/bookcover.jpg)
 - [github repo for handsonml-3](https://github.com/ageron/handson-ml3)
 - [EDA on the California housing dataset - kaggle notebook](https://www.kaggle.com/code/olanrewajurasheed/california-housing-dataset)
 


## Contact

If you have any feedback/are interested in collaborating, please reach out to me at vasanth1627@gmail.com


## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)