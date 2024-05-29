## Title: Predicting Failure of Johannesburg Stock Exchange Companies Using Machine Learning

<br clear="both">

<div align="center">
  <img height="300" width="100%" src="https://github.com/GodfreyElia/bankruptcy_prediction_with_rawdata/blob/main/Files/Bankruptcy_Prediction.png"  />
</div>

### 1. Description

The dynamic nature of the contemporary business environment has made corporate failures commonplace, and so raised the alarm for corporation to start considering their going concern seriously. On the higher end, all sorts of businesses can benefit from knowing their likelihood of failure with a good degree of statistical certainty. In this project I adopt and compare 7 different techniques of predicting bankruptcy, namely:

  (i). Machine Learning - Ensemble
  
    1. Random Forest
    2. Gradient boosting
    3. Bagging
    
  (ii). Machine Learning - Base Learner
  
    4. Support Vector Machines (SVM)
    5. Artificial Neural Networks (ANN)

  (iii). Traditional Statistical Methods
  
    6. Logistic regression

  (iv). Other
  
    7. k-Nearest Neighbours

### 2. Data

Dataset:
   
This project has employed financial statements data of 512 Johannesburg Stock Exchange companies (270 dead and 242 alive) obtained from DataStream database. The companies represent 11 different industries including industrials, financials, and consumer services. 

Furthermore, the data spans 25 years spanning 1997 and 2022. This has been done deliberately to capture the impact of the two prominent global crises, namely: the financial crisis of 2008 and the Covid 19 pandemic of 2019, on the predictive ability of the models being studied.

<br clear="both">

<div align="Left">
  <img height="60%" width="75%" src="https://github.com/GodfreyElia/bankruptcy_prediction_with_rawdata/blob/main/Files/Industries%20Summary.png"  />
</div>
<br>

Variables:

The below lists down all the independent variables used in this project. The dependent variable was a dummy variable called 'Bankruptcy' whereby 0 represented live companies and 1 represented defunct companies.

<br clear="both">

<div align="Left">
  <img height="60%" width="75%" src="https://github.com/GodfreyElia/bankruptcy_prediction_with_rawdata/blob/main/Files/Variables.png"  />
</div>
<br>

Sampling procedure:

This project’s target population was all JSE companies currently existent on the exchange and those companies that delisted from the exchange between 1997 and 2022.  An initial sample consisting of 967 JSE companies was drawn from the DataStream database to begin with. Among the 967 companies, 721 companies had been delisted from the exchange between 1997 and 2023 for various reasons including bankruptcy. The rest of the companies, 246, were still listed on the exchange. The paper applied the following criteria to select the final pool of companies to use in model training, validation, and testing:

    1.	The company was still alive or had been delisted on the exchange between 1997 and 2022.
    2.	All the companies have at least five years of data available (see the attached image for a concession on this).
    3.	Those delisted, had been delisted because of bankruptcy and not any other causes such as mergers or being acquired by another company.

<br clear="both">

<div align="Left">
  <img height="60%" width="75%" src="https://github.com/GodfreyElia/bankruptcy_prediction_with_rawdata/blob/main/Files/Sample%20Selection%20Process.png"  />
</div>
<br>

### 3. Exploratory Data Analysis

Exploring our data allows us to form judgements on the nature of our data and distribution of our variables. This is inviriable in any statistical and modeling undertakingas as it enables the analyst to gauge the suitability of the descriptors for the task at hand. In this project, I use two main methods of exploring our data, namely: descriptive statistics and graphical visualisation. 
<br>

3.1. Descriptive Statistics

Descriptive statistics offers a tremendous way of appreciating how our data is spread across the descriptors. The main summary statistics explored in this project include measures of central tendency, location, and dispersion.

<br clear="both">

<div align="Left">
  <img height="60%" width="75%" src="https://github.com/GodfreyElia/bankruptcy_prediction_with_rawdata/blob/main/Files/Summary_Stats.jpg"  />
</div>
<br>

3.2. Graphical Visualisation

Presenting our data visually complements summary statistics by enabling us the opportunity to appreciate the distribution of our data visually. Thus we can solidify our judgements and conclusions regarding the nature of our data. In this project, I use an array of visualisations to understand the data more clearly before using it for any analysis and modeling.

  3.2.1.  Distribution of Companies by Industry
<br clear="both">

<div align="Left">
  <img height="60%" width="75%" src="https://github.com/GodfreyElia/bankruptcy_prediction_with_rawdata/blob/main/Files/Industry_Distribution.png"  />
</div>
<br>

Briefly, one can safely conclude that there are by far many financials and industrials companies represented in the dataset than any other industry. This observation is crucial as financial statements and hence variables vary significantly by industry.

  3.2.2  Age Distribution of Companies by Industry.
<div align="Left">
  <img height="60%" width="75%" src="https://github.com/GodfreyElia/bankruptcy_prediction_with_rawdata/blob/main/Files/Distribution_by_Age.png"  />
</div>
<br>

A keen inspection of the ages of the companies tells us that the data is skewed to the right. Thus, there are many younger companies

  3.2.3. Boxplots of Variables

Boxplots offer a quick way of inspecting the presence of outliers in our data which may potentially skew the results of our analysis. The below boxplots indicate the presence of ouliers in nearly all predictors with exceptions of the Age and Bankruptcy variables. However, as our data are mostly ratios, its not suprising that mean is of all the variables is centrally, towards zero.

<div align="Left">
  <img height="60%" width="75%" src="https://github.com/GodfreyElia/bankruptcy_prediction_with_rawdata/blob/main/Files/Boxplot%20of%20Numerical%20Variables.png"  />
</div>
<br>

  3.2.4.  Distribution of Return on Assets by Age

In this project, ROA has been confined to the ratio of retained earnings (as opposed to shareholder's profit) and total assets. Using this definition, we find that ROA is almost normally distributed for companies accross all age groups. Furthermore, the correlation between age and ROA is almost zero.

<div align="Left">
  <img height="60%" width="75%" src="https://github.com/GodfreyElia/bankruptcy_prediction_with_rawdata/blob/main/Files/Return%20on%20Assets.png"  />
</div>
<br>

  3.2.5.  Book Value of Equity : Total Liabilities

The ratio of a company's equity (BV) to total liabilities is a measure of solvency. This metric indicates a company's ability to manage debt and therefore deal with risk. In this project, we have compared the BVE:TLs of defunct and live companies, and we find no meaningful difference. This is more so because, gearing, as it were, is commonplace in the corporate horizon, and thus candidates of bankruptcy and non-bankruptcy alike are most likely to gear, sometimes substantially.

<div align="Left">
  <img height="60%" width="75%" src="https://github.com/GodfreyElia/bankruptcy_prediction_with_rawdata/blob/main/Files/Book_V_E-Total_Liabilities.png"  />
</div>
<br>

  3.2.6.  Current Ratio

Current ratio is one of the most important financial ratios, and it is used categorically to measure a company's liquidity and therefore its stability. By definition, current ratio assesses a company's ability to pay off its creditors (liabilites) falling due within one year using short term liquid assets. As the assessment will later prove, the current ratio has been ranked as the fourth most important descriptor and predictor of bankruptcy by the 7 models on aggregate.

<div align="Left">
  <img height="60%" width="75%" src="https://github.com/GodfreyElia/bankruptcy_prediction_with_rawdata/blob/main/Files/Current_Ratio.png"  />
</div>
<br>

  3.2.7.  Relationship Between Asset Growth and Sales Growth

In our attempt to predict failure, we resorted to measure the growth assets, and sales between period x, and period x+t, such that x+t is either the period that the company died, or this data was extracted. We find a reasonable correlation between the variables.

<div align="Left">
  <img height="60%" width="75%" src="https://github.com/GodfreyElia/bankruptcy_prediction_with_rawdata/blob/main/Files/Assets%20Growth%20Vs%20Sales%20Growth.png"  />
</div>
<br>

### 4. Statistical Testing

4.1.  Normality test:

It is crucial that we understand the nature of our data (whether parametric or non-parametric) before any subsequent statistical tests and modeling. To achieve this, I used the two popular standard normality tests: Anderson-Darling test, and Shapiro-Wilk test, it was discovered that the data were nonparametric.

  a.  AD test

<div align="Left">
  <img height="60%" width="75%" src="https://github.com/GodfreyElia/bankruptcy_prediction_with_rawdata/blob/main/Files/AD.png"  />
</div>
<br>
  *  Null hypothesis: The variable is parametric ie normally distributed.
  <br>
  *  Alternative Hypothesis : The variable is non-parametric ie not normally distributed.
  <br>
  *  Conclusion: Reject the null hypothesis in favour of the alternative hypothesis.
  <br>
  <br>
  b. Shapiro - Wilk Test
  <br>
<div align="Left">
  <img height="60%" width="75%" src="https://github.com/GodfreyElia/bankruptcy_prediction_with_rawdata/blob/main/Files/SW.png"  />
</div>
<br>
  *  Null hypothesis: The variable is parametric ie normally distributed.
  <br>
  *  Alternative Hypothesis : The variable is non-parametric ie not normally distributed.
  <br>
  *  Conclusion: Reject the null hypothesis in favour of the alternative hypothesis.
  <br>
  <br>

4.2. Significance tests

The aim of this test is understand how important is each variable in determining the likelihood of corporate failure. We used a range of tests, the first two being the Spearmans paired t-test (inspired by the fact that our dependent variables are non-parametric) and Principle Component Analysis (CPA). Later on, I considered three more methods which involved asking the prediction models which variables they found most useful.

  a. Spearman's Paired T-test

<br clear="both">

<div align="Left">
  <img height="60%" width="75%" src="https://github.com/GodfreyElia/bankruptcy_prediction_with_rawdata/blob/main/Files/Spearman.png"  />
</div>
<br>

Using the Spearmans paired t-test (predictor ~ bankruptcy) we find all variables except the solvency and retained earnings to total assets ratios to be significant in explaining bankruptcy at the 99.99% confidence level.

  b. Principal Component Analysis - PCA

<br clear="both">

<div align="Left">
  <img height="60%" width="75%" src="https://github.com/GodfreyElia/bankruptcy_prediction_with_rawdata/blob/main/Files/PCA_2_Dimensions.png"  />
</div>

<br clear="both">

<div align="Left">
  <img height="60%" width="75%" src="https://github.com/GodfreyElia/bankruptcy_prediction_with_rawdata/blob/main/Files/PCA1.png"  />
</div>
<br>

The fundamental aim of PCA is to reduce the dimensions of a dataset into a smaller set of uncorrelated variables which capture most of the variance in the dataset. Because PCA can adequately identify and linearly combine variables that retain the most explanatory power in a data sample, I have adopted it in this project to aid in feature selection. PCA has reduced the original dataset of 11 quantitative descriptors into a 9-principal components dataset. A closer examination of the first and second principal components (which together explain about 36% of the variance) indicates that EBIT:Total Assets and Retained Earnings: Total Assets, for instance, have significant explanatory powers compared to the rest of the variables.

The figure below shows the Eigenvalues of the PCA

<br clear="both">

<div align="Left">
  <img height="60%" width="75%" src="https://github.com/GodfreyElia/bankruptcy_prediction_with_rawdata/blob/main/Files/PCA_EV.png"  />
</div>
<br>

  4.3.  Correlation

The last test we will perform before attempting to build our models for predicting bankruptcy is correlation test. Here, we will attempt to remove any variables that are strongly correlated as they may compromise the predictive strength of our models.

<br clear="both">

<div align="Left">
  <img height="60%" width="75%" src="https://github.com/GodfreyElia/bankruptcy_prediction_with_rawdata/blob/main/Files/Corr1.png"  />
</div>
<br>

The image below makes it more clear to see where correlations are high, positive or negative. Overall, I determine that the data is not strongly correlated and as such all variables can be used in modelling.
<br clear="both">

<div align="Left">
  <img height="60%" width="75%" src="https://github.com/GodfreyElia/bankruptcy_prediction_with_rawdata/blob/main/Files/Corr2.png"  />
</div>
<br>

SECTION II: Machine Learning Modeling
----

#### Workflow and Data balance
<br clear="both">

<div align="Left">
  <img height="60%" width="75%" src="https://github.com/GodfreyElia/Bankruptcy-Prediction-Using-Machine-Learning/blob/main/Files/MachineLStructure.png"  />
</div>
<br>

Source: Basavaraju et al. (2019)

<br>

We use the above workflow to train, validate and test our models. In every prediction or classification enterprise, it is pivotal to ensure that our data is trained and tested using separate datasets drawn, if possible, from the same population. This prevents our models from overtly memorising the underlying data when making predictions. I have thus included the below diagram to show how our data is balance between the training, and testing datasets.

<br clear="both">

<div align="Left">
  <img height="60%" width="75%" src="https://github.com/GodfreyElia/Bankruptcy-Prediction-Using-Machine-Learning/blob/main/Files/Balance.png"  />
</div>
<br>

#### Performance evaluation

In machine learning, the choice of the performace evaluation metric is as important as the model itself. Indeed different scholars and practitioners of the Machine Learning school recommend several evaluation metrics to be used to evaluate prediction models. Therefore, for this project I have sought to apply 7 different machine learning performance metrics to reduce interpretation error and to enhance comparability of the different models. See below the definition of the metrics and section 5 for the specific model scores.

<br clear="both">

<div align="Left">
  <img height="60%" width="75%" src="https://github.com/GodfreyElia/Bankruptcy-Prediction-Using-Machine-Learning/blob/main/Files/Metrics.png"  />
</div>
<br>

### 5. Models

In this project, I have used Rstudio (R language) to implement the machine learning models. The models were trained on 70 per cent of the sample data, and tested on the remaining 30 percent. For Support Vector Machines and Artifical Neural Networks, the data were first normalised before being applied to modelling. Furthermore, for the Random Forest model, two models with different choices of hyperparameters were used, and the final prediction was the average predictioon of the two models.

<br clear="both">

<div align="Left">
  <img height="60%" width="75%" src="https://github.com/GodfreyElia/Bankruptcy-Prediction-Using-Machine-Learning/blob/main/Files/Methods.png"  />
</div>
<br>

  5.1. Random Forest

Random forest is a supervised ensemble learning algorithm that combines multiple decision trees to make 
a prediction (Breiman, 2001). The basic idea behind the  random forest algorithm is to create a forest of decision trees where each tree is trained on a subset of the original data, and each branch is based on a randomly selected subset of features (Breiman, 2001). 

<br clear="both">

<div align="Left">
  <img height="60%" width="60%" src="https://github.com/GodfreyElia/Bankruptcy-Prediction-Using-Machine-Learning/blob/main/Files/RF.png"  />
</div>
<br>

  5.2. Boosting Model

Boosting involves training multiple models sequentially, and each subsequent model focuses on retraining on the instances that the previous models failed to learn and the results from each model are aggregated to form a final prediction (Hastie, Tibshirani & Friedman, 2009). Furthermore, like bagging, boosting can be used with a variety of different base models, including decision trees, linear regression, and logistic regression. Boosting can be used to improve the accuracy of a model, especially when there is high bias in the data. Bias is used in machine learning and statistics to mean the difference between the predicted and the actual value (Pawełek, 2019).

<br clear="both">

<div align="Left">
  <img height="60%" width="60%" src="https://github.com/GodfreyElia/Bankruptcy-Prediction-Using-Machine-Learning/blob/main/Files/Boost.png"  />
</div>
<br>
  5.3. Bagging Model

Bagging is an ensemble learning method that stands for bootstrap aggregating, and it involves creating multiple models on bootstrap samples of the data and combining their results to make a prediction (Hastie, Tibshirani &Friedman, 2009). Bagging can be used with a variety of different base models, including decision trees, linear regression, and logistic regression (Breimann, 1996). Bagging can improve the accuracy and robustness of a model, especially when there is high variance in the data (Breimann, 1996).

<br clear="both">

<div align="Left">
  <img height="60%" width="60%" src="https://github.com/GodfreyElia/Bankruptcy-Prediction-Using-Machine-Learning/blob/main/Files/Bag.png"  />
</div>
<br>
  5.4. Support Vector Machines

Support Vector Machines, or SVMs, are a widely used machine learning model, particularly for classification tasks. SVMs work by finding the hyperplane that separates the data into different classes with the maximum margin (Hua et al., 2007). The main advantage of SVMs is their ability to handle high-dimensional datasets and produce accurate predictions (Pal et al., 2016; Hua et al., 2007; & Mitchell, 1997). They are also very good at handling datasets with complex boundaries between classes (Hua et al., 2007). Furthermore, SVMs are resistant to overfitting, which is a common problem in machine learning.

<br clear="both">

<div align="Left">
  <img height="60%" width="60%" src="https://github.com/GodfreyElia/Bankruptcy-Prediction-Using-Machine-Learning/blob/main/Files/SVM.png"  />
</div>
<br>
  5.5. Artifical Neural Networks

Artificial Neural Networks (ANN) is a machine learning model that is inspired by the structure of the human brain (James et al., 2013). ANN is highly versatile and can be used for a wide range of machine learning problems, including classification, regression, and unsupervised learning (James et al., 2013) ANN can also handle very large datasets with high-dimensional feature spaces.

<br clear="both">

<div align="Left">
  <img height="60%" width="60%" src="https://github.com/GodfreyElia/Bankruptcy-Prediction-Using-Machine-Learning/blob/main/Files/ANN.png"  />
</div>
<br>
  5.6.  k-Nearest Neighbours

KNN is a type of lazy learning algorithm, meaning that it does not learn a model from the training data, but rather it memorises all the training data points and uses them to make predictions (Alpaydin, 2010). The KNN algorithm works by finding the K-nearest neighbours to a given data point in the training set and using their labels to make a prediction for that data point (Alpaydin, 2010). The value of K refers to the number of neighbours that the model should look, and larger values of K tend to produce smoother decision boundaries, while smaller values of K can lead to overfitting.

<br clear="both">

<div align="Left">
  <img height="60%" width="60%" src="https://github.com/GodfreyElia/Bankruptcy-Prediction-Using-Machine-Learning/blob/main/Files/kNN.png"  />
</div>
<br>
  5.7. Logistic Regression

Owing to its simplicity and interpretability, logistic regression is a choice algorithm for making binary predictions. As a linear model, logistic regression uses a sigmoid function to transform the output of a linear combination of input features into a probability of belonging to a particular class (Hastie, Tibshirani, & Friedman, 2009).

<br clear="both">

<div align="Left">
  <img height="60%" width="60%" src="https://github.com/GodfreyElia/Bankruptcy-Prediction-Using-Machine-Learning/blob/main/Files/Logit.png"  />
</div>
<br>

#### Model Receiver Operating Characteristic (ROC)

The final metric used, and here below presented separately, to evaluate the models is the ROC curve. In simple terms, the ROC curve measures (reveals) how much a given classification model is better than predicting with no model at all i.e. guessing outcomes (Pal et al., 2016). The ROC curve trades off between the sensitivity and the type 1 error of the model. The area under the ROC curve is called AUC. The higher the AUC the better the predictive strength of the model.

<br clear="both">

<div align="Left">
  <img height="60%" width="60%" src="https://github.com/GodfreyElia/Bankruptcy-Prediction-Using-Machine-Learning/blob/main/Files/Out_of_Sample_ROC.png"  />
</div>
<br>

### 6. Goodness of Fit

To measure the goodness of fit of the models, I applied two main metrics: the brier score (BS), and AUCs of the models when tested on training data (in-sample performance) and on test data (out-of-sample performance). Briefly, the brier score is the mean squared difference between the predicted probabilities and the actual outcomes (Pal et al., 2016). The brier score is analogous to the mean squared error (MSE) in regression algorithms. The criteria outlined in the diagram below has been applied to identify a good fit, an underfit, and an overfit model. For reference, a high performance is considered to have an AUC of 80% or above, and a brier score (BS) of 20% or less (Fawcett, 2006; Vickers & Elkin, 2006). The higher (lower) the AUC score (brier score) the better the fit.

<br clear="both">

<div align="Left">
  <img height="60%" width="60%" src="https://github.com/GodfreyElia/Bankruptcy-Prediction-Using-Machine-Learning/blob/main/Files/BS%20~%20AUC%201.png"  />
</div>
<br>

Criteria:

<br clear="both">

<div align="Left">
  <img height="60%" width="60%" src="https://github.com/GodfreyElia/Bankruptcy-Prediction-Using-Machine-Learning/blob/main/Files/CFI.png"  />
</div>
<br>

Source: CFI, 2022.

### 7. Model Ranking

The most important question to answer at this stage of the project is perhaps: which classifier is the best at generalising labels? To tackle this question empirically, I have employed 2 methods to quantitatively rank the performance of the models. These methods are the De Long test and the Brier Score (BS). While all these techniques are quantitative and may ignore the qualitative aspects of the classifiers’ abilities, future projects will focus on methods that consider such factors tas he consistency of the models across time and their resilience to changes in data size.

  7.1.  De Long Test
  
The De Long test calculates the difference in the area under the ROC curve (AUC) between two classifiers and uses this difference to construct a z-statistic and calculate a p-value. The null hypothesis for the De Long test is that the AUCs of the two classifiers are equal, and the alternative hypothesis is that they are not equal. This paper used the logistic regression classifier as the benchmark. Furthermore, the models were then ranked based on how much they outperformed the logit classifier. Based on this criterion, the De Long test ranked the models as follows, from best to least: RF, Boosting, Bagging, SVM, ANN, kNN, LR. Notice: the Random Forest model and the Boosting model tied on the first place.

<br clear="both">

<div align="Left">
  <img height="60%" width="60%" src="https://github.com/GodfreyElia/Bankruptcy-Prediction-Using-Machine-Learning/blob/main/Files/Ranking1.png"  />
</div>
<br>

  7.2. Brier Score

As I defined previously, the brier score is the mean squared difference between the predicted probabilities and the actual outcomes (source: Pal et al., 2016). The Brier score always ranges from 0 to 1, with lower values indicating better predictive accuracy. Using the brier score, the models were ranked as follows, from most accurate to least accurate at 1 year prior to bankruptcy: RF, Boosting, Bagging, SVM, ANN, kNN, LR.

<br clear="both">

<div align="Left">
  <img height="60%" width="60%" src="https://github.com/GodfreyElia/Bankruptcy-Prediction-Using-Machine-Learning/blob/main/Files/Panel%20B.png"  />
</div>
<br>

#### Verdict:

From the above, we can observe that the ensemble models always rank in top 3, making it safe to conclude that they are better classifiers. Finally, the logit model which was the only pure statistical model always rank last.

----

Thank you!

Please check my other projects [here:](https://github.com/GodfreyElia/GodfreyElia/tree/main)

