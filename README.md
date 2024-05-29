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

We use the above workflow to train, validate and test our model. In every prediction or classification enterprise, it is pivotal to ensure that our data is trained and tested using separate datasets drawn, if possible, from the same population. This prevents our models from overtly memorising the underlying data when making predictions. I have thus included the below diagram to show how our data is balance between the training, and testing datasets.

<br clear="both">

<div align="Left">
  <img height="60%" width="75%" src="https://github.com/GodfreyElia/Bankruptcy-Prediction-Using-Machine-Learning/blob/main/Files/Balance.png"  />
</div>
<br>

### 5. Models

  5.1. Random Forest

Random forest is a supervised ensemble learning algorithm that combines multiple decision trees to make 
a prediction (Breiman, 2001). The basic idea behind the  random forest algorithm is to create a forest of decision trees where each tree is trained on a subset of the original data, and each branch is based on a randomly selected subset of features (Breiman, 2001).

