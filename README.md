## Title: Predicting Failure of Johannesburg Stock Exchange Companies Using Machine Learning

#### Description
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

#### Data

Dataset:
   
This project has employed financial statements data of 512 Johannesburg Stock Exchange companies (270 dead and 242 alive) obtained from DataStream database. The companies represent 11 different industries including industrials, financials, and consumer services. 

Furthermore, the data spans 25 years spanning 1997 and 2022. This has been done deliberately to capture the impact of the two prominent global crises, namely: the financial crisis of 2008 and the Covid 19 pandemic of 2019, on the predictive ability of the models being studied.

<br clear="both">

<div align="Left">
  <img height="60%" width="60%" src="https://github.com/GodfreyElia/bankruptcy_prediction_with_rawdata/blob/main/Files/Industries%20Summary.png"  />
</div>
<br>

Variables:

The below lists down all the independent variables used in this project. The dependent variable was a dummy variable called 'Bankruptcy' whereby 0 represented live companies and 1 represented defunct companies.

<br clear="both">

<div align="Left">
  <img height="60%" width="60%" src="https://github.com/GodfreyElia/bankruptcy_prediction_with_rawdata/blob/main/Files/Variables.png"  />
</div>



    

