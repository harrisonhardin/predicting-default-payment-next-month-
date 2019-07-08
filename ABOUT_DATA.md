## Data Set Information:  
  
This research aimed at the case of customersâ€™ default payments in Taiwan and  
compares the predictive accuracy of probability of default among six data mining  
methods. From the perspective of risk management, the result of predictive  
accuracy of the estimated probability of default will be more valuable than the  
binary result of classification - credible or not credible clients. Because the  
real probability of default is unknown, this study presented the novel  
â€œSorting Smoothing Methodâ€  
  
## My Description 
  
This data sets describes the defaulting of credit card information in Taiwan of  
various accounts.  In the interest of reducing the harm caused in defaulting on  
credit cards one must take into consideration the context of the predictive  
probability of classification rather than the binary classification itself.  To  
me this means overclassifying rows towards default to a degree is more  
acceptable than a loss in accuracy.  
  
Another way of putting this is that by increasing the nubmer of false positives  
for default would mean taking a better safe than sorry approach to the problem.  
Apparrantly other researchers have applied "novel" sorting and smoothing  
methods.  
  
## Attribute Info  
* This research employed a binary variable, default payment (Yes = 1, No = 0),  
  as the response variable. This study reviewed the literature and used the  
  following 23 variables as explanatory variables  
* id: unique identifier  
* limit\_bal: Amount of the given credit (NT dollar): it includes both the individual  
  consumer credit and his/her family (supplementary) credit  
* sex: Gender (1 = male; 2 = female)  
* education: Education (1 = graduate school; 2 = university; 3 = high school; 4  
  = others)  
* marraige: Marital status (1 = married; 2 = single; 3 = others)  
* age: Age (year)  
* pay\_0  -pay\_6 ~pay\_1: History of past payment. We tracked the past monthly  
  payment records (from April to September, 2005) as follows: pay\_0 = the  
  repayment status in September, 2005; pay\_2 = the repayment status in August,  
  2005; . .  .;pay\_6 = the repayment status in April, 2005. The measurement  
  scale for the repayment status is:  -1 = pay duly; 1 = payment delay for one  
  month; 2 = payment delay for two months; . . .; 8 = payment delay for eight  
  months; 9 = payment delay for nine months and above  
* bill\_amt1 -bill\_amt6: Amount of bill statement (NT dollar). bill\_amt1 =  
  amount of bill statement in September, 2005; bill\_2 = amount of bill  
  statement in August, 2005; . . .; X17 = amount of bill statement in April,  
  2005  
* pay\_amt1 -pay\_amt6: Amount of previous payment (NT dollar). pay\_amt1 =  
  amount paid in September, 2005; pay\_amt2 = amount paid in August, 2005; . .  
  .;pay\_amt6 = amount paid in April, 2005  
  
## Feature Engineering  
  
#### Useless Features  
* removed id  
  
#### Features Pruning/Generation  
* having features that are not on the education scale as unknown(category 4)
* one hot encode education status  
* one hot encode marriage status  
* maybe log transform  each payment with experimented base? history because
  human proportionality  
* create feature for the cumulative amount owed accross all months  
  
#### Feature Transformation  
* min max age?  Remove outliers?  
  
#### What is an Outlier (Row) in the context of our Data Set  
* someone who is responsible and conservative with credit card payments  
  
  
#### Interesting Plotted Features  
* education v limit\_bal  
* sex v limit\_bal  
