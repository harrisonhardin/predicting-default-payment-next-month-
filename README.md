# Predicting Default Payment Next Month  
  
## My Description   
  
[ This data ](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) set describes the defaulting of credit card information in Taiwan of  
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
  
## How this Project is Structured  
  
1. **preproc_n_eval.py**  
  
	This *preproc_n_eval.py* script has two parts: preprocessing the data and  
	evaluating the scores of the most effective naive logistic classifiers.  For  
	the preprocessing, the script takes in the original data  
	*default_of_credit_card_clients.csv* and will preprocess data in the  
	following ways:  
  
	* rename columns  
	* drop the id column  
	* convert the data into numeric types  
	* remove education and marriage errata   
	* one hot encode marriage status, education, and sex  
	* create a new set of columns of each month that the payment on credit was  
	  made duly or not  
  
	For the logistic classification each feature has a logistic classifier  
	predict the target class with a k fold cross validation of 10.  The results  
	are then saved to json for later interpretation in the *df_evaluation.json*.  
	Interaction terms between each of the variables are also generated and run   
	through the same classification process where the results will be interpreted  
	later in the *interaction_evaluation.json*.  
  
	Finally the all of the preprocessed data is saved to *dataframe.csv* and the  
	interactions data frame is saved as *interactions_df.csv*.  
  
2. **finalize_data.py**  
  
	The purpose of the *finalize_data.py* is to find out what the most important  
	features of the data set, including interactions terms, are.  To do this all  
	of the results from fitting naive models to each feature individually are  
	loaded in from their respective *df_evaluation.json* and  
	*interactions_df.json* files respectively.  Then the top 10% of scores in each  
	category are grabbed from the interaction terms, removing duplicates.  The  
	same is done for the main dataframe except the top 50% are taken instead.  
	Finally the data is put on a min max scale and exported to  
	*selected_features.csv*.  
  
3. **modeling.py**  
  
	The modeling strategy is for this script to first take a numeric arguement  
	corresponding the model that will be used.  Each model has a corresponding number   
	that it refers to:  
  
	* LogisticRegression  
	* GaussianMixture  
	* DecisionTreeClassifier  
	* RandomForestClassifier  
	* AdaBoostClassifier  
	* GradientBoostingClassifier  
	* KNeighborsClassifier  
  
	Then the hyper-paramter config *model_config.toml* will be read and random  
	values from the config will be reading as hyper parameters for the given  
	model.  The model is then trained and its predictions will then be  
	evaluated.  The model will then append its score to an inline dictionary as  
	a line to the *model_manifest.txt*.  The model will then be pickled and  
	saved with the filename #####model.pkl to be used for a later date.  The
	number of the model is determined by the file *models_created.txt* which
	just contains a single number of models created which is incremeneted each
	time the file is read from.
  
4. **multipython**  
	  
	is a golang binary that can run multiple python scripts at a time while also  
	managing the number of jobs, number of iterations, execution time limits,  
	and whether or not to halt all jobs on failure.  It keeps track of all  
	process it calls to run and will kill them all on keyboard interupts or  
	individually if the time of execution is too long.  
  
	```bash 
	[maximillian@ThinkPadT440 modeling]$ ./multipython -h   
	MULTIPYTHON   
	a program that multiplexes the same python script across multiple go routines   
	and iterativaly passes system arguements in the range [k,n) to it   
	Usage:   
	  -f    fatal set to false will allow other instances of scripts to continue (default true)   
	  -j int   
			number of jobs allocated to run at a single time (default 4)   
	  -k int   
			starting bound of iteration   
	  -n int   
			stopping bound of iteration (default 1)   
	  -p string   
			prgram filename (default "main.py")   
	  -t int   
			time limit in seconds for a process to finish (default 60)   
  
  
	```  
	I ran with the following configuration:  
  
	```bash
	[maximillian@ThinkPadT440 modeling]$ ./multipython -p modeling.py -t 300 -n 20000 -f=false -j 4  
	```  

5. **model_manifest.txt**

	here is a snippet from the final result.  Each row contains the model and
	the k fold of 3 cross validation results.  I had run this model for about 5
	hours straight before terminating the program.

	``` json
	[{"model": "00001LogisticRegression", "avg_accuracy": 0.7195043656908063, "avg_precision": 0.6909528188441677, "avg_f1": 0.7749920660107902, "avg_recall": 0.623356732610083}, {"model": "00001LogisticRegression", "avg_accuracy": 0.7281715459681561, "avg_precision": 0.6963132979486444, "avg_f1": 0.7821463100225589, "avg_recall": 0.6274560496380558}, {"model": "00001LogisticRegression", "avg_accuracy": 0.7250256805341551, "avg_precision": 0.7006360522820997, "avg_f1": 0.7694197113908505, "avg_recall": 0.6431412806364686}]
	[{"model": "00002DecisionTreeClassifier", "avg_accuracy": 0.7817796610169492, "avg_precision": 0.7735962166122695, "avg_f1": 0.8089997213708554, "avg_recall": 0.7411614550095724}, {"model": "00002DecisionTreeClassifier", "avg_accuracy": 0.7831920903954802, "avg_precision": 0.7704126725134272, "avg_f1": 0.812562742004876, "avg_recall": 0.7324198552223371}, {"model": "00002DecisionTreeClassifier", "avg_accuracy": 0.7762583461736005, "avg_precision": 0.7657772699778213, "avg_f1": 0.8039796782387807, "avg_recall": 0.7310406775311177}]
	[{"model": "00004GaussianMixture", "avg_accuracy": 0.49698253723677455, "avg_precision": 0.0, "avg_f1": 0.0, "avg_recall": 0.0}, {"model": "00004GaussianMixture", "avg_accuracy": 0.5033384694401644, "avg_precision": 0.0, "avg_f1": 0.0, "avg_recall": 0.0}, {"model": "00004GaussianMixture", "avg_accuracy": 0.4996789933230611, "avg_precision": 0.0, "avg_f1": 0.0, "avg_recall": 0.0}]
	[{"model": "00005RandomForestClassifier", "avg_accuracy": 0.49698253723677455, "avg_precision": 0.0, "avg_f1": 0.0, "avg_recall": 0.0}, {"model": "00005RandomForestClassifier", "avg_accuracy": 0.49666153055983564, "avg_precision": 0.6636925188743994, "avg_f1": 0.49666153055983564, "avg_recall": 1.0}, {"model": "00005RandomForestClassifier", "avg_accuracy": 0.4996789933230611, "avg_precision": 0.0, "avg_f1": 0.0, "avg_recall": 0.0}]
	[{"model": "00007LogisticRegression", "avg_accuracy": 0.722200821777093, "avg_precision": 0.6984458847306433, "avg_f1": 0.7692661958857845, "avg_recall": 0.6395660497766432}, {"model": "00007LogisticRegression", "avg_accuracy": 0.7297765793528506, "avg_precision": 0.7033617591091691, "avg_f1": 0.7732837439950411, "avg_recall": 0.6450361944157187}, {"model": "00007LogisticRegression", "avg_accuracy": 0.7250256805341551, "avg_precision": 0.6971646751042918, "avg_f1": 0.7763779527559055, "avg_recall": 0.6326190170665982}]
	[{"model": "00006GaussianMixture", "avg_accuracy": 0.49698253723677455, "avg_precision": 0.0, "avg_f1": 0.0, "avg_recall": 0.0}, {"model": "00006GaussianMixture", "avg_accuracy": 0.5033384694401644, "avg_precision": 0.0, "avg_f1": 0.0, "avg_recall": 0.0}, {"model": "00006GaussianMixture", "avg_accuracy": 0.4996789933230611, "avg_precision": 0.0, "avg_f1": 0.0, "avg_recall": 0.0}]
	```

	There are a total of 956 results successfully documented.

	```bash
	[maximillian@ThinkPadT440 modeling]$ wc -l model_manifest.txt 
	956 model_manifest.txt 
	```

	Out of 1076 models created

	```
	[maximillian@ThinkPadT440 modeling]$ cat models_created.txt 
	1076
	```

	The link for the multipython program can be found at this repository [https://github.com/maxsei/multipython](https://github.com/maxsei/multipython).
	To use the binary you must build it for your operating system using the go
	programming language.  You could add the binary to your `/usr/bin` or copy
	it to the project directory to use it.
