'''
        This project is done by Indresh Muniraj, Student ID - 873559931.
        The project is Kaggle Titanic Survival Prediction taken from Kaggle. The dataset is also taking from 
    kaggle. The train.csv contains information of the passengers like Name, Age, Sex, PClass, SibSp, 
    Parch, Embarked, Fare, Ticket, cabin and survived. The test.csv will contain all inforamtion as
    of train.csv except survived. The survival prediction is done on test.csv dataset.
        This project is about prediction of survivality of passengers using between machine learning
    algorithms. The algorithms used in this are logistic regression, random forest and xgboost model.
        ~Logistic regression is the appropriate regression analysis to conduct when the dependent variable 
    is dichotomous (binary).  Like all regression analyses, the logistic regression is a predictive 
    analysis. 
        ~Random forests or random decision forests is an ensemble learning method for classification, 
    regression and other tasks that operates by constructing a multitude of decision trees at training 
    time. For classification tasks, the output of the random forest is the class selected by most trees. 
    For regression tasks, the mean or average prediction of the individual trees is returned.
        ~XGBoost is a scalable and highly accurate implementation of gradient boosting that pushes the 
    limits of computing power for boosted tree algorithms, being built largely for energizing machine 
    learning model performance and computational speed.
'''


import numpy as np       # to work on large set of numeric datatypes.
import pandas as pd      # to work on dataframe
import seaborn as sns    # used for visualization
import matplotlib.pyplot as plt   # used to plot graphs
from sklearn.preprocessing import StandardScaler   # to scale the dataframe to unit variance
from sklearn.model_selection import cross_val_score  # used for evaluation performance, returning list of
                                                        # scores for each fold. 
from sklearn.linear_model import LogisticRegression  # implementation of logistic regression    
from sklearn.ensemble import RandomForestClassifier  # implementation of Random Forest Model
from xgboost import XGBClassifier      # implementation of XGBoost Model.
from sklearn.ensemble import VotingClassifier      # ML model that trains numerous models and predicts an 
                                                    # output based on highest probability.
from sklearn.model_selection import GridSearchCV   # loops through the parameters and fits the model
                                                        # on training set.

train_data = pd.read_csv("train.csv")       # contains passenger information including survived
test_data = pd.read_csv("test.csv")     # contain passsenger information encluding survived

'''
    Adding to column to the dataset calling training for both train_data and test_data and the value
    is 1 and 0 respectively to differentiate them. Then survived column is add to test_data and values
    are updated as Null. Then combinig both dataset into a single dataset to gather as much information
    as possible.
'''
train_data['training'] = 1
test_data['training'] = 0
test_data['Survived'] = np.NaN
combined_data = pd.concat([train_data,test_data])

print(train_data.info())    # print info about datatype and non null count for each column.
print(train_data.isna().sum())  # print info about null count for each column.
print(train_data.columns)       # prints column names of train_data.
print(test_data.columns)        # print column names of test_data.
print(combined_data.columns)    # print column names of combined_data.

'''
Survived
    To show a bar-chart of how many passengers survived or not the ship wreck.
'''
sns.barplot(x = train_data["Survived"].value_counts().index, y = train_data["Survived"].value_counts()).set_title("Survived")
plt.show()

'''
Pclass
    To see the how many passengers belong the 1st or 2nd or 3rd class.
    Then printing the survival rate for each class.
'''
sns.barplot(x = train_data["Pclass"].value_counts().index,y = train_data["Pclass"].value_counts()).set_title("Pclass")
plt.show()
train_pclass = train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(train_pclass)

'''
Age
    To see passenger distribution via the age using histogram and bar chart. The bar chart is not very 
    clear as there are too many values present but the histogram is much better. From the graph we can 
    say that passengers median age is around 30 ages. 
    The survival rate based on the age tells that younger passengers have higher chance of survival
    than elder passengers.
'''
plt.hist(train_data["Age"])
plt.title("Age")
plt.show()
train_age = pd.pivot_table(train_data, index = 'Survived', values = ['Age'])
print(train_age)

'''
Sex
    To see check which gender had high chance of survival. From the graph, we can see that there were
    nearly twice the male passenger than the female passengers. But the survival rate is more for 
    female passengers than compared to male passenegers.
'''
sns.barplot(x = train_data["Sex"].value_counts().index,y = train_data["Sex"].value_counts()).set_title("Sex")
plt.show()
train_sex = train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(train_sex)

'''
SibSp
    This information is about number of siblings and spouse that acompanied the passengers. For the
    graph we can see that there were many passenger who were alone or had another passengers
    acompanying them. The survived rate shows that passengers with same family has high survival rate.
'''
plt.hist(train_data["SibSp"])
plt.title("SibSp")
plt.show()
train_Sibsp = pd.pivot_table(train_data, index = 'Survived', values = ['SibSp'])
#print(train_Sibsp)

'''
Parch
    This information is about the passengers parents or children accompanying them. This also has same
    result as that of Sibsp. Passengers with smaller or no family has high rate of survival.
'''
plt.hist(train_data["Parch"])
plt.title("Parch")
plt.show()
train_Parch = pd.pivot_table(train_data, index = 'Survived', values = ['Parch'])
#print(train_Parch)

train_family = pd.pivot_table(train_data, index = 'Survived', values = ['SibSp', 'Parch'])
print(train_family)

'''
Ticket
    To see how the ticket is distributed among the passengers in bar chart.
    The survival rate is not providing proper result since these is to many tickets groups.
'''
sns.barplot(x = train_data["Ticket"].value_counts().index,y = train_data["Ticket"].value_counts()).set_title("Ticket")
plt.show()
train_ticket = train_data[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(train_ticket)

'''
Embarked
    Using bar chart, to see at which place the passenger enter the ship. From the graph we can see that
    most of the passengers entered from S port.
    The survival rate shows that passengers who embarked at C had higher chance of survival than
    compare to other two embarking ports.
'''
sns.barplot(x = train_data["Embarked"].value_counts().index,y = train_data["Embarked"].value_counts()).set_title("Embarked")
plt.show()
train_Embarked = train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(train_Embarked)

'''
Fare
    To see fare range the passenger brought based on class and cabin.
    From the survival rate, we can say that the passengers who paid higher fare had higher chance
    of survival.
'''
sns.barplot(x = train_data["Fare"].value_counts().index,y = train_data["Fare"].value_counts()).set_title("Fare")
plt.show()
plt.hist(train_data["Fare"])
plt.title("Fare")
plt.show()
train_fare = pd.pivot_table(train_data, index = 'Survived', values = ['Fare'])
print(train_fare)

'''
Cabin
    From the graph we don't get much information as there are too many cabin information.
    The survival rate is also not much informativeas there are too many cabin information.
'''
sns.barplot(x = train_data["Cabin"].value_counts().index, y = train_data["Cabin"].value_counts()).set_title("Cabin")
plt.show()
train_cabin = train_data[['Cabin', 'Survived']].groupby(['Cabin'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(train_cabin)

'''
Feature Engineering on Cabin
    Here I am grouping the cabin values based on the number of rooms or based the passenger has taken,
    called the cabin_grouping. By doing this, the cabin group is down to 5 groups. Next I am checking
    the survival rate based the grouping done.
    Again the cabin values are grouped on bases of cabin letter. Here the cabin values can be grouped 
    into 9 groups. The group n indicates the passenger cabin value is missing and the missing values
    is considered as a sepearte group. Also the survival rate is checked.
'''
train_data['cabin_grouping'] = train_data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
print(train_data['cabin_grouping'].value_counts())  

print(pd.pivot_table(train_data, index = 'Survived', columns = 'cabin_grouping', values = 'Ticket' ,aggfunc ='count'))

train_data['cabin_letter'] = train_data.Cabin.apply(lambda x: str(x)[0])
print(train_data['cabin_letter'].value_counts())
print(pd.pivot_table(train_data,index='Survived',columns='cabin_letter', values = 'Name', aggfunc='count'))

sns.barplot(x = train_data["cabin_letter"].value_counts().index, y = train_data["cabin_letter"].value_counts()).set_title("Cabin_letter")
plt.show()

'''
Feature Engineering on Ticket
    The ticket values is first grouped based on the passenger having the ticket or not called 
    ticket_number. If the passenger does not have ticket, considering it as a null value. 
    Then grouping based on the ticket letter.
    In ticket_number, we 230 values missing. Based the ticket_letter, we get many groups.
    Then checking for survival rate for both grouping and plotting group for both groups.    
'''
train_data['ticket_number'] = train_data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
train_data['ticket_letter'] = train_data.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)
print(train_data['ticket_number'].value_counts())
print(train_data['ticket_letter'].value_counts())

print(pd.pivot_table(train_data,index='Survived',columns='ticket_number', values = 'Ticket', aggfunc='count'))
print(pd.pivot_table(train_data,index='Survived',columns='ticket_letter', values = 'Ticket', aggfunc='count'))

sns.barplot(x = train_data["ticket_number"].value_counts().index,y = train_data["ticket_number"].value_counts()).set_title("ticket_number")
plt.show()
sns.barplot(x = train_data["ticket_letter"].value_counts().index,y =  train_data["ticket_letter"].value_counts()).set_title("ticket_letter")
plt.show()

'''
Feature Engineering on Name
    Here the names are grouped on the based of their titles. We get 17 groups based on their title. 
    Also checking the survival rate based on the titles. A graph is also plotted.
'''
train_data['title_group'] = train_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
print(pd.pivot_table(train_data,index='Survived',columns='title_group', values = 'Name', aggfunc='count'))
print(train_data['title_group'].value_counts())
sns.barplot(x = train_data["title_group"].value_counts().index,y = train_data["title_group"].value_counts()).set_title("title_group")
plt.show()


'''
Processing for model
    Doing the same feature engineering of Cabin, ticket and Name on the combined dataset.
'''
combined_data['cabin_grouping'] = combined_data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
combined_data['cabin_letter'] = combined_data.Cabin.apply(lambda x: str(x)[0])
combined_data['ticket_number'] = combined_data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
combined_data['ticket_letter'] = combined_data.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)
combined_data['title_group'] = combined_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

'''
The details of the feature engineering that was done on the combined dataset.
print(combined_data['cabin_grouping'].value_counts())
print(combined_data['cabin_letter'].value_counts())
print(combined_data['ticket_number'].value_counts())
print(combined_data['ticket_letter'].value_counts())
print(combined_data['title_group'].value_counts())'''

'''
    Updating the missing or null values of Age and Fare column with mean values of the column
    respectively.
'''
combined_data.Age = combined_data.Age.fillna(train_data.Age.mean())
combined_data.Fare = combined_data.Fare.fillna(train_data.Fare.mean())

combined_data.dropna(subset=['Embarked'],inplace = True)  # dropping the missing or null values from
                                                           # embarked cloumn.
'''
    Using logarithm on fare values to normalize the values to give more semblance of a normal
    distribution.
'''
combined_data['norm_fare'] = np.log(combined_data.Fare+1)
combined_data['norm_fare'].hist()
plt.show()

''' coverting the Pclass values to category for panda dummy variables. '''
combined_data.Pclass = combined_data.Pclass.astype(str)

''' using One hot encoder, creating panda dummy variables from categorical datatypes'''
combined_dummies = pd.get_dummies(combined_data[['Pclass','Sex','Age','SibSp','Parch','norm_fare','Embarked','cabin_letter','cabin_grouping','ticket_number','title_group','training']])

'''
    Separating the combined dataset of panda dummy variables into train_x and test_y based on the column
    training. If training is equal to 1, assigning it to train_x else assigning it to test_x. Also,
    creating another variable called train_y that contains survival inforamtion of train dataset.
    These variables will used in training the different algorithm is generate the prediction.
'''
train_x = combined_dummies[combined_dummies.training == 1].drop(['training'], axis =1)
test_x = combined_dummies[combined_dummies.training == 0].drop(['training'], axis =1)
train_y = combined_data[combined_dummies.training==1].Survived
train_y.shape
plt.show()

'''
Scaling data
    StandardScaler() - Standardize features by removing the mean and scaling to unit variance.
    First, copy the panda variables to a variable called combined_dummies_scaled. Then for column
    values of Age, SibSp, Parch and norm_fare applying the StandardScaler(). Then printing the 
    StandardScalar() values.
    Here, we again separate the combined_dummies_scaled into train_x_scaled andtest_x_scaled based on
    the column called training. If training value is 1 then assigning it to train_x_scaled else
    assigning the value to test_x_scaled. Also assigning the survived values ti train_y.
    These variables will be used during the cross validation on each of the modeling algorithm.
'''
scale = StandardScaler()
combined_dummies_scaled = combined_dummies.copy()
combined_dummies_scaled[['Age','SibSp','Parch','norm_fare']]= scale.fit_transform(combined_dummies_scaled[['Age','SibSp','Parch','norm_fare']])
print(combined_dummies_scaled)

train_x_scaled = combined_dummies_scaled[combined_dummies_scaled.training == 1].drop(['training'], axis =1)
test_x_scaled = combined_dummies_scaled[combined_dummies_scaled.training == 0].drop(['training'], axis =1)

train_y = combined_data[combined_data.training==1].Survived

'''
building model - Logistic regression
    Here we are checking the cross-validation on train_x and train_y values using Logistic regression.
    The maximum iteration can be upto 2000.
    The cross_val_score() function is used to perform the evaluation, taking the datasets(here it is
    train_x and train_y) and model (logistic regression with max iteration of 2000 - log_regr) and 
    number of cross-validations( here it is 10) and returns a list of scores calculated for each fold.
    The same process is repeated for cross_val_score() function but with scaled train_x values (that is
    train_x_scaled).
'''
log_regr = LogisticRegression(max_iter = 2000)
cv = cross_val_score(log_regr,train_x,train_y,cv=10)
print("\n",cv)
print("Logistic Regression - ",cv.mean())

cv = cross_val_score(log_regr,train_x_scaled,train_y,cv=10)
print("\n",cv)
print("Logistic Regression with scaled values - ",cv.mean())

'''
building model - Random Forest
    Here we are checking the cross-validation on train_x and train_y values using Random Forest.
    random_state - it controls both the randomness of the bootstrapping of the samples used in the 
    building trees and the sampling of the features to consider when looking for the best split at 
    each node.
    The cross_val_score() function is used to perform the evaluation, taking the datasets(here it is
    train_x and train_y) and model (randon forest method with random_state 10) and 
    number of cross-validations( here it is 10) and returns a list of scores calculated for each fold.
    The same process is repeated for cross_val_score() function but with scaled train_x values (that is
    train_x_scaled).
'''
ran_for = RandomForestClassifier(random_state = 10)
cv = cross_val_score(ran_for,train_x,train_y,cv=10)
print("\n",cv)
print("\nRandom Forest - ",cv.mean())

cv = cross_val_score(ran_for,train_x_scaled,train_y,cv=10)
print("\n",cv)
print("\nRandom Forest with scaled values - ",cv.mean())

'''
building model - xgboost
    Here we are checking the cross-validation on train_x and train_y values using Xgboost method.
    random_state - it is random number seed that is used for generating reproducible results.
    use_label_encoder - it is a label encoder from scikit-learn.
    eval_metric - it is evaluation metrics for validation data. rmse is root mean square error.
    The cross_val_score() function is used to perform the evaluation, taking the datasets(here it is
    train_x and train_y) and model (randon forest method with random_state 10) and 
    number of cross-validations( here it is 10) and returns a list of scores calculated for each fold.
    The same process is repeated for cross_val_score() function but with scaled train_x values (that is
    train_x_scaled).
'''
xgboost = XGBClassifier(random_state = 10, use_label_encoder=False, eval_metric='rmse')
cv = cross_val_score(xgboost,train_x,train_y,cv=5)
print("\n",cv)
print("\nXgboost model - ",cv.mean())

cv = cross_val_score(xgboost,train_x_scaled,train_y,cv=5)
print("\n",cv)
print("\nXgboost model with scaled values - ",cv.mean())

'''
    The votingclassifier takes all inputs and averages the results. Here soft classifier is used that 
    will the averages of the confidence of each model.
'''
voting_clf = VotingClassifier(estimators = [('log_regr',log_regr), ('ran_for', ran_for), ('xgboost', xgboost)], voting = 'soft') 
cv = cross_val_score(voting_clf,train_x_scaled,train_y,cv=10)
#print("\n",cv)
print("\nVoting classifier - ",cv.mean())


voting_clf.fit(train_x_scaled,train_y)  # fit the estimators
y_hat_base_vc = voting_clf.predict(test_x_scaled).astype(int)   # Perdict class label for test_x_scaled
basic_submission = {'PassengerId': test_data.PassengerId, 'Survived': y_hat_base_vc}    # assigning the predicted values to test dataset passengers.
base_submission = pd.DataFrame(data=basic_submission)   #  creating a data frame with predicted values.
base_submission.to_csv('base_submission.csv', index=False)  # writing the dataframe to csv file.

'''
model turing - logistic regression
    Here model turing is done to improve the model prediction rate. For the logistic regression model,
    adding few parameters,
    max_iter - maximum iteration upto 2000
    penalty - to imposes  a penalty for having too many variables. penalty are l1 and l2.
    solver - algorithm used for optimization problem. this depends on the penalty chosen.
    The GridSearchCV() function is an exhaustic search done over above mentioned paramaters for the 
    logistic regression estimator.
    Then using the fit() for training the model.
    Then using the predict() function based on best score of the estimator taken for fit() for the
    prediction.
    Then converting the predicted values into a data frame and storing the values in csv file.
'''
param_grid = {'max_iter' : [2000],
              'penalty' : ['l1', 'l2'],
              'C' : np.logspace(-4, 4, 20),
              'solver' : ['liblinear']}
clf_lr = GridSearchCV(log_regr, param_grid = param_grid, cv = 100, verbose = True, n_jobs = -1)
best_clf_lr = clf_lr.fit(train_x_scaled,train_y)
print("\n Logistic Regression with tuned model - ",str(best_clf_lr.best_score_))

lr_predict = best_clf_lr.best_estimator_.predict(test_x_scaled).astype(int)
lr_data = {'PassengerId': test_data.PassengerId, 'Survived': lr_predict}
lr_prediction = pd.DataFrame(data=lr_data)
lr_prediction.to_csv('Logistic_Regression_prediction.csv', index=False)

'''
model turning - Random Forest
    Here model tuning is done on random forest to improve the prediction rate. For the random forest
    model, following parameter are used,
    n_estimators - number of treesto be created in forest.
    criterion - used to measure the quality of a split. Gini is gini impurity and entropy is to gain 
    information.
    bootstrap - used for building trees.
    max_depth - maximum depth of the tree.
    max_features - the number of features to consider when looking for the best split.
        auto is max_features = sprt(n_features)
        sprt is same auto
        int is to consider max_features features at each split.
    min_samples_leaf - minimum number of samples required to be in leaf node.
    min_samples_split - minimum number of samples required to split an internal node.
    The GridSearchCV() function is an exhaustic search done over above mentioned paramaters for the 
    random forest estimator.
    Then using the fit() for training the model.
    Then using the predict() function based on best score of the estimator taken for fit() for the
    prediction.
    Then converting the predicted values into a data frame and storing the values in csv file.
'''
param_grid =  {'n_estimators': [400,450,500,550],
               'criterion':['gini','entropy'],
               'bootstrap': [True],
               'max_depth': [20, 25, 30],
               'max_features': ['auto','sqrt', 10],
               'min_samples_leaf': [2,3],
               'min_samples_split': [2,3]}
                                  
clf_rf = GridSearchCV(ran_for, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_rf = clf_rf.fit(train_x_scaled,train_y)
print("\n Random forest with tuned model - ",str(best_clf_rf.best_score_))

rf_predict = best_clf_rf.best_estimator_.predict(test_x_scaled).astype(int)
rf_data = {'PassengerId': test_data.PassengerId, 'Survived': rf_predict}
rf_prediction = pd.DataFrame(data=rf_data)
rf_prediction.to_csv('Random_Forest_prediction.csv', index=False)

'''
model turing - xgboost
    Here model tuning is done on random forest to improve the prediction rate. For the xgboost model, 
    following parameter are used,
    n_estimators - number of gradient boosted tree, equivalent to number of boosting rounds.
    colsample_bytree - subsample ratio of columns when constructing each tree.
    max_depth - maximum tree depth for the learner.
    reg_alpha - L1 regularization term on weights.
    reg_lambda - L2 regularization term on weights.
    subsample - subsample ratio of the training instance.
    learning_rate - boosting learning rate.
    gamma - mimimum loss reduction required to make a further partition on a leaf node.
    min_child_weight - minimum sum of instance weight needed.
    sampling_method - method used to sample the training instances. Uniform is used for training each
        instancethat has equal probabilty of being selected.
    The GridSearchCV() function is an exhaustic search done over above mentioned paramaters for the 
    xgboost estimator.
    Then using the fit() for training the model.
    Then using the predict() function based on best score of the estimator taken for fit() for the
    prediction.
    Then converting the predicted values into a data frame and storing the values in csv file.
'''
param_grid = {
    'n_estimators': [450,500,550],
    'colsample_bytree': [0.75,0.8,0.85],
    'max_depth': [None],
    'reg_alpha': [1],
    'reg_lambda': [2, 5, 10],
    'subsample': [0.55, 0.6, .65],
    'learning_rate':[0.5],
    'gamma':[.5,1,2],
    'min_child_weight':[0.01],
    'sampling_method': ['uniform']
}

clf_xgb = GridSearchCV(xgboost, param_grid = param_grid, cv = 10, verbose = True, n_jobs = -1)
best_clf_xgb = clf_xgb.fit(train_x_scaled,train_y)
print("\n Xgboost model with tuned model - ",str(best_clf_lr.best_score_))

xgb_predict = best_clf_xgb.best_estimator_.predict(test_x_scaled).astype(int)
xgb_data = {'PassengerId': test_data.PassengerId, 'Survived': xgb_predict}
xgb_prediction = pd.DataFrame(data=xgb_data)
xgb_prediction.to_csv('xgb_prediction.csv', index=False)