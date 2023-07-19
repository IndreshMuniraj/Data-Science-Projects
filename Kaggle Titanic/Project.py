import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# loading the train and test dataset
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
print(train_data.info())
#print(test_data.info())

# to get total number of missing or null data from each column.
print(train_data.isna().sum())  
#print("\n")
#print(test_data.isna().sum())

# checking the survival rate on Pclass
train_pclass = train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(train_pclass)
plt.hist(train_data["Pclass"])
#plt.hist(train_pclass)
plt.title("Plcass")
plt.show()

# checking on age
train_age = train_data[['Age', 'Survived']].groupby(['Age'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(train_age)
plt.hist(train_age)
plt.title("Age")
plt.show()

train_age_survival = pd.pivot_table(train_data, index= "Survived", values = "Age")
print(train_age_survival)

train_data["Age"] = train_data["Age"].fillna(train_data.Age.median())
test_data["Age"] = test_data["Age"].fillna(test_data.Age.median())

print(train_data["Age"].isna().sum(), test_data["Age"].isna().sum())

# checking names and titles
train_data['name'] = train_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
print(train_data['name'].value_counts())

train_name = train_data[['name', 'Survived']].groupby(['name'], as_index=False).mean().sort_values(by='Survived')
print(train_name)

sns.barplot(train_data['name'].value_counts().index, train_data['name'].value_counts())
plt.show()

train_data = train_data.assign(fname = train_data.Name.str.split(",").str[0])
train_data["title"] = pd.Series([i.split(",")[1].split(".")[0].strip() for i in train_data.Name], index=train_data.index)
#print(train_data["title"])

test_data = test_data.assign(fname = test_data.Name.str.split(",").str[0])
test_data["title"] = pd.Series([i.split(",")[1].split(".")[0].strip() for i in test_data.Name], index=test_data.index)
#print(test_data["title"])

train_data.drop("Name", axis=1, inplace=True)
test_data.drop("Name", axis=1, inplace=True)

print(train_data.fname.nunique())  # returns number of unique names
print(train_data.title.nunique())   # returns number of unique titles
print(test_data.fname.nunique())    # returns number of unique names
print(test_data.title.nunique())    # returns number of unique titles

sns.countplot(x= "title", data= train_data)
plt.show()

print(train_data["title"].unique())
print(test_data["title"].unique())

group_title = ["Mr", "Mrs", "Miss", "Master", "Ms"]
other_title = []
for i in train_data["title"].unique():
    if i not in group_title:
        other_title.append(i)
for i in test_data["title"].unique():
    if i not in group_title:
        other_title.append(i)
#print(other_title)

# replacing the title with numeric values
train_data["title"] = train_data["title"].replace(other_title, "Other")
train_data["title"] = train_data["title"].map({"Mr":0, "Miss":0, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Master":2, "Other":3})
test_data["title"] = test_data["title"].replace(other_title, "Other")
test_data["title"] = test_data["title"].map({"Mr":0, "Miss":0, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Master":2, "Other":3})

print(train_data.title.isna().sum())    # checking for null values
print(test_data.title.isna().sum())     # checking for null values

# checking on sex
train_sex = train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print("\n", train_sex)
plt.hist(train_data["Sex"])
plt.title("Sex")
plt.show()
sex_pclass = train_data.assign(sex_class = train_data['Sex'] + "_" + train_data['Pclass'].astype('str'))
train_sex_pclass = sex_pclass[['sex_class', 'Survived']].groupby(['sex_class'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print("\n",train_sex_pclass)

'''
train_data = train_data.assign(sex_class = train_data['Sex'] + "_" + train_data['Pclass'].astype("str"))
test_data = test_data.assign(sex_class = test_data['Sex'] + "_" + test_data['Pclass'].astype("str"))

# replacing the title with numeric values
train_data["Sex"] = train_data["Sex"].map({"female":0, "male":1})
test_data["Sex"] = test_data["Sex"].map({"female":0, "male":1})

train_data["sex_class"] = train_data["sex_class"].map({"female_1":0, "female_2":1, "female_3":2, "male_1":4, "male_2":5, "male_3":6})
test_data["sex_class"] = test_data["sex_class"].map({"female_1":0, "female_2":1, "female_3":2, "male_1":4, "male_2":5, "male_3":6})
'''

#checking on siblings and spouse
train_siblings = train_data[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print("\n",train_siblings)
plt.hist(train_data["SibSp"])
plt.title("Siblings & Spouse")
plt.show()

#checking on Parents and childers
train_parch = train_data[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print("\n",train_parch )
plt.hist(train_data["Parch"])
plt.title("Parent & Children")
plt.show()

# combining the SilSp and Parch
sibsp_parch = train_data.assign(family_size = train_data['SibSp'] + train_data['Parch'])
train_sibsp_parch = sibsp_parch[['family_size', 'Survived']].groupby(['family_size'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print("\n", train_sibsp_parch)

sns.barplot(train_data['SibSp'].value_counts().index, train_data['SibSp'].value_counts())
plt.show()

sns.barplot(train_data['Parch'].value_counts().index, train_data['Parch'].value_counts())
plt.show()

# checking ticket data
train_data['numeric_ticket'] = train_data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
print(train_data['numeric_ticket'].value_counts())

sns.barplot(train_data['numeric_ticket'].value_counts().index, train_data['numeric_ticket'].value_counts())
plt.show()

train_numtt = train_data[['numeric_ticket', 'Survived']].groupby(['numeric_ticket'], as_index = False).mean().sort_values(by='Survived')
print(train_numtt)

train_data['ticket_letters'] = train_data.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)
print(train_data['ticket_letters'].value_counts())

sns.barplot(train_data['ticket_letters'].value_counts().index, train_data['ticket_letters'].value_counts())
plt.show()


train_data["ticket_prefix"] = pd.Series([len(i.split()) > 1 for i in train_data.Ticket], index=train_data.index)
ticket = train_data[['ticket_prefix', 'Survived']].groupby(['ticket_prefix'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print("\n",ticket)

sns.barplot(train_data['ticket_prefix'].value_counts().index, train_data['ticket_prefix'].value_counts())
plt.show()

train_data["ticket_prefix"] = pd.Series([len(i.split()) > 1 for i in train_data.Ticket], index=train_data.index)
test_data["ticket_prefix"] = pd.Series([len(i.split()) > 1 for i in test_data.Ticket], index=test_data.index)

train_data.drop("ticket_prefix", axis=1, inplace=True)
test_data.drop("ticket_prefix", axis=1, inplace=True)
train_data.drop("Ticket", axis=1, inplace=True)
test_data.drop("Ticket", axis=1, inplace=True)


#checking on embarked
train_embarked = train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(train_embarked)

sns.barplot(train_data['Embarked'].value_counts().index, train_data['Embarked'].value_counts())
plt.show()

train_data["Embarked"] = train_data["Embarked"].fillna("S")
test_data["Embarked"] = test_data["Embarked"].fillna("S")
print(train_data["Embarked"].isna().sum(), test_data["Embarked"].isna().sum())

#checking fare
plt.hist(train_data["Fare"])
plt.show()
train_data["Fare"] = train_data["Fare"].fillna(train_data.Fare.median())
test_data["Fare"] = test_data["Fare"].fillna(test_data.Fare.median())

#checking cabin
train_data.drop("Cabin", axis=1, inplace=True)
test_data.drop("Cabin", axis=1, inplace=True)

# catrgorzing the dataset into numerical and catergory type.
#train_num = train_data[['Age','SibSp','Parch','Fare']]      # numerical type
#train_cat = train_data[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']]     # category type

print(train_data.isna().sum())
print(test_data.isna().sum())

train_dummies = pd.get_dummies(train_data)
test_dummies = pd.get_dummies(test_data)

scale = StandardScaler()
train_dummies_scaled = train_dummies.copy()
test_dummies_scaled = test_dummies.copy()

log_reg = LogisticRegression(max_iter = 1000)
cross_valve = cross_val_score(log_reg,train_dummies,test_dummies,cv=5)
print(cross_valve)
print(cross_valve.mean())


