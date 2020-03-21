from sklearn.linear_model import LinearRegression
import pandas as pd

read_csv = pd.read_csv('example_crt_prediction.csv')
read_csv['CTR'] = read_csv['CTR'].str.replace('%', '').str.replace(',', '.').astype(float)
read_csv['Position'] = read_csv['Position'].str.replace(',', '.').astype(float)
read_csv['CTR'] = pd.to_numeric(read_csv['CTR'])
read_csv['Position'] = pd.to_numeric(read_csv['Position'])
read_csv.dropna(inplace=True)
read_csv = read_csv.round(0)
#print(read_csv.head())
print(read_csv.corr()['CTR'])
features = ['Position', 'Impressions']
target = 'CTR'
train = read_csv.sample(frac=.8)
test = read_csv.loc[~read_csv.index.isin(train.index)]
linearregressionmodel = LinearRegression()
linearregressionmodel.fit(train[features], train[target])


# Test how the model performes against the Training data we split above...
prediction_score = linearregressionmodel.score(test[features], test[target])
print("The score of prediction for LinearRegressionModel is: {}".format(prediction_score))



# Print Predictions for all created Models

# Define parameters for Predictions
# (in this case: what CTR we have vor a Keyword on position 2 with 200 impressions)
position = 1.0
impressions = 20
data = [[position, impressions]]  # needs to be same count as features

df_to_predict = pd.DataFrame(data = data, index=[0], columns=features)
res = linearregressionmodel.predict(df_to_predict)
print("LinearRegressionModel predicted:       {}% CTR".format(int(res[0])))

