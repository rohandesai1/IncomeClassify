import pandas
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

BASE_PATH = input("provide base path: ")


census = pandas.read_csv(BASE_PATH + "final_cleaned_census.csv")

census.drop("Unnamed: 0", axis = 1, inplace = True) 



tr_features = pandas.read_csv(BASE_PATH + 'train_features.csv')
tr_labels = pandas.read_csv(BASE_PATH + 'train_labels.csv')

val_features = pandas.read_csv(BASE_PATH + 'validation_features.csv')
val_labels = pandas.read_csv(BASE_PATH + 'validation_labels.csv')

te_features = pandas.read_csv(BASE_PATH + 'test_features.csv')
te_labels = pandas.read_csv(BASE_PATH + 'test_labels.csv')[:3256]

for dataset in [tr_features, val_features, te_features]:
    dataset.drop("Unnamed: 0", axis = 1, inplace = True)


def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))


rf = RandomForestClassifier()

test_params = {
    'n_estimators': [5, 10, 100, 200],
    'max_depth': [2, 5, 10, 20, None]
}


#cv = GridSearchCV(rf, test_params, cv = 5)
#cv.fit(tr_features, tr_labels.values.ravel())

#print_results(cv)



rf1 = RandomForestClassifier(max_depth = 10, n_estimators = 100)
rf1.fit(tr_features, tr_labels.values.ravel())

rf2 = RandomForestClassifier(max_depth = 5, n_estimators = 200)
rf2.fit(tr_features, tr_labels.values.ravel())

rf3 = RandomForestClassifier(max_depth = 5, n_estimators = 100)
rf3.fit(tr_features, tr_labels.values.ravel())


for md in [rf1, rf2, rf3]:
    prediction = md.predict(val_features)[:3256]
    accuracy = round(accuracy_score(val_labels, prediction), 3)
    precision = round(precision_score(val_labels, prediction), 3)
    recall = round(recall_score(val_labels, prediction), 3)
    print('MAX DEPTH: {} / # OF EST: {} -- A: {} / P: {} / R: {}'.format(md.max_depth, md.n_estimators,accuracy,precision,  recall))


y_pred = rf3.predict(te_features)
accuracy = round(accuracy_score(te_labels, y_pred), 3)
precision = round(precision_score(te_labels, y_pred), 3)
recall = round(recall_score(te_labels, y_pred), 3)
print('MAX DEPTH: {} / # OF EST: {} -- A: {} / P: {} / R: {}'.format(rf2.max_depth,rf2.n_estimators,accuracy,precision,recall))


input_categories = tr_features.columns
print(input_categories)
person_data = {}

for category in input_categories:
    current_data = int(input("{}: ".format(category)))
    person_data[category] = current_data

person_data = pandas.DataFrame(person_data, index = [0])

prediction = rf1.predict(person_data)

print(prediction)