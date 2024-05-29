X_raw = df[features]
y_raw = df['Survival']
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.21348, shuffle=True, random_state=0)
# To apply the different imputers, we first have to split our data into seperate numerical and categorical data
X_train_num = X_train_raw[numFeatures]
X_train_cat = X_train_raw[categFeatures]
X_test_num = X_test_raw[numFeatures]
X_test_cat = X_test_raw[categFeatures]

numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')
scaler = MinMaxScaler()

numeric_imputer.fit(X_train_num)
X_train_num_imp = numeric_imputer.transform(X_train_num)
X_test_num_imp = numeric_imputer.transform(X_test_num)
    
# Fit on the numeric training data
scaler.fit(X_train_num_imp)

# Transform the training and test data
X_train_num_sca = scaler.transform(X_train_num_imp)
X_test_num_sca = scaler.transform(X_test_num_imp)

categorical_imputer.fit(X_train_cat)
X_train_cat_imp = categorical_imputer.transform(X_train_cat)
X_test_cat_imp = categorical_imputer.transform(X_test_cat)
# create the encoder object
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
# Fit encoder on teh training data
encoder.fit(X_train_cat_imp)
# Transform the test and train data
X_train_onehot = encoder.transform(X_train_cat_imp)
X_test_onehot = encoder.transform(X_test_cat_imp)

X_train = np.concatenate([X_train_num_sca, X_train_onehot], axis=1)
X_test = np.concatenate([X_test_num_sca, X_test_onehot], axis=1)

model = SVC() # Create our prediction model object
model.fit(X_train, y_train) # fit our logistic regression model - 'train' the model

# We can make a prediction with the training data
y_pred_train = model.predict(X_train)
y_pred = model.predict(X_test)

# The accuracy score: 1 for perfect prediction
print('Accuracy: {:.4f}'.format(accuracy_score(y_test, y_pred)))
print('Precision: {:.4f}'.format(precision_score(y_test, y_pred)))
print('Recall: {:.4f}'.format(recall_score(y_test, y_pred)))
print('F1 Score: {:.4f}'.format(f1_score(y_test, y_pred)))
print('Balanced Accuracy: {:.4f}'.format(balanced_accuracy_score(y_test, y_pred)))
print('Macro averaged F1 Score: {:.4f}'.format(f1_score(y_test, y_pred, average='macro')))
print('Weighted averaged F1 Score: {:.4f}'.format(f1_score(y_test, y_pred, average='weighted')))
# Confusion matrix
confusion_mat = sklearn.metrics.confusion_matrix(y_test, y_pred, normalize='all')
print(f'Confusion matrix: \n', confusion_mat)
# Visualize the confusion matrix
sklearn.metrics.ConfusionMatrixDisplay(confusion_mat, display_labels=['Non survival', 'Survival']).plot(cmap=plt.cm.Blues)
plt.grid(False)
# The classification report, which contains accuracy, precision, recall, F1 score
# Note, the Precision/Recall/F1 in the report match the positive class (1.0) in the report
print(sklearn.metrics.classification_report(y_test, y_pred))
