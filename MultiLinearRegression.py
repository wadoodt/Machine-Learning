features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity']
X_raw = df[['median_income', 'ocean_proximity']]
y_raw = df['median_house_value']
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, shuffle=True, random_state=0)
# To apply the different imputers, we first have to split our data into seperate numerical and categorical data
X_train_num = X_train_raw.select_dtypes(include=np.number)
X_train_cat = X_train_raw.select_dtypes(exclude=np.number)
X_test_num = X_test_raw.select_dtypes(include=np.number)
X_test_cat = X_test_raw.select_dtypes(exclude=np.number)

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

# Create linear regression object
obj = sklearn.linear_model.LinearRegression()

# Train the model using the training sets
obj.fit(X_train, y_train)

# We can make a prediction with the training data
y_pred_train = obj.predict(X_train)
y_pred = obj.predict(X_test)

print(sklearn.metrics.r2_score(y_test, y_pred))
print((sklearn.metrics.mean_squared_error(y_test, y_pred))**0.5)
print(sklearn.metrics.r2_score(y_train, y_pred_train))
print((sklearn.metrics.mean_squared_error(y_train, y_pred_train))**0.5)
