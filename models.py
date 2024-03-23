import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, accuracy_score

# def get_mining_technique(mineral_type):
#     # correct file paths
#     file_path = "dataset/mining_technique.csv"
#     df = pd.read_csv(file_path)

#     # Perform one-hot encoding for categorical variables
#     df = pd.get_dummies(df, columns=['Mineral Type', 'Mineral Hardness', 'Temperature', 'Humidity', 'Equipment Type', 'Cutting speed', 'feed rate', 'Wear and Tear Rate'])
#     # Separate features and target variable
#     X = df.drop('Mining Technique', axis=1)
#     y = df['Mining Technique']

#     # Split the dataset into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     model = DecisionTreeClassifier()

#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)

#     # Evaluate the accuracy
#     accuracy = accuracy_score(y_test, y_pred)

#     mineral_type = "Coal"  # Change this to the mineral type you want to predict for
#     input_data = df[df[f'Mineral Type_{mineral_type}'] == 1].drop('Mining Technique', axis=1) 
#     prediction = model.predict(input_data)
#     # print(f"Predicted Wear and Tear Rate for {mineral_type}: {prediction[0]}")

#     return prediction[0]


# def get_mineral_concentration(state, mineral_type):
#     # dummy code

#     return

def check(state_name):
    return state_name

def predict_resource_and_mining(state,data):
  # Load the dataset
#   data = pd.read_csv("{{url_for('data',filename='Dataset1.csv')}}")

  # Drop rows with null values
  data.dropna(inplace=True)

  # Separate features and target variable
  X = data.drop(columns=['Resource Concentration (g/t)', 'Mining Method'])
  y = data[['Resource Concentration (g/t)', 'Mining Method']]

  # Define categorical features
  categorical_features = ['States', 'Rock Type', 'Mineral', 'Extraction Equipment', 'Mineral Hardness', 'Humidity', 'Cutting Speed', 'Wear and tear rate' ]

  # Define preprocessing steps for categorical features
  categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
  ])

  # Define preprocessing steps for numerical features
  numeric_features = ['Latitude', 'Longitude', 'Elevation (m)', 'Drilling Depth (m)', 'Temperature (°C)', 'Past Extraction Results (tons)', 'Power used (HP)']
  numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
  ])

  # Combine preprocessing steps for categorical and numerical features
  preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

  # One-hot encode the target variable 'Mining Method'
  encoder = OneHotEncoder()
  y_encoded = encoder.fit_transform(y[['Mining Method']]).toarray()  # Convert to dense array

  # Split the data into training and testing sets
  X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

  # Define the model
  model = RandomForestRegressor(n_estimators=100, random_state=42)

  # Train the model
  model.fit(preprocessor.fit_transform(X_train), y_train_encoded)

  # Reverse mapping for mining techniques
  reverse_mapping = {idx: technique for idx, technique in enumerate(encoder.categories_[0])}

  # Get data corresponding to the provided state
  state_data = data[data['States'] == state]

  # Predict resource concentration and mining technique for each mineral
  predictions = {}
  for mineral in state_data['Mineral']:
    mineral_data = state_data[state_data['Mineral'] == mineral]
    X_mineral = mineral_data.drop(columns=['Resource Concentration (g/t)', 'Mining Method'])
    prediction = model.predict(preprocessor.transform(X_mineral))
    # Extract resource concentration and mining method from prediction
    resource_concentration = prediction[0][0]
    predicted_mining_method_idx = int(prediction[0][1])  # Convert to integer
    if predicted_mining_method_idx in reverse_mapping:
        predicted_mining_method = reverse_mapping[predicted_mining_method_idx]
    else:
        predicted_mining_method = 'Unknown'
    predictions[mineral] = [resource_concentration, predicted_mining_method]

  output_df = pd.DataFrame(predictions).T
  output_df.columns = ['Predicted Resource Concentration (g/t)', 'Predicted Mining Method']
  return output_df


# output = predict_resource_and_mining(state)
# print(output)