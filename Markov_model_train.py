import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from hmmlearn import hmm
import joblib
 
# Load the data from the CSV file into a pandas DataFrame
df = pd.read_csv("DatasetC.csv")
# Extract the "Value" column as the observations for the HMM
observations = df[['skill_id', 'correct', 'emotion_id']].values
# Impute missing values in the observations
imputer = SimpleImputer(strategy='constant', fill_value=0)
observations = imputer.fit_transform(observations)

# Define the number of states for the HMM
n_states = 7
 
# Initialize the HMM model
model = hmm.GaussianHMM(n_components=n_states, covariance_type="full")
 
# Fit the HMM model to the observations
model.fit(observations)
 
# Predict the hidden states for each observation
hidden_states = model.predict(observations)

# Save the model
filename = 'trained_model'
joblib.dump(model, filename)
print(hidden_states)

# Load the saved model
loaded_model = joblib.load(filename)