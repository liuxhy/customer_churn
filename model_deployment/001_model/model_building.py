import pickle
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.preprocessing import StandardScaler
from utils import Transformer

# Load data
df = pd.read_csv("telco.csv")

# Label encoding 'Churn'
le = LabelEncoder()
df['Churn'] = le.fit_transform(df['Churn'])

# Split the data
splitter = GroupShuffleSplit(test_size=.20, n_splits=2, random_state=7)
split = splitter.split(df, groups=df['customerID'])
train_inds, test_inds = next(split)

train = df.iloc[train_inds]
test = df.iloc[test_inds]

# Define features and target
features = list(df.columns)
target = 'Churn'
features.remove(target)
features.remove('customerID')  # Assuming 'customerID' should not be used as a feature

X = train[features]
y = train[target]
X_test = test[features]
y_test = test[target]

# Define SMOTE
smote = SMOTE(random_state=42)

# Define CatBoost parameters
params = {'verbose': False,
          'iterations': 200,
          'learning_rate': 0.01,
          'depth': 7}

# Define pipeline steps
steps = [('tf', Transformer()),
         ('smote', smote),
         ('rescale', StandardScaler()),
         ('cat', CatBoostClassifier(**params))]

# Create and fit pipeline
model = imbpipeline(steps)

try:
    model = model.fit(X, y)
    print("Model training completed successfully.")
except Exception as e:
    print("Error during model training:", e)

# Save the model
try:
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully.")
except Exception as e:
    print("Error saving the model:", e)