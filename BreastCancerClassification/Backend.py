import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC  # Import SVC

# 1. Load dataset
df = pd.read_csv(r'C:\Users\91939\Desktop\AI&DS\GITHUB REPOSITORIES\BreastCancerClassification\brest cancer.txt', header=None)

# 2. Rename columns
df.columns = ['Id', 'Clump_thickness', 'Uniformity_Cell_Size', 'Uniformity_Cell_Shape', 'Marginal_Adhesion', 
              'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']

# 3. Drop irrelevant column
df.drop('Id', axis=1, inplace=True)

# 4. Convert Bare_Nuclei to numeric
df['Bare_Nuclei'] = pd.to_numeric(df['Bare_Nuclei'], errors='coerce')

# 5. Handle missing values using median strategy
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
df['Bare_Nuclei'] = imputer.fit_transform(df['Bare_Nuclei'].values.reshape(-1, 1))

# 6. Separate features and target variable
X = df.drop('Class', axis=1)
y = df['Class']

# 7. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 8. Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for future use
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 9. Define models including SVC
models = {
    'KNN': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVC': SVC()  # Add Support Vector Classifier (SVC)
}

# 10. Train, evaluate, and save each model
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Print performance metrics
    print(f"--- {model_name} ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Model accuracy score: {accuracy_score(y_test, y_pred)}")
    
    bias = model.score(X_train, y_train)
    variance = model.score(X_test, y_test)
    print(f"Bias: {bias}")
    print(f"Variance: {variance}")
    
    # Classification report
    print(classification_report(y_test, y_pred))
    
    # Save the model using pickle
    filename = f'{model_name.replace(" ", "_").lower()}_breastcancer.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model has been saved as {filename}\n")

