import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(filename='credit_risk_dataset.csv'):
    df = pd.read_csv(filename, encoding='latin-1')
    print("Shape of DataFrame:", df.shape)
    print(df.isnull().sum())
    return df

def preprocess_data(df):
    # Convert categorical variables to numerical
    df['person_home_ownership'] = df['person_home_ownership'].map({'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3})
    df['loan_intent'] = df['loan_intent'].map({'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2, 'VENTURE': 3, 'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5})
    df['loan_grade'] = df['loan_grade'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y': 1, 'N': 0})
    
    # Handle any remaining missing values
    df = df.fillna(df.mode().iloc[0])
    
    # Scale the features
    scaler = StandardScaler()
    X = df.drop('loan_status', axis=1)
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler for future use
    joblib.dump(scaler, 'scaler.pkl')
    
    return X_scaled, df['loan_status']

if __name__ == "__main__":
    df = load_data()
    X_scaled, y = preprocess_data(df)
    print("Preprocessing complete. Data ready for model training.")