import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from kerastuner.tuners import RandomSearch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load transaction data
data = pd.read_csv("synthetic_data.csv")

# Feature Engineering
data['TimeDifference'] = data['TransactionTime'] - data['MiningTime']

# Separate features and target
X = data[['TransactionTime', 'MiningTime', 'TimeDifference']]
y = data['Fraudulent']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a more complex neural network architecture
def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units_1', min_value=32, max_value=512, step=32),
                    activation='relu', input_dim=X_train_scaled.shape[1]))
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(units=hp.Int('units_2', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Perform hyperparameter tuning using Keras Tuner
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='keras_tuner',
    project_name='fraud_detection'
)

tuner.search(X_train_scaled, y_train, epochs=10, validation_split=0.2)

# Get the best hyperparameters
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the final model with the best hyperparameters
final_model = tuner.hypermodel.build(best_hp)
history = final_model.fit(X_train_scaled, y_train, epochs=10, validation_split=0.2, verbose=0)  # Set verbose to 0

# Predictions
y_pred_proba = final_model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype(int)  # Apply threshold for classification

# Create a DataFrame with original TransactionTime, MiningTime, and predicted labels
predictions_df = pd.DataFrame({
    'TransactionTime': X_test['TransactionTime'],
    'MiningTime': X_test['MiningTime'],
    'PredictedFraudulent': ['fraudulent' if pred == 1 else 'not fraudulent' for pred in y_pred]
})

# Display the DataFrame
print(predictions_df)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print F1 score and accuracy
print("F1 Score:", f1)
print("Accuracy:", accuracy)

# Plot confusion matrix
labels = ['Not Fraudulent', 'Fraudulent']
cm = pd.DataFrame(conf_matrix, index=labels, columns=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot decision boundary
plt.figure(figsize=(10, 6))
sns.scatterplot(x='TransactionTime', y='MiningTime', hue='PredictedFraudulent', data=predictions_df, palette='coolwarm')
plt.xlabel('Transaction Time')
plt.ylabel('Mining Time')
plt.title('Decision Boundary')
plt.legend(title='Predicted Fraudulent', loc='upper right')
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# 
