import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import lightgbm as lgb
import statsmodels.api as sm
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#read data overall
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


df = pd.concat([train, test], axis=0)


data = df.drop(columns=['Unnamed: 0', 'id'], errors='ignore')


data.dropna(subset=['Arrival Delay in Minutes'], inplace=True)


 # Encode categorical variables
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
    
    
# Encode the target variable
le_target = LabelEncoder()
data['satisfaction'] = le_target.fit_transform(data['satisfaction'])  # 1: Satisfied, 0: Neutral or Dissatisfied



# Separate features and target variable
X = data.drop(columns=['satisfaction'])
y = data['satisfaction']

# Standardize numerical features
numerical_columns = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
scaler = StandardScaler()
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)




# Train a logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
model.fit(X_train, y_train)



# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]



y_testdata = pd.DataFrame({
    'predicted_probabilities': y_pred_proba,
    'actual_values': y_test
})


class_0_probs = y_testdata[y_testdata['actual_values'] == 0]['predicted_probabilities']
class_1_probs = y_testdata[y_testdata['actual_values'] == 1]['predicted_probabilities']

# Plot density plots for each class
plt.figure(figsize=(10, 6))
sns.kdeplot(class_0_probs.values, color='blue', fill=True, label='Class 0 (Negative)')
sns.kdeplot(class_1_probs.values, color='orange', fill=True, label='Class 1 (Positive)')
plt.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, label='Decision Threshold (0.5)')

# Add plot details
plt.title('Density Plot of Predicted Probabilities', fontsize=14)
plt.xlabel('Predicted Probability', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()



plt.figure(figsize=(10, 6))
sns.histplot(y_pred_proba, kde=True, bins=30, color='skyblue', stat='density')
plt.title('Distribution of Predicted Probabilities from Logistic Regression')
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.show()


# Evaluate the model
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"Accuracy: {accuracy:.2f}")
print(f"ROC-AUC: {roc_auc:.2f}")



# Display simplified classification report visually
report = classification_report(y_test, y_pred, output_dict=True)
summary_metrics = {
    'Metric': ['Precision', 'Recall', 'F1-Score'],
    'Score': [
        report['weighted avg']['precision'],
        report['weighted avg']['recall'],
        report['weighted avg']['f1-score']
    ]
}
summary_df = pd.DataFrame(summary_metrics)

# Plot classification report as bar chart
plt.figure(figsize=(8, 6))
sns.barplot(x='Metric', y='Score', data=summary_df, palette='viridis')
plt.title('Classification Report Metrics')
plt.ylim(0, 1)
plt.ylabel('Score')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Plot Enhanced ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(10, 8))
sns.set_theme(style="whitegrid")
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.fill_between(fpr, tpr, color='blue', alpha=0.1)
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Null Model')
plt.title('ROC Curve', fontsize=16)
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(visible=True, which='both', linestyle='--', linewidth=0.7, alpha=0.8)
plt.show()




cm = confusion_matrix(y_test, y_pred)

# Normalize the confusion matrix to show percentages
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))

# Set a refined style
sns.set(style="whitegrid", palette="pastel")  

# Create the heatmap with improved aesthetics
ax = sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap="coolwarm", 
                 cbar_kws={'label': 'Percentage'},
                 xticklabels=['Satisfied', 'Dissatisfied/Neutral'], 
                 yticklabels=['Satisfied', 'Dissatisfied/Neutral'],
                 linewidths=2, linecolor='white', 
                 annot_kws={"size": 14, "weight": 'bold', "color": 'white'})

# Adding a subtle grid and adjusting for better readability
plt.title('Confusion Matrix', fontsize=18, weight='bold', color='black')
plt.xlabel('Predicted Satisfaction', fontsize=15, weight='bold', color='black')
plt.ylabel('True Satisfaction', fontsize=15, weight='bold', color='black')

# Adjust tick parameters for better legibility
plt.xticks(fontsize=13, rotation=0)
plt.yticks(fontsize=13, rotation=0)
plt.tight_layout()
plt.show()


X = sm.add_constant(X)
model = sm.Logit(y, X)
result = model.fit()

# Print Summary
print(result.summary())



# Train a logistic regression model
model = LogisticRegression(random_state=42)

cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

# Bar plot for cross-validation scores
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(cv_scores) + 1), cv_scores, color='orchid')
plt.xlabel('Fold Number')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Accuracy Scores')
plt.show()



# Separate features and target variable
X = data.drop(columns=['satisfaction'])
y = data['satisfaction']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)



# Evaluate the model
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
cm = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")



# Get feature importance
feature_importance = model.feature_importances_
feature_names = X_train.columns

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='lightcoral')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.title('Feature Importance from LightGBM')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()



cm = confusion_matrix(y_test, y_pred)

# Normalize the confusion matrix to show percentages
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))

# Set a refined style
sns.set(style="whitegrid", palette="pastel")  

# Create the heatmap with improved aesthetics
ax = sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap="coolwarm", 
                 cbar_kws={'label': 'Percentage'},
                 xticklabels=['Satisfied', 'Dissatisfied/Neutral'], 
                 yticklabels=['Satisfied', 'Dissatisfied/Neutral'],
                 linewidths=2, linecolor='white', 
                 annot_kws={"size": 14, "weight": 'bold', "color": 'white'})

# Adding a subtle grid and adjusting for better readability
plt.title('Confusion Matrix', fontsize=18, weight='bold', color='black')
plt.xlabel('Predicted Satisfaction', fontsize=15, weight='bold', color='black')
plt.ylabel('True Satisfaction', fontsize=15, weight='bold', color='black')

# Adjust tick parameters for better legibility
plt.xticks(fontsize=13, rotation=0)
plt.yticks(fontsize=13, rotation=0)
plt.tight_layout()
plt.show()
