#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb
import statsmodels.api as sm
import warnings
#%%
warnings.filterwarnings('ignore')
# %% [markdown]
# ## Data Overview
# **Gender:** Gender of the passengers (*Female*, *Male*)
#
# **Customer Type:** The customer type (Loyal customer, disloyal customer)
#
# **Age:** The actual age of the passengers
#
# **Type of Travel:** Purpose of the flight of the passengers (Personal Travel, Business Travel)
#
# **Class:** Travel class in the plane of the passengers (Business, Eco, Eco Plus)
#
# **Flight distance:** The flight distance of this journey
#
# **Inflight wifi service:** Satisfaction level of the inflight wifi service (0:Not Applicable;1-5)
#
# **Departure/Arrival time convenient:** Satisfaction level of Departure/Arrival time convenient
#
# **Ease of Online booking:** Satisfaction level of online booking
#
# **Gate location:** Satisfaction level of Gate location
#
# **Food and drink:** Satisfaction level of Food and drink
#
# **Online boarding:** Satisfaction level of online boarding
#
# **Seat comfort:** Satisfaction level of Seat comfort
#
# **Inflight entertainment:** Satisfaction level of inflight entertainment
#
# **On-board service:** Satisfaction level of On-board service
#
# **Leg room service:** Satisfaction level of Leg room service
#
# **Baggage handling:** Satisfaction level of baggage handling
#
# **Check-in service:** Satisfaction level of Check-in service
#
# **Inflight service:** Satisfaction level of inflight service
#
# **Cleanliness:** Satisfaction level of Cleanliness
#
# **Departure Delay in Minutes:** Minutes delayed when departure
#
# **Arrival Delay in Minutes:** Minutes delayed when Arrival
#
# **Satisfaction:** Airline satisfaction level(Satisfaction, neutral or dissatisfaction)
# %%
data1 = pd.read_csv('train.csv', index_col=0)
data2 = pd.read_csv('test.csv', index_col=0)
data = pd.concat([data1, data2])
data.head()
#%%
data.shape
# %%
data.info()
# %%
data.describe()
# %%
for column in data.columns:
  print(f'{column}: {(data1[column] == 0).sum()}')
# %%
columns_to_lcean = ['Inflight wifi service', 'Departure/Arrival time convenient',
                    'Ease of Online booking', 'Gate location', 'Food and drink',
                    'Online boarding', 'Seat comfort', 'Inflight entertainment',
                    'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness']

for column in columns_to_lcean:
  data[column] = data[column].replace(0, np.nan)
data.dropna(inplace=True)
# %%
data.shape
# %%
data.describe()
# %% [markdown]
# # EDA
# %%
data['satisfaction_group'] = data['satisfaction'].apply(
    lambda x: 'Satisfied' if x == 'satisfied' else 'Neutral or Dissatisfied'
)

# Aggregating counts for satisfaction group and gender
grouped_data = (
    data.groupby(['satisfaction_group', 'Gender']).size().reset_index(name='count')
)

# Calculating overall satisfaction group proportions (for bar heights)
total_counts = grouped_data.groupby('satisfaction_group')['count'].sum().reset_index(name='total')
grouped_data = grouped_data.merge(total_counts, on='satisfaction_group')
grouped_data['overall_percentage'] = grouped_data['total'] / grouped_data['total'].sum() * 100

# Calculate gender proportions within each satisfaction group
grouped_data['gender_percentage'] = grouped_data['count'] / grouped_data['total'] * 100

# Create the bar plot
fig = px.bar(
    grouped_data,
    x='satisfaction_group',
    y='count',  # Set bar heights to overall group proportions
    color='Gender',
    text='count',
    title="Satisfaction Levels by Group with Gender Breakdown",
    labels={'satisfaction_group': 'Satisfaction Group', 'count': 'Count'},
    color_discrete_sequence=['#FF99C2', '#66B2FF']  # Soft pastel pink and blue, slightly more vibrant
)

# Update bar traces
fig.update_traces(
    texttemplate='%{text}',  # Show raw counts on bars
    textposition='inside',    # Position text inside the bars for clarity
    marker_line=dict(width=0.5, color='black')  # Add thinner borders around bars
)

# Adding annotations for percentages outside the bars
for satisfaction_group, percentage in zip(total_counts['satisfaction_group'], total_counts['total'] / total_counts['total'].sum() * 100):
    fig.add_annotation(
        x=satisfaction_group,
        y=total_counts.loc[total_counts['satisfaction_group'] == satisfaction_group, 'total'].values[0] + 2500,  # Position above the bar further up
        text=f"{percentage:.1f}%",  # Format percentage to 1 decimal place
        showarrow=False,
        font=dict(size=14, color="black"),
        align="center"  # Align text in the center above the bar
    )

# Adding annotations for gender breakdown within each satisfaction group
fig.update_traces(
    hovertemplate="<b>%{y}</b><br>Count: %{text}<br>Gender: %{customdata[0]}<br>Proportion: %{customdata[1]:.1f}%",
    customdata=['Gender', 'gender_percentage']
)

# Improving layout
fig.update_layout(
    title=dict(x=0.5, xanchor="center"),  # Center the title
    width=1000,  # Adjust the width of the plot to make it wider
    barmode='stack',
    xaxis=dict(title="Satisfaction Group", showgrid=True, linecolor='black'),  # Add x-axis line
    yaxis=dict(title="Count", linecolor='black'),  # Add y-axis line
    template="plotly_white",
    font=dict(size=14),
    title_font=dict(size=25),
    legend_title=dict(text="Gender"),
    legend=dict(bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="gray"),
    plot_bgcolor="rgba(0,0,0,0)"
)
# fig.write_html("gender_satisfaction.html")
fig.show()
# %%
satisfied = data[data['satisfaction'] == 'satisfied']
dissatisfied = data[data['satisfaction'] == 'neutral or dissatisfied']
# %%
numerical_columns = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
# %%
for column in numerical_columns:
    satisfied_data = satisfied[column]
    dissatisfied_data = dissatisfied[column]

    satisfied_mean = np.mean(satisfied_data)
    dissatisfied_mean = np.mean(dissatisfied_data)

    # Creating the figure
    fig = go.Figure()

    # Satisfied KDE with shaded area and mean line
    satisfied_kde = gaussian_kde(satisfied_data)
    x_vals = np.linspace(satisfied_data.min(), satisfied_data.max(), 500)
    satisfied_kde_line = satisfied_kde(x_vals)
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=satisfied_kde_line,
        line=dict(color='green', width=3),
        name=f'Satisfied (Mean: {satisfied_mean:.2f})',
        fill='tozeroy',
        fillcolor='rgba(0, 128, 0, 0.3)'
    ))
    fig.add_shape(
        type="line",
        x0=satisfied_mean,
        x1=satisfied_mean,
        y0=0,
        y1=satisfied_kde_line.max(),
        line=dict(dash='dash', color='green'),
        xref="x",
        yref="y"
    )

    # Dissatisfied KDE with shaded area and mean line
    dissatisfied_kde = gaussian_kde(dissatisfied_data)
    x_vals = np.linspace(dissatisfied_data.min(), dissatisfied_data.max(), 500)
    dissatisfied_kde_line = dissatisfied_kde(x_vals)
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=dissatisfied_kde_line,
        line=dict(color='red', width=3),
        name=f'Neutral/Dissatisfied (Mean: {dissatisfied_mean:.2f})',
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.3)'
    ))
    fig.add_shape(
        type="line",
        x0=dissatisfied_mean,
        x1=dissatisfied_mean,
        y0=0,
        y1=dissatisfied_kde_line.max(),
        line=dict(dash='dash', color='red'),
        xref="x",
        yref="y"
    )

    # Layout settings
    fig.update_layout(
        title=dict(text=f"Distribution of {column}", x=0.5, xanchor="center"),
        xaxis_title=column,
        yaxis_title="Density",
        xaxis=dict(title=f"{column}", linecolor='black'),  # Add x-axis line
        yaxis=dict(title="Percentage (%)", linecolor='black'),  # Add y-axis line
        font=dict(size=14),
        title_font=dict(size=25),
        width=1500,
        showlegend=True,
        template="plotly_white"
    )
    # fig.write_html(f"{column}_distribution.html")

    fig.show()
# %%
fig = px.box(
    data,
    x='Class',
    y='Age',
    color='Class',
    category_orders={'Class': ['Business', 'Eco Plus', 'Eco']},  # Custom order
    color_discrete_sequence=px.colors.qualitative.Vivid,  # A vibrant color scheme
    title='Distribution of Age Across Classes'
)

# Customizing the layout
fig.update_layout(
    title_font_size=20,
    title=dict(x=0.5, xanchor="center"),
    xaxis_title='Class',
    yaxis_title='Age',
    xaxis_title_font_size=14,
    yaxis_title_font_size=14,
    width=800,  # Narrower plot
    height=600,  # Balanced height
    xaxis_tickangle=0,  # Ensure the category labels stay horizontal
    xaxis_tickmode='linear',  # Adjust tick spacing to ensure proper box width
    margin=dict(l=40, r=40, t=40, b=40)  # Adjust margins to maximize box width
)

# Adjusting the box widths indirectly
fig.update_traces(
    width=0.5  # Manually setting the box width
)
# fig.write_html("box_plot_age.html")
# Show the plot
fig.show()
# %%
fig = px.box(
    data,
    x='Class',
    y='Flight Distance',
    color='Class',
    category_orders={'Class': ['Business', 'Eco Plus', 'Eco']},  # Custom order
    color_discrete_sequence=px.colors.qualitative.Vivid,  # A vibrant color scheme
    title='Distribution of Flight Distance Across Classes'
)

# Customize the layout
fig.update_layout(
    title_font_size=20,
    title=dict(x=0.5, xanchor="center"),
    xaxis_title='Class',
    yaxis_title='Flight Distance',
    xaxis_title_font_size=14,
    yaxis_title_font_size=14,
    width=800,  # Narrower plot
    height=600,  # Balanced height
    xaxis_tickangle=0,  # Ensure the category labels stay horizontal
    xaxis_tickmode='linear',  # Adjust tick spacing to ensure proper box width
    margin=dict(l=40, r=40, t=40, b=40)  # Adjust margins to maximize box width
)

# Adjusting the box widths indirectly
fig.update_traces(
    width=0.5  # Manually setting the box width
)
# fig.write_html("box_plot_flight_distance.html")
# Show the plot
fig.show()
# %%
# Grouping data
customer_type_satisfaction = data.groupby(['Customer Type', 'satisfaction']).size().unstack()

# Convert to a DataFrame suitable for plotting
df = customer_type_satisfaction.reset_index().melt(id_vars=['Customer Type'], var_name='Satisfaction', value_name='Count')

# Calculate percentage and round to 2 decimal places
df['Percentage'] = df.groupby('Customer Type')['Count'].transform(lambda x: 100 * x / x.sum())
df['Percentage'] = df['Percentage'].round(2)  # Round to 2 decimals

# Adjust 'Customer Type' values for proper labels
df['Customer Type'] = df['Customer Type'].replace({'disloyal Customer': 'Disloyal Customer'})

# Create the plot using Plotly
fig = px.bar(
    df,
    x='Customer Type',
    y='Count',
    color='Satisfaction',
    barmode='group',
    text='Percentage',  # Display percentage on the bar
    color_discrete_map={
        'satisfied': '#88D498',  # Soft pastel green
        'neutral or dissatisfied': '#F5A9A9'  # Soft pastel red
    },
    title='Relationship between Customer Type and Satisfaction',
    labels={'Customer Type': 'Customer Type'}  # Remove automatic y-axis label setting
)

# Add text inside bars and borders around them
fig.update_traces(
    texttemplate='%{text}',  # Show raw counts on bars
    textposition='inside',  # Position text inside the bars for clarity
    marker_line=dict(width=0.5, color='black')  # Add thinner borders around bars
)

# Update layout to customize the appearance and make the plot narrower
fig.update_layout(
    title=dict(x=0.5, xanchor="center"),
    xaxis_title='Customer Type',
    yaxis_title='Count',  # Explicitly set Y-axis label
    xaxis=dict(title="Customer Type", showgrid=True, linecolor='black'),  # Add x-axis line
    yaxis=dict(linecolor='black'),  # Add y-axis line
    legend_title='Satisfaction',
    template='plotly_white',
    width=800,  # Narrower plot width
    height=600,  # Balanced height
    margin=dict(l=40, r=40, t=40, b=40)  # Tight margins
)

# Update text to make percentage visible
# fig.update_traces(texttemplate='%{text}%', textposition='outside')


# Show the plot
fig.show()

# %%
travel_class_satisfaction_count = data.groupby(['satisfaction', 'Class']).size().unstack()

# Converting to a DataFrame suitable for plotting
df = travel_class_satisfaction_count.reset_index().melt(id_vars=['satisfaction'], var_name='Class', value_name='Count')

df['Percentage'] = df.groupby('satisfaction')['Count'].transform(lambda x: 100 * x / x.sum())
df['Percentage'] = df['Percentage'].round(2)  # Round to 2 decimals

# Adjusting 'Customer Type' values for proper labels
df['satisfaction'] = df['satisfaction'].replace({'neutral or dissatisfied': 'Neutral or Dissatisfied'})
df['satisfaction'] = df['satisfaction'].replace({'satisfied': 'Satisfied'})
# Create the plot using Plotly
fig = px.bar(
    df,
    x='satisfaction',
    y='Count',
    color='Class',
    text='Percentage',
    barmode='group',
    color_discrete_sequence=px.colors.qualitative.Set2,  # Set colors for each class
    title='Relationship between Type of Travel and Satisfaction',
    labels={'satisfaction': 'Satisfaction', 'Count': 'Count'}
)

# Update layout
fig.update_layout(
    title=dict(x=0.5, xanchor="center"),
    xaxis_title='Satisfaction',
    yaxis_title='Count',
    xaxis=dict(title="Satisfaction", showgrid=True, linecolor='black'),  # Add x-axis line
    yaxis=dict(linecolor='black'),
    legend_title='Class',
    template='plotly_white',
    font=dict(size=14),
    width=1100,
    height=600,
    margin=dict(l=40, r=40, t=40, b=40)
)
fig.update_traces( # Position text inside the bars for clarity
    marker_line=dict(width=0.5, color='black'),  # Add thinner borders around bars
    texttemplate='%{text}%', textposition='outside'
)
# fig.write_html("travel_class_satisfaction_count.html")
# Show the plot
fig.show()
# %%
travel_class_count = data.groupby(['Type of Travel', 'Class']).size().unstack()

# Converting to a DataFrame suitable for plotting
df = travel_class_count.reset_index().melt(id_vars=['Type of Travel'], var_name='Class', value_name='Count')

df['Percentage'] = df.groupby('Type of Travel')['Count'].transform(lambda x: 100 * x / x.sum())
df['Percentage'] = df['Percentage'].round(2)  # Round to 2 decimals

# Adjusting 'Customer Type' values for proper labels
df['Type of Travel'] = df['Type of Travel'].replace({'Business travel': 'Business Travel'})
# Creating the plot using Plotly
fig = px.bar(
    df,
    x='Type of Travel',
    y='Count',
    color='Class',
    text='Percentage',
    barmode='group',
    color_discrete_sequence=px.colors.qualitative.Set2,  # Set colors for each class
    title='Relationship between Type of Travel and Class',
    labels={'Type of Travel': 'Type of Travel', 'Count': 'Count'}
)

# Update layout
fig.update_layout(
    title=dict(x=0.5, xanchor="center"),
    xaxis_title='Type of Travel',
    yaxis_title='Count',
    xaxis=dict(title="Type of Travel", showgrid=True, linecolor='black'),  # Add x-axis line
    yaxis=dict(linecolor='black'),
    legend_title='Class',
    template='plotly_white',
    font=dict(size=14),
    width=1100,
    height=600,
    margin=dict(l=40, r=40, t=40, b=40)
)
fig.update_traces( # Position text inside the bars for clarity
    marker_line=dict(width=0.5, color='black'),  # Add thinner borders around bars
    texttemplate='%{text}%', textposition='outside'
)
# fig.write_html("travel_class_count.html")
# Show the plot
fig.show()
# %% [markdown]
# # Modeling
#%%
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

df = pd.concat([train, test], axis=0)

#%%
data = df.drop(columns=['Unnamed: 0', 'id'], errors='ignore')
data.dropna(subset=['Arrival Delay in Minutes'], inplace=True)

#%%
 # Encode categorical variables
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
#%%    
    
# Encoding the target variable
le_target = LabelEncoder()
data['satisfaction'] = le_target.fit_transform(data['satisfaction'])  # 1: Satisfied, 0: Neutral or Dissatisfied

#%%

# Separating features and target variable
X = data.drop(columns=['satisfaction'])
y = data['satisfaction']
#%%
# Standardizing numerical features
numerical_columns = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
scaler = StandardScaler()
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
#%%
# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#%%
# Training a logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
model.fit(X_train, y_train)


#%%
# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]



y_testdata = pd.DataFrame({
    'predicted_probabilities': y_pred_proba,
    'actual_values': y_test
})


class_0_probs = y_testdata[y_testdata['actual_values'] == 0]['predicted_probabilities']
class_1_probs = y_testdata[y_testdata['actual_values'] == 1]['predicted_probabilities']
#%%
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

#%%

plt.figure(figsize=(10, 6))
sns.histplot(y_pred_proba, kde=True, bins=30, color='skyblue', stat='density')
plt.title('Distribution of Predicted Probabilities from Logistic Regression')
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.show()

#%%
# Evaluating the model
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"Accuracy: {accuracy:.2f}")
print(f"ROC-AUC: {roc_auc:.2f}")


#%%
# Displaying simplified classification report visually
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

# Plotting classification report as bar chart
plt.figure(figsize=(8, 6))
sns.barplot(x='Metric', y='Score', data=summary_df, palette='viridis')
plt.title('Classification Report Metrics')
plt.ylim(0, 1)
plt.ylabel('Score')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#%%
# Plotting Enhanced ROC curve
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

#%%


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
#%%
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

#%%

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

#%%

# Separate features and target variable
X = data.drop(columns=['satisfaction'])
y = data['satisfaction']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#%%

# Evaluating the model
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
cm = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

#%%

# Getting feature importance
feature_importance = model.feature_importances_
feature_names = X_train.columns

# Creating a DataFrame for better visualization
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

#%%

cm = confusion_matrix(y_test, y_pred)

# Normalizing the confusion matrix to show percentages
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))

# Setting a refined style
sns.set(style="whitegrid", palette="pastel")  

# Creating the heatmap with improved aesthetics
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

# Adjusting tick parameters for better legibility
plt.xticks(fontsize=13, rotation=0)
plt.yticks(fontsize=13, rotation=0)
plt.tight_layout()
plt.show()
