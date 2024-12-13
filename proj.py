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
# %% [markdown]
# # Data Overview
# **Gender:** Gender of the passengers (Female, Male)
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

# Aggregate counts for satisfaction group and gender
grouped_data = (
    data.groupby(['satisfaction_group', 'Gender']).size().reset_index(name='count')
)

# Calculate overall satisfaction group proportions (for bar heights)
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

# Add annotations for percentages outside the bars
for satisfaction_group, percentage in zip(total_counts['satisfaction_group'], total_counts['total'] / total_counts['total'].sum() * 100):
    fig.add_annotation(
        x=satisfaction_group,
        y=total_counts.loc[total_counts['satisfaction_group'] == satisfaction_group, 'total'].values[0] + 2500,  # Position above the bar further up
        text=f"{percentage:.1f}%",  # Format percentage to 1 decimal place
        showarrow=False,
        font=dict(size=14, color="black"),
        align="center"  # Align text in the center above the bar
    )

# Add annotations for gender breakdown within each satisfaction group
fig.update_traces(
    hovertemplate="<b>%{y}</b><br>Count: %{text}<br>Gender: %{customdata[0]}<br>Proportion: %{customdata[1]:.1f}%",
    customdata=['Gender', 'gender_percentage']
)

# Improve layout
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

    # Create the figure
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

# Customize the layout
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

# Convert to a DataFrame suitable for plotting
df = travel_class_satisfaction_count.reset_index().melt(id_vars=['satisfaction'], var_name='Class', value_name='Count')

df['Percentage'] = df.groupby('satisfaction')['Count'].transform(lambda x: 100 * x / x.sum())
df['Percentage'] = df['Percentage'].round(2)  # Round to 2 decimals

# Adjust 'Customer Type' values for proper labels
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

# Convert to a DataFrame suitable for plotting
df = travel_class_count.reset_index().melt(id_vars=['Type of Travel'], var_name='Class', value_name='Count')

df['Percentage'] = df.groupby('Type of Travel')['Count'].transform(lambda x: 100 * x / x.sum())
df['Percentage'] = df['Percentage'].round(2)  # Round to 2 decimals

# Adjust 'Customer Type' values for proper labels
df['Type of Travel'] = df['Type of Travel'].replace({'Business travel': 'Business Travel'})
# Create the plot using Plotly
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
# %%
