import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the data
data = pd.read_csv('Unemployment in India.csv')

# Step 2: Preview the data
print(data.head())

# Step 3: Convert the date column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Step 4: Set the Date as the index
data.set_index('Date', inplace=True)

# Step 5: Plotting the unemployment rate
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x=data.index, y='Unemployment_Rate', marker='o')
plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')

# Highlight key events
plt.axvline(pd.Timestamp('2020-03-01'), color='red', linestyle='--', label='COVID-19 Start')
plt.axvline(pd.Timestamp('2020-04-01'), color='orange', linestyle='--', label='Peak Unemployment')

plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
