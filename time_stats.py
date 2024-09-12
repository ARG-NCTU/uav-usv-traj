import pandas as pd
data = pd.read_csv("data/uav-csv/ALL.csv")
df = pd.DataFrame(data)
df_split = df['Trajectory_file_path;Trajectory_id;Longitude;Latitude;Total_distance;Timestamp'].str.split(';', expand=True)
df_split.columns = ['Trajectory_file_path', 'Trajectory_id', 'Longitude', 'Latitude', 'Total_distance', 'Timestamp']

# read data/uav-csv/統整飛行軌跡.xlsx
df2 = pd.read_excel("data/uav-csv/Integrate.xlsx")
# Remove duplicate rows
df2.drop_duplicates(inplace=True)
# Add a column 'Timestamp' to df2, read the 'Flight time' column and split it, '2023-10-02 10:55:20-11:02:05' -> '2023-10-02T10:55:20'
df2['Timestamp'] = df2['Flight time'].apply(lambda x: x.split(' ')[0] + 'T' + x.split(' ')[1].split('-')[0])
# print(df2['Timestamp'])
# print each of number of '作物' in dict
print(df2['作物'].value_counts())
# If item of df_split['Timestamp'] is same as item of df2['Timestamp'], add item of df2['作物'] to item of df_split['Crop']
df_split['Crop'] = df_split['Timestamp'].apply(lambda x: df2['作物'][df2['Timestamp'] == x].values[0] if len(df2['Timestamp'][df2['Timestamp'] == x]) > 0 else 'Unknown')
print(df_split['Crop'].value_counts())
# Save the new dataframe to a new csv file
# df_split.copy().to_csv("data/uav-csv/ALL_with_crop.csv", index=False)
print('The new csv file has been saved as data/uav-csv/ALL_with_crop.csv.')

# If df_split['Timestamp'] has a same value as df2['Timestamp'], set df_split['belongs_to'] to 'AB', else set it to 'A'
df_split['belongs_to'] = df_split['Timestamp'].apply(lambda x: 'A&B' if len(df2['Timestamp'][df2['Timestamp'] == x]) > 0 else 'A-B')

# If df2['Timestamp'] has a same value as df_split['Timestamp'], set df2['belongs_to'] to 'AB', else set it to 'B'
df2['belongs_to'] = df2['Timestamp'].apply(lambda x: 'A&B' if len(df_split['Timestamp'][df_split['Timestamp'] == x]) > 0 else 'B-A')


# add a new column 'flight_time' to df_split, first read the df['Trajectory_file_path'] csv file path, then count the number of rows in the csv file as the seconds of the flight_time
df_split['flight_time'] = 0
for i in range(len(df_split)):
    path = df_split['Trajectory_file_path'][i]
    data_i = pd.read_csv(path)
    df_i = pd.DataFrame(data_i)
    df_split.loc[i, 'flight_time'] = len(df_i.copy())


# Use matplotlib to plot each month of the flight_time of df_split['Crop']
import matplotlib.pyplot as plt
df_split['month'] = df_split['Timestamp'].apply(lambda x: x.split('-')[1])
df_split['month'] = df_split['month'].astype(int)

# Dict to map month numbers to month names
month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
               7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

# First, let's modify the data to separate '水稻' and '其他'
df_split['Crop_group'] = df_split['Crop'].apply(lambda x: '水稻' if x == '水稻' else ('其他' if x == '其他' else None))

# Now, we'll create two separate bar plots: one for '水稻' and one for '其他'
# Dictionary to hold flight times per month for each crop group
month_flight_time_rice = {}
month_flight_time_other = {}

for i in range(1, 13):
    month_flight_time_rice[i] = df_split['flight_time'][(df_split['month'] == i) & (df_split['Crop_group'] == '水稻')].sum() / 3600
    month_flight_time_other[i] = df_split['flight_time'][(df_split['month'] == i) & (df_split['Crop_group'] == '其他')].sum() / 3600

# Create the bar plot for '水稻'
plt.figure(figsize=(10, 6))
plt.bar(month_flight_time_rice.keys(), month_flight_time_rice.values(), label='Rice', color='blue')

# Create the bar plot for '其他' on the same graph
plt.bar(month_flight_time_other.keys(), month_flight_time_other.values(), label='Other crops', color='green')

# Use the month names as x-ticks
plt.xticks(ticks=range(1, 13), labels=[month_names[i] for i in range(1, 13)])

# Set labels and title
plt.xlabel('Month of 2023')
plt.ylabel('Flight time (hours)')
plt.title('Flight time of each month of 2023 for rice and other crops')

# Add a legend to differentiate between '水稻' and '其他'
plt.legend()

# Save the plot
plt.savefig('data/flight_time_of_each_month_rice_vs_other.png')


# Plot week as x-axis
df_split['week'] = df_split['Timestamp'].apply(lambda x: pd.Timestamp(x).week)
df_split['week'] = df_split['week'].astype(int)

# Dictionary to hold flight times per week for each crop group
week_flight_time_rice = {}
week_flight_time_other = {}

for i in range(1, 53):
    flight_time_rice = df_split['flight_time'][(df_split['week'] == i) & (df_split['Crop_group'] == '水稻')].sum()
    flight_time_other = df_split['flight_time'][(df_split['week'] == i) & (df_split['Crop_group'] == '其他')].sum()
    if flight_time_rice > 0:
        week_flight_time_rice[i] = flight_time_rice / 3600
    if flight_time_other > 0:
        week_flight_time_other[i] = flight_time_other / 3600

# Create the bar plot for '水稻'
plt.figure(figsize=(20, 6))
plt.xticks(ticks=range(min(week_flight_time_rice.keys()) - 1, max(week_flight_time_rice.keys()) + 1))
plt.bar(week_flight_time_rice.keys(), week_flight_time_rice.values(), label='Rice', color='blue')

# Create the bar plot for '其他' on the same graph
plt.bar(week_flight_time_other.keys(), week_flight_time_other.values(), label='Other crops', color='green') 

# Set labels and title
plt.xlabel('Week of 2023')
plt.ylabel('Flight time (hours)')
plt.title('Flight time of each week of 2023 for rice and other crops')

# Add a legend to differentiate between '水稻' and '其他'
plt.legend()

# Save the plot
plt.savefig('data/flight_time_of_each_week_rice_vs_other.png')

# Plot the Venn diagram of the number of df_split['belongs_to'] 'A&B' and 'A-B' with label df_split['Crop_group'], df2['belongs_to'] 'A&B' and 'B-A' with label df2['作物'], A&B show once
from matplotlib_venn import venn2

# Count occurrences of 'A&B' and 'A-B' in df_split['belongs_to']
split_counts = df_split['belongs_to'].value_counts()

# Count occurrences of 'A&B' and 'B-A' in df2['belongs_to']
df2_counts = df2['belongs_to'].value_counts()
print('df2_counts:', df2_counts)

# Set the sizes for the Venn diagram
venn_data = {
    'A': split_counts.get('A-B', 0),  # Total for A (A-B)
    'B': df2_counts.get('B-A', 0),      # Total for B (B-A)
    'AB': split_counts.get('A&B', 0)  # Overlapping count for A&B
}

# Plot Venn diagram
plt.figure(figsize=(6, 6))
venn = venn2(subsets=(venn_data['A'], venn_data['B'], venn_data['AB']),
             set_labels=('Our trajectory stats', "Earthgen's trajectory stats"))

# Set the fill colors
venn.get_patch_by_id('10').set_color('red')    # A-B filled with red
venn.get_patch_by_id('01').set_color('blue')   # B-A filled with blue
venn.get_patch_by_id('11').set_color('green')  # A&B filled with green

# Optional: Set transparency (alpha) for a better overlap visual effect
venn.get_patch_by_id('10').set_alpha(0.5)
venn.get_patch_by_id('01').set_alpha(0.5)
venn.get_patch_by_id('11').set_alpha(0.5)

# Save and display the Venn diagram
plt.title("Venn Diagram of our trajectory stats and Earthgen's trajectory stats")
plt.savefig('data/venn_diagram.png')
# plt.show()



