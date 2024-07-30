####################################################################################################

# Example 1: What is the average of all distance of trajectories?
import pandas as pd
data = pd.read_csv("data/uav-csv/ALL.csv")
df = pd.DataFrame(data)
df_split = df['Trajectory_file_path;Trajectory_id;Longtitude;Latitude;Total_distance;Timestamp'].str.split(';', expand=True)
df_split.columns = ['Trajectory_file_path', 'Trajectory_id', 'Longtitude', 'Latitude', 'Total_distance', 'Timestamp']

total_distances = df_split["Total_distance"].astype(float).tolist()
# Calculate the sum using a for loop avoiding the use of the df.sum() function
total_sum = 0
for distance in total_distances:
    total_sum += distance

# Calculate the average
average_distance = total_sum / len(total_distances)
print('The average distance of all trajectories is', average_distance, 'meters.')

####################################################################################################

# Example 2: Which distance of the trajectory is the longest??
import pandas as pd
data = pd.read_csv("data/uav-csv/ALL.csv")
df = pd.DataFrame(data)
df_split = df['Trajectory_file_path;Trajectory_id;Longtitude;Latitude;Total_distance;Timestamp'].str.split(';', expand=True)
df_split.columns = ['Trajectory_file_path', 'Trajectory_id', 'Longtitude', 'Latitude', 'Total_distance', 'Timestamp']

total_distances = df_split["Total_distance"].astype(float).tolist()
# Calculate the sum using a for loop avoiding the use of the df.sum() function
longest_distance = 0
for distance in total_distances:
    if distance > longest_distance:
        longest_distance = distance

print('The longest distance of the trajectory is', longest_distance, 'meters.')

# Find the row with the longest distance
all_rows = df_split.values.tolist()
longest_distance_trajectory = []
for row in all_rows:
    if float(row[4]) == longest_distance:
        longest_distance_trajectory = row
        break
print('The trajectory with the longest distance is:')
print(longest_distance_trajectory)


####################################################################################################

# Example 3: Which trajectories are in the period from 2023/07/16 7AM to 2023/07/16 8AM?
import pandas as pd
data = pd.read_csv("data/uav-csv/ALL.csv")
df = pd.DataFrame(data)
df_split = df['Trajectory_file_path;Trajectory_id;Longtitude;Latitude;Total_distance;Timestamp'].str.split(';', expand=True)
df_split.columns = ['Trajectory_file_path', 'Trajectory_id', 'Longtitude', 'Latitude', 'Total_distance', 'Timestamp']

# format of time input: "YYYY-MM-DDTHH:MM:SS"
start_time = "2023-07-16T07:00:00"
end_time = "2023-07-16T08:00:00"
time_range = df_split['Timestamp'].tolist()

# Find the row with the longest distance
all_rows = df_split.values.tolist()
period_trajectories = []
for row in all_rows:
    if row[5] >= start_time and row[5] <= end_time:
        period_trajectories.append(row)
print('The trajectories in the period from', start_time, 'to', end_time, 'are:')
# print 10 first trajectories if there are too many
if len(period_trajectories) > 10:
    for i in range(10):
        print(period_trajectories[i])
else:    
    for trajectory in period_trajectories:
        print(trajectory)
        
####################################################################################################
