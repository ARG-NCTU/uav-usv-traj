import csv
import math
import random
from datetime import datetime, timedelta

# Helper function to calculate new GPS coordinates after moving a certain distance
def move_to_new_position(lon, lat, distance, angle, drift=0):
    R = 6378137  # Earth radius in meters
    angle_with_drift = angle + drift  # Apply drift to the angle
    d_lon = distance * math.sin(math.radians(angle_with_drift)) / (R * math.cos(math.radians(lat)))
    d_lat = distance * math.cos(math.radians(angle_with_drift)) / R

    new_lon = lon + math.degrees(d_lon)
    new_lat = lat + math.degrees(d_lat)

    return new_lon, new_lat

# Starting position
start_lon = 120.9970239959511
start_lat = 24.785754866828896
trajectory_id = 1
timestamp = datetime(2024, 8, 27, 14, 37, 0)

# Initialize data list
data = []

# Function to add points to the CSV data list
def add_point(lon, lat, id, traj_id, timestamp):
    data.append([lon, lat, id, traj_id, timestamp.isoformat()])

# Generate initial manual control points
id = 1
for _ in range(20):  # Add 20 random points before BT starts
    lon, lat = move_to_new_position(start_lon, start_lat, random.uniform(2, 4), random.uniform(0, 360))
    add_point(lon, lat, id, trajectory_id, timestamp)
    timestamp += timedelta(seconds=random.randint(1, 3))
    id += 1

# Execute the behavior tree (BT)
bt_steps = [
    # ("explore_forward", 0),
    # ("rotate_left", -90),
    # ("explore_forward", -90),
    # ("rotate_left", 179.9),
    # ("explore_forward", 179.9),
    # ("rotate_right", -90),
    # ("explore_forward", -90),
    # ("rotate_right", 0),
    # ("explore_forward", 0),
    # ("rotate_left", -90),
    # ("explore_forward", -90),
    # ("rotate_left", 179.9),
    # ("explore_forward", 179.9),
    # # Move to next area
    # ("rotate_right", -90),
    # ("explore_forward", -90),
    # ("explore_forward", -90),
    # ("explore_forward", -90),
    # ("rotate_right", 0),
    # ##############################
    # ("explore_forward", 0),
    # ("rotate_left", -90),
    # ("explore_forward", -90),
    # ("rotate_left", 179.9),
    # ("explore_forward", 179.9),
    # ("rotate_right", -90),
    # ("explore_forward", -90),
    # ("rotate_right", 0),
    # ("explore_forward", 0),
    # ("rotate_left", -90),
    # ("explore_forward", -90),
    # ("rotate_left", 179.9),
    # ("explore_forward", 179.9),
    ("explore_forward", 0),
    ("explore_forward", 0),
    ("explore_forward", 0),
    ("explore_forward", 0),
    ("explore_forward", 0),
    ("rotate_left", -90),
    ("explore_forward", -90),
    ("explore_forward", -90),
    ("explore_forward", -90),
    ("explore_forward", -90),
    ("explore_forward", -90),
    ("rotate_left", 179.9),
    ("explore_forward", 179.9),
    ("explore_forward", 179.9),
    ("explore_forward", 179.9),
    ("explore_forward", 179.9),
    ("explore_forward", 179.9),
    ("rotate_left", 90),
    ("explore_forward", 90),
    ("explore_forward", 90),
    ("explore_forward", 90),
    ("explore_forward", 90),
    ("rotate_left", 0),
    ("explore_forward", 0),
    ("explore_forward", 0),
    ("explore_forward", 0),
    ("explore_forward", 0),
    ("rotate_left", -90),
    ("explore_forward", -90),
    ("explore_forward", -90),
    ("explore_forward", -90),
    ("rotate_left", 179.9),
    ("explore_forward", 179.9),
    ("explore_forward", 179.9),
    ("explore_forward", 179.9),
    ("rotate_left", 90),
    ("explore_forward", 90),
    ("explore_forward", 90),
    ("rotate_left", 0),
    ("explore_forward", 0),
    ("explore_forward", 0),
    ("rotate_left", -90),
    ("explore_forward", -90),
    ("rotate_left", 179.9),
    ("explore_forward", 179.9),

]

current_lon = lon
current_lat = lat

for step, rotation in bt_steps:
    if step == "explore_forward":
        num_points = 50  # More points for straight lines
        for _ in range(num_points):  # Add points while moving forward
            drift = random.uniform(-5, 5)  # Left-right drift in degrees
            current_lon, current_lat = move_to_new_position(current_lon, current_lat, 0.1 + random.uniform(-0.01, 0.01), rotation, drift=drift)
            add_point(current_lon, current_lat, id, trajectory_id, timestamp)
            timestamp += timedelta(seconds=random.randint(1, 2))
            id += 1
    elif step == "rotate_left":
        for _ in range(10):  # Add more points while rotating to simulate fine control
            current_lon, current_lat = move_to_new_position(current_lon, current_lat, random.uniform(-0.01, 0.01), rotation)
            add_point(current_lon, current_lat, id, trajectory_id, timestamp)
            timestamp += timedelta(seconds=random.randint(1, 2))
            id += 1
    elif step == "rotate_right":
        for _ in range(10):  # Add more points while rotating to simulate fine control
            current_lon, current_lat = move_to_new_position(current_lon, current_lat, random.uniform(-0.01, 0.01), rotation)
            add_point(current_lon, current_lat, id, trajectory_id, timestamp)
            timestamp += timedelta(seconds=random.randint(1, 2))
            id += 1

# Generate final manual control points
for _ in range(20):  # Add 20 random points after BT ends
    current_lon, current_lat = move_to_new_position(current_lon, current_lat, random.uniform(2, 4), random.uniform(0, 360))
    add_point(current_lon, current_lat, id, trajectory_id, timestamp)
    timestamp += timedelta(seconds=random.randint(1, 3))
    id += 1

# Write to CSV
with open("data/uav-csv/generated_trajectory.csv", "w", newline='') as file:
    writer = csv.writer(file, delimiter=';')
    writer.writerow(["Longtitude", "Latitude", "ID", "Trajectory_id", "Timestamp"])
    writer.writerows(data)

print("CSV file generated: 'data/uav-csv/generated_trajectory.csv'")
