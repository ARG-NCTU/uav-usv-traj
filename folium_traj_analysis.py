import pandas as pd
import geopandas as gpd
import movingpandas as mpd
import math
import folium as fl
from streamlit_folium import st_folium
import streamlit as st
import matplotlib.pyplot as plt
from geopy.distance import geodesic

st.set_page_config(page_title="Trajectory Analysis")
st.header("Trajectory Analysis To Generate Custom Exploration Pattern Block Behavior Tree")

####################################################################################################

# Load ALL trajectories
data = pd.read_csv("data/uav-csv/ALL.csv")
df_all = pd.DataFrame(data)
df_all_split = df_all['Trajectory_file_path;Trajectory_id;Longitude;Latitude;Total_distance;Timestamp'].str.split(';', expand=True)
df_all_split.columns = ['Trajectory_file_path', 'Trajectory_id', 'Longitude', 'Latitude', 'Total_distance', 'Timestamp']

# Convert latitude and Longitude to float
df_all_split['Latitude'] = df_all_split['Latitude'].astype(float)
df_all_split['Longitude'] = df_all_split['Longitude'].astype(float)
df_all_split['Total_distance'] = df_all_split['Total_distance'].astype(float)

# Display the center of all trajectories
st.write("Displaying the centers of all trajectories on the map...")
st.write("Click on the map to analysis nearby trajectories if you want.")

# Calculate the center of all trajectories
center_of_centers = df_all_split.groupby('Trajectory_id')[['Latitude', 'Longitude']].mean()

# Randomly select 100 trajectories if there are more than 100
if len(center_of_centers) > 100:
    random_trajectory_ids = center_of_centers.sample(n=100, random_state=1).index
    center_of_centers = center_of_centers[center_of_centers.index.isin(random_trajectory_ids)]

# Create a Folium map centered around the mean latitude and longitude
center_lat, center_lng = center_of_centers.mean()
map_folium_all = fl.Map(location=[center_lat, center_lng], zoom_start=10)

# Plot the centers of all trajectories on the map
for idx, row in center_of_centers.iterrows():
    fl.Marker(location=[row['Latitude'], row['Longitude']]).add_to(map_folium_all)

# Add LatLngPopup to capture user's click
map_folium_all.add_child(fl.LatLngPopup())

# Display the map in Streamlit
output_all = st_folium(map_folium_all, width=700, height=500)

# Process the selected point on the map
if output_all.get("last_clicked"):
    clicked_lat = output_all["last_clicked"]["lat"]
    clicked_lng = output_all["last_clicked"]["lng"]
    st.write(f"Selected point: ({clicked_lat}, {clicked_lng})")
else:
    clicked_lat = center_lat
    clicked_lng = center_lng

# Ask the user the top number of trajectories which meet the criteria
top_n = st.slider("Select the top number of trajectories which meet the criteria to analyze:", min_value=1, max_value=5, value=1)

# Streamlit multiselect to choose which analysises to perform (Time, Location, Total distance)
selected_options = st.multiselect("Select the analysis to perform (multiselection):", ["Time", "Location", "Total distance"])

# If "Time" is selected
if "Time" in selected_options:
    # Convert the 'Timestamp' column to datetime
    df_all_split['Timestamp'] = pd.to_datetime(df_all_split['Timestamp'])

    # Ask the user the starting timestamp and ending timestamp for the analysis
    st.write("Select the starting and ending timestamps for the analysis (default is the entire dataset range):")
    start_date = st.date_input("Select the starting date for the analysis:", value=df_all_split['Timestamp'].min().date())
    start_time = st.time_input("Select the starting time for the analysis:", value=df_all_split['Timestamp'].min().time())
    end_date = st.date_input("Select the ending date for the analysis:", value=df_all_split['Timestamp'].max().date())
    end_time = st.time_input("Select the ending time for the analysis:", value=df_all_split['Timestamp'].max().time())

    # Change the format of the selected timestamps
    start_timestamp = f"{start_date}T{start_time}"
    end_timestamp = f"{end_date}T{end_time}"

    # Filter the trajectories based on the selected timestamp range
    df_all_filtered = df_all_split[(df_all_split['Timestamp'] >= start_timestamp) & (df_all_split['Timestamp'] <= end_timestamp)].copy()
else:
    df_all_filtered = df_all_split.copy()

# Function to find the closest trajectories
def find_closest_trajectories(lat, lng, df):
    selected_point = (lat, lng)
    df['distance'] = df.apply(lambda row: geodesic(selected_point, (row['Latitude'], row['Longitude'])).km, axis=1)
    return df.sort_values('distance')

# Perform the analysis based on priority
if "Location" in selected_options and "Total distance" in selected_options:
    index_location = selected_options.index("Location")
    index_total_distance = selected_options.index("Total distance")

    if index_location < index_total_distance:
        # Perform location-based analysis first
        st.write("Analyzing based on location first...")
        df_all_filtered = find_closest_trajectories(clicked_lat, clicked_lng, df_all_filtered).copy()

        df_all_filtered = df_all_filtered.head(top_n).copy()

        # Then sort by total distance
        analysis_options = ["shortest", "longest"]
        selected_analysis = st.selectbox("Select the total distance analysis type:", analysis_options, index=1)

        if selected_analysis == "shortest":
            df_all_filtered = df_all_filtered.nsmallest(len(df_all_filtered), 'Total_distance').copy()
        elif selected_analysis == "longest":
            df_all_filtered = df_all_filtered.nlargest(len(df_all_filtered), 'Total_distance').copy()

    else:
        # Perform total distance-based analysis first
        st.write("Analyzing based on total distance first...")
        analysis_options = ["shortest", "longest"]
        selected_analysis = st.selectbox("Select the total distance analysis type:", analysis_options, index=0)

        if selected_analysis == "shortest":
            df_all_filtered = df_all_filtered.nsmallest(len(df_all_filtered), 'Total_distance').copy()
        elif selected_analysis == "longest":
            df_all_filtered = df_all_filtered.nlargest(len(df_all_filtered), 'Total_distance').copy()

        df_all_filtered = df_all_filtered.head(top_n).copy()

        # Then sort by location
        df_all_filtered = find_closest_trajectories(clicked_lat, clicked_lng, df_all_filtered).copy()

elif "Location" in selected_options:
    # If only Location is selected
    st.write("Analyzing based on location...")
    df_all_filtered = find_closest_trajectories(clicked_lat, clicked_lng, df_all_filtered).copy()

elif "Total distance" in selected_options:
    # If only Total distance is selected
    st.write("Analyzing based on total distance...")
    analysis_options = ["shortest", "longest"]
    selected_analysis = st.selectbox("Select the total distance analysis type:", analysis_options, index=0)

    if selected_analysis == "shortest":
        df_all_filtered = df_all_split.nsmallest(len(df_all_split), 'Total_distance').copy()
    elif selected_analysis == "longest":
        df_all_filtered = df_all_split.nlargest(len(df_all_split), 'Total_distance').copy()

# Apply top_n filtering after all other filters
df_all_filtered = df_all_filtered.head(top_n).copy()

# Display the filtered DataFrame
st.write(df_all_filtered)

number_of_trajectories = len(df_all_filtered)
st.write(f"Analyzing the {number_of_trajectories} selected trajectories based on the criteria.")


####################################################################################################


# Create tabs for each trajectory
tabs = st.tabs([f"Trajectory {i+1}" for i in range(number_of_trajectories)])

# Iterate over the selected trajectories and display analysis in each tab
for i, tab in enumerate(tabs):
    with tab:
        # Load the CSV data
        csv_file_path = df_all_filtered['Trajectory_file_path'].iloc[i]
        data = pd.read_csv(csv_file_path)
        df = pd.DataFrame(data)
        df_split = df['Longitude;Latitude;ID;Trajectory_id;Timestamp'].str.split(';', expand=True)
        df_split.columns = ['Longitude', 'Latitude', 'ID', 'Trajectory_id', 'Timestamp']

        # Convert the necessary columns to appropriate data types
        df_split['Longitude'] = df_split['Longitude'].astype(float)
        df_split['Latitude'] = df_split['Latitude'].astype(float)
        df_split['Timestamp'] = pd.to_datetime(df_split['Timestamp']).dt.tz_localize(None)

        # Streamlit slider to select BT start and end points based on latitude and Longitude
        start_index = st.slider('Select the start point of the BT:', 0, len(df_split)-1, 1, key=f'start_index_{i}')
        end_index = st.slider('Select the end point of the BT:', 0, len(df_split)-1, len(df_split)-2, key=f'end_index_{i}')

        # Describe this graph
        st.write("The blue line represents the trajectory of the UAV. The solid line represents the trajectory controlled by the BT, while the dashed lines represent the manual control before and after the BT.")

        # Initialize map with a default center and zoom level
        if 'map_center' not in st.session_state:
            st.session_state.map_center = [df_split['Latitude'].mean(), df_split['Longitude'].mean()]
        if 'zoom_level' not in st.session_state:
            st.session_state.zoom_level = 15

        # Initialize the map with stored zoom level and center
        map_folium = fl.Map(location=st.session_state.map_center, zoom_start=st.session_state.zoom_level)

        # Plot manual control before BT (more slightly transparent and dotted)
        fl.PolyLine(locations=df_split[['Latitude', 'Longitude']].iloc[:start_index].values, color="blue", opacity=0.5, dash_array='5').add_to(map_folium)

        # Plot BT controlled trajectory
        fl.PolyLine(locations=df_split[['Latitude', 'Longitude']].iloc[start_index:end_index+1].values, color="blue").add_to(map_folium)

        # Plot manual control after BT (only if there are points after the BT)
        if end_index + 1 < len(df_split):
            fl.PolyLine(locations=df_split[['Latitude', 'Longitude']].iloc[end_index+1:].values, color="blue", opacity=0.5, dash_array='5').add_to(map_folium)

        # Display the map and capture the new zoom level and center
        output = st_folium(map_folium, width=700, height=500)

        # Update the session state with the new center and zoom level if they change
        if output.get('zoom'):
            st.session_state.zoom_level = output['zoom']
        if output.get('center'):
            st.session_state.map_center = [output['center']['lat'], output['center']['lng']]

        st.write("Analysis is started...")

        # Filter the DataFrame to the selected range
        df_filtered = df_split.iloc[start_index:end_index+1].copy()

        # Convert Longitude and latitude to x and y coordinates
        origin_long = df_filtered['Longitude'].iloc[0]
        origin_lat = df_filtered['Latitude'].iloc[0]

        df_filtered['x'] = (df_filtered['Longitude'] - origin_long) * 111320 * math.cos(math.radians(origin_lat))
        df_filtered['y'] = (df_filtered['Latitude'] - origin_lat) * 110540

        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame(df_filtered, geometry=gpd.points_from_xy(df_filtered.x, df_filtered.y))

        # Ensure the GeoDataFrame has the correct CRS (Coordinate Reference System)
        gdf.set_crs(epsg=4326, inplace=True)

        # Group the data by 'Trajectory_id' to create trajectories
        trajectory_collection = mpd.TrajectoryCollection(gdf, traj_id_col='Trajectory_id', t='Timestamp', x='x', y='y')

        # Take the first trajectory for demonstration (if there are multiple)
        trajectory = trajectory_collection.trajectories[0]

        # Clean the trajectory by removing outliers
        cleaned = mpd.OutlierCleaner(trajectory).clean(alpha=2)

        # Smooth the cleaned trajectory using Kalman Smoother
        process_noise_std = st.slider('Set noise standard deviation for Kalman Smoother:', 0.0, 5.0, 2.0, key=f'process_noise_std_{i}')
        measurement_noise_std = st.slider('Set measurement noise standard deviation for Kalman Smoother:', 0.1, 10.0, 5.0, key=f'measurement_noise_std_{i}')
        smoothed = mpd.KalmanSmootherCV(cleaned).smooth(process_noise_std=process_noise_std, measurement_noise_std=measurement_noise_std)

        # Function to calculate the distance between two points
        def calculate_distance(point1, point2):
            return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5
        
        # Initialize variables for tracking movement and coordinates
        previous_point = None
        previous_direction = None
        cumulative_direction_change = 0
        cumulative_distance_change = 0

        # Sliders to manually adjust the direction and distance thresholds for simplifying the trajectory
        direction_for_simplify = st.slider('Set direction (degree) threshold for simplifying trajectory:', 1, 180, 10, key=f'direction_for_simplify_{i}')
        distance_for_simplify = st.slider('Set distance (meter) threshold for simplifying trajectory:', 0.1, 10.0, 1.0, key=f'distance_for_simplify_{i}')

        # Lists to store coordinates for original, smoothed, and simplified trajectories
        x_coords_original = []
        y_coords_original = []
        x_coords_smoothed = []
        y_coords_smoothed = []
        x_coords_simplified = []
        y_coords_simplified = []
        yaw_simplified = []

        # First loop: Generate simplified trajectory
        for index, point in smoothed.df.iterrows():
            # Original coordinates
            original_point = trajectory.df.loc[index, 'geometry']
            x_coords_original.append(original_point.x)
            y_coords_original.append(original_point.y)
            
            # Smoothed coordinates
            x_coords_smoothed.append(point.geometry.x)
            y_coords_smoothed.append(point.geometry.y)
            
            # Calculate direction and distance
            if previous_point is not None:
                # Calculate the current direction
                diff_x = point.geometry.x - previous_point.x
                diff_y = point.geometry.y - previous_point.y
                current_direction = math.degrees(math.atan2(diff_y, diff_x))
                
                # Calculate the distance from the previous point
                distance_change = calculate_distance(previous_point, point.geometry)
                cumulative_distance_change += distance_change
                
                if previous_direction is not None:
                    # Calculate the angle difference
                    angle_difference = current_direction - previous_direction
                    cumulative_direction_change += angle_difference 
                    cumulative_direction_change = (cumulative_direction_change + 180) % 360 - 180  # Normalize to [-180, 180] range
                    
                    # Simplify the trajectory: Only keep points with significant direction or distance change
                    if abs(cumulative_direction_change) >= direction_for_simplify or cumulative_distance_change >= distance_for_simplify:
                        x_coords_simplified.append(previous_point.x)
                        y_coords_simplified.append(previous_point.y)
                        yaw_simplified.append(previous_direction)

                        # Reset cumulative direction and distance changes
                        cumulative_direction_change = 0
                        cumulative_distance_change = 0
                
                # Update the previous direction
                previous_direction = current_direction
            
            # # Ensure the first point is added to the simplified trajectory
            # if previous_point is None:
            #     x_coords_simplified.append(point.geometry.x)
            #     y_coords_simplified.append(point.geometry.y)
            #     yaw_simplified.append(current_direction)
            
            previous_point = point.geometry

        # Ensure the last point is included in the simplified trajectory
        if (x_coords_simplified[-1], y_coords_simplified[-1]) != (x_coords_smoothed[-1], y_coords_smoothed[-1]):
            x_coords_simplified.append(x_coords_smoothed[-1])
            y_coords_simplified.append(y_coords_smoothed[-1])
            yaw_simplified.append(current_direction)

        
        # # Second loop: Generate behavior tree using the simplified trajectory
        # behavior_tree_plan = []

        # # Function to calculate direction between two points
        # def calculate_direction(x1, y1, x2, y2):
        #     return math.degrees(math.atan2(y2 - y1, x2 - x1))

        # # Function to calculate the difference between two angles
        # def angle_difference(angle1, angle2):
        #     diff = (angle2 - angle1 + 180) % 360 - 180
        #     return diff

        # # Function to update position based on direction and distance
        # def move_forward(x, y, distance, direction):
        #     new_x = x + distance * math.cos(math.radians(direction))
        #     new_y = y + distance * math.sin(math.radians(direction))
        #     return new_x, new_y
        
        # # Sliders to manually adjust the behavior tree thresholds
        # explore_forward_m = st.slider('Set forward distance (meter) threshold for "explore_forward" action:', 0.0, 10.0, 1.0, key=f'explore_forward_m_{i}')
        # rotation_degree = st.slider('Set direction (degree) threshold for "rotate_left" and "rotate_right" actions:', 1, 180, 10, key=f'rotation_degree_{i}')
        # # distance_threshold = st.slider('Set distance threshold for "rotate_left" and "rotate_right" action:', 0.0, 10.0, 1.0, key=f'distance_threshold_{i}')

        # # Initialize variables
        # current_x = x_coords_simplified[0]
        # current_y = y_coords_simplified[0]
        # current_direction = calculate_direction(x_coords_simplified[0], y_coords_simplified[0], x_coords_simplified[1], y_coords_simplified[1])

        # # Lists to store coordinates for bt
        # x_coords_bt = [current_x]
        # y_coords_bt = [current_y]
            
        # # Loop through the simplified trajectory points to generate the behavior tree
        # for index in range(1, len(x_coords_simplified) - 1):
        #     next_x = x_coords_simplified[index + 1]
        #     next_y = y_coords_simplified[index + 1]

        #     # Calculate distance between current and next point
        #     distance = calculate_distance(gpd.points_from_xy([current_x], [current_y])[0], gpd.points_from_xy([next_x], [next_y])[0])
        #     # if distance > distance_threshold:
        #     if True:
        #         # Add 'explore_forward' actions until the distance is covered
        #         num_forward_steps = int(distance // explore_forward_m)
        #         for _ in range(num_forward_steps):
        #             behavior_tree_plan.append('explore_forward')
        #             current_x, current_y = move_forward(current_x, current_y, explore_forward_m, current_direction)
        #             x_coords_bt.append(current_x)
        #             y_coords_bt.append(current_y)

        #         # Calculate the direction to the next point
        #         next_direction = calculate_direction(current_x, current_y, next_x, next_y)
                
        #         # Calculate the angle difference between current direction and next direction
        #         direction_diff = angle_difference(current_direction, next_direction)
                
        #         # Add 'rotate_left' or 'rotate_right' actions based on the angle difference until it's aligned
        #         num_rotation_steps = int(abs(direction_diff) // rotation_degree)
        #         for _ in range(num_rotation_steps):
        #             if direction_diff < 0:
        #                 behavior_tree_plan.append('rotate_right')
        #                 current_direction -= rotation_degree
        #             else:
        #                 behavior_tree_plan.append('rotate_left')
        #                 current_direction += rotation_degree
                    
        #             current_direction = (current_direction + 180) % 360 - 180  # Normalize to [-180, 180] range

      
        # Print the behavior tree plan
        # st.write("This is the behavior tree plan generated by the analysis: ")
        # st.write(behavior_tree_plan)

        # Plot the analysis results in a graph
        st.write("Plotting the analysis results...")
        plt.figure(figsize=(10, 8))
        plt.plot(x_coords_original, y_coords_original, marker='o', linestyle='-', color='red', label='Original Trajectory')
        plt.plot(x_coords_smoothed, y_coords_smoothed, marker='o', linestyle='-', color='orange', label='Smoothed Trajectory (Analysis)')
        plt.plot(x_coords_simplified, y_coords_simplified, marker='o', linestyle='-', color='green', label='Simplified Trajectory (Analysis)')
        # plt.plot(x_coords_bt, y_coords_bt, marker='o', linestyle='-', color='blue', label='Behavior Tree Plan')
        plt.title("Trajectory Analysis")
        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.legend()
        st.pyplot(plt)

        # Save waypoints to a yaml file
        # format: [[x1, y1, z1, yaw1], [x2, y2, z2, yaw2], ...]
        waypoints = []
        for x, y, yaw in zip(x_coords_simplified, y_coords_simplified, yaw_simplified):
            waypoints.append([x, y, 0, yaw])
        st.write("The waypoints for the simplified trajectory are:")
        st.write(waypoints)
        
        import os
        st.write("Saving the analysis results to a yaml file...")
        # json file name should include the start lat and lng of trajectory, the timestamp of the trajectory, and the number of the subtrees
        start_latitude = df_filtered['Latitude'].iloc[0]
        start_longitude = df_filtered['Longitude'].iloc[0]
        start_timestamp = df_filtered['Timestamp'].iloc[0].date()
        num_waypoints = len(waypoints)
        yaml_file_name = f"traj_analysis_results_{start_latitude}_{start_longitude}_{start_timestamp}_num_points_{num_waypoints}.yaml"
        save_dir = "data/traj_analysis_results/"
        os.makedirs(save_dir, exist_ok=True)
        with open(save_dir + yaml_file_name, 'w') as f:
            f.write("waypoint: [\n")
            for waypoint in waypoints:
                f.write(f"  [{waypoint[0]}, {waypoint[1]}, {waypoint[2]}, {waypoint[3]}]\n")
            f.write("]\n")
        st.write(f"Analysis results saved to {save_dir + yaml_file_name}")

        