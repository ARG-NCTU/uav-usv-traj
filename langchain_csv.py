import pandas as pd
import geopandas as gpd
import movingpandas as mpd
import math
import folium as fl
from streamlit_folium import st_folium
import streamlit as st
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from langchain_experimental.agents import create_csv_agent
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
from io import StringIO
import re

def main():
    load_dotenv()

    # Check if the API key is set in the environment
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("The OPENAI_API_KEY environment variable is not set.")

    st.set_page_config(page_title="Ask your CSV")
    st.header("Trajectory Extractor & Analysis Tool")
    st.subheader("Trajectory Extractor with LLM")

    model_options = [
        "gpt-4",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-3.5-turbo"
    ]
    
    selected_model = st.selectbox("Select a model", model_options)

    csv_data = pd.read_csv("data/uav-csv/ALL.csv")
    csv_string = csv_data.to_csv(index=False)
    csv_file_like = StringIO(csv_string)

    # Read the prompt from the "prompt.txt" file
    prompt_text = ""
    try:
        with open("prompt.txt", "r") as prompt_file:
            prompt_text = prompt_file.read()
    except FileNotFoundError:
        st.error("The 'prompt.txt' file was not found. Please make sure it is in the correct location.")

    user_question = st.text_input("Ask a question about trajectories dataset: ")
    output_format_req = "Please provide whole completed information about the trajectory. Do not miss any information of trajectory file path."
    if user_question and "llm_response" not in st.session_state:
        with st.spinner(text="In progress..."):
            # Determine which class to use based on the selected model
            if selected_model == "gpt-3.5-turbo-instruct":
                llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"], model_name=selected_model, temperature=0)
            else:
                llm = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"], model=selected_model, temperature=0)
            
            agent = create_csv_agent(
                llm,
                csv_file_like,
                verbose=True,
                allow_dangerous_code=True,
                handle_parsing_errors=True
            )

            # Combine the prompt text with the user question
            combined_question = f"{prompt_text}{user_question}\n\n{output_format_req}"
            response = agent.run(combined_question)
            # Store the LLM response in session state
            st.session_state.llm_response = response

    if user_question and "llm_response" in st.session_state:    
        # Extract CSV file path from the response
        st.success(st.session_state.llm_response)
        csv_file_path = extract_csv_filepath(st.session_state.llm_response)
        st.subheader("Trajectory Analysis To Generate Custom Exploration Pattern Block Behavior Tree Plan")
        if csv_file_path:
            st.success(f"Analyze your requested CSV file: {csv_file_path}")
            data = pd.read_csv(csv_file_path)
            df = pd.DataFrame(data)
            df_split = df['Longitude;Latitude;ID;Trajectory_id;Timestamp'].str.split(';', expand=True)
            df_split.columns = ['Longitude', 'Latitude', 'ID', 'Trajectory_id', 'Timestamp']

            # Convert the necessary columns to appropriate data types
            df_split['Longitude'] = df_split['Longitude'].astype(float)
            df_split['Latitude'] = df_split['Latitude'].astype(float)
            df_split['Timestamp'] = pd.to_datetime(df_split['Timestamp']).dt.tz_localize(None)

            # Streamlit slider to select BT start and end points based on latitude and Longitude
            start_index = st.slider('Select the start point of the BT:', 0, len(df_split)-1, 1, key=f'start_index')
            end_index = st.slider('Select the end point of the BT:', 0, len(df_split)-1, len(df_split)-2, key=f'end_index')

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
            process_noise_std = st.slider('Set noise standard deviation for Kalman Smoother:', 0.0, 5.0, 2.0, key=f'process_noise_std')
            measurement_noise_std = st.slider('Set measurement noise standard deviation for Kalman Smoother:', 0.1, 10.0, 5.0, key=f'measurement_noise_std')
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
            direction_for_simplify = st.slider('Set direction (degree) threshold for simplifying trajectory:', 1, 180, 10, key=f'direction_for_simplify')
            distance_for_simplify = st.slider('Set distance (meter) threshold for simplifying trajectory:', 0.1, 10.0, 1.0, key=f'distance_for_simplify')

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
                
                previous_point = point.geometry

            # Ensure the last point is included in the simplified trajectory
            if (x_coords_simplified[-1], y_coords_simplified[-1]) != (x_coords_smoothed[-1], y_coords_smoothed[-1]):
                x_coords_simplified.append(x_coords_smoothed[-1])
                y_coords_simplified.append(y_coords_smoothed[-1])
                yaw_simplified.append(current_direction)

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


        else:
            st.warning("No CSV file path found in the response.")

def extract_csv_filepath(response_text):
    # Use regex to find any CSV filename in the response text
    csv_file_path = re.search(r"([A-Za-z]:\\|\.\/|\/)?([\w\s-]+[\\/])*[\w\s-]+\.csv", response_text)
    if csv_file_path:
        return csv_file_path.group(0)
    return None

if __name__ == "__main__":
    main()
