import folium as fl
from streamlit_folium import st_folium
import streamlit as st
import pandas as pd
from geopy.distance import geodesic
import matplotlib.pyplot as plt

# Load and split the CSV file
data = pd.read_csv('data/uav-csv/ALL.csv')

# Split the single string column into multiple columns
df = data.iloc[:, 0].str.split(';', expand=True)

# Rename the columns appropriately
df.columns = ['Trajectory_file_path', 'Trajectory_id', 'Longitude', 'Latitude', 'Total_distance', 'Timestamp']

# Convert latitude and longitude to float
df['Latitude'] = df['Latitude'].astype(float)
df['Longitude'] = df['Longitude'].astype(float)

# Select 1~100 trajectories
random_trajectory_ids = df['Trajectory_id'].sample(n=100, random_state=1)
random_trajectories = df[df['Trajectory_id'].isin(random_trajectory_ids)]
random_centers = random_trajectories.groupby('Trajectory_id')[['Latitude', 'Longitude']].mean()

# Create a Folium map centered around the mean latitude and longitude
center_lat, center_lng = random_centers.mean()
m = fl.Map(location=[center_lat, center_lng], zoom_start=10)
# m = fl.Map()

# Plot the random trajectories on the map
for idx, row in random_centers.iterrows():
    fl.Marker(location=[row['Latitude'], row['Longitude']]).add_to(m)

# Add LatLngPopup to capture user's click
m.add_child(fl.LatLngPopup())

# Display map in Streamlit
map = st_folium(m, height=350, width=700)

# Function to find closest trajectories
def find_closest_trajectories(lat, lng, top_n, df):
    selected_point = (lat, lng)
    df['distance'] = df.apply(lambda row: geodesic(selected_point, (row['Latitude'], row['Longitude'])).km, axis=1)
    return df.nsmallest(top_n, 'distance')

data = None
if map.get("last_clicked"):
    clicked_lat = map["last_clicked"]["lat"]
    clicked_lng = map["last_clicked"]["lng"]
    st.write(f"Selected point: ({clicked_lat}, {clicked_lng})")

    # Select number of closest trajectories
    top_n = st.slider("Select number of closest trajectories", min_value=1, max_value=20, value=5)

    # Find the closest trajectories
    closest_trajectories = find_closest_trajectories(clicked_lat, clicked_lng, top_n, df)
    st.write(f"Displaying the {top_n} closest trajectories")

    if not closest_trajectories.empty:
        # Convert Timestamp to datetime if not already done
        closest_trajectories['Timestamp'] = pd.to_datetime(closest_trajectories['Timestamp'])

        # Extract hour from the Timestamp and create 3-hour bins
        bins = [0, 3, 6, 9, 12, 15, 18, 21, 24]
        labels = ['00-03', '03-06', '06-09', '09-12', '12-15', '15-18', '18-21', '21-24']
        closest_trajectories['3hr_interval'] = pd.cut(closest_trajectories['Timestamp'].dt.hour, bins=bins, right=False, labels=labels)

        # Group by the 3-hour interval and count the occurrences
        interval_counts = closest_trajectories['3hr_interval'].value_counts().sort_index()

        # Display bar chart of trajectory counts per 3-hour interval
        # st.bar_chart(interval_counts)

        # Create a bar chart with proper labels
        fig, ax = plt.subplots()
        ax.bar(interval_counts.index, interval_counts.values)
        ax.set_xlabel('3-Hour Interval of the Day')
        ax.set_ylabel('Trajectory Count')
        ax.set_title('Trajectory Counts per 3-Hour Interval')

        # Display the chart in Streamlit
        st.pyplot(fig)