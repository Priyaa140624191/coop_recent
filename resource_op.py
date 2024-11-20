import pandas as pd
import streamlit as st
from streamlit_folium import folium_static
import folium
import numpy as np
from sklearn.cluster import DBSCAN
import time
import requests
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(file_path, dtype={"Karcher reference": str})
    return df


def store_type_distribution(df):
    """
    Calculate the distribution of different Store Types.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.Series: Distribution of Store Types.
    """
    return df['Store_Type'].value_counts()

def resource_type_distribution(df):
    """
    Calculate the distribution of different Solution Types.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.Series: Distribution of Solution Types.
    """
    return df['Resource_Type'].value_counts()

def store_resource_crosstab(df):
    """
    Create a crosstab of Store Type and Solution Type.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: Crosstab of Store Type vs. Solution Type.
    """
    return pd.crosstab(df['Store_Type'], df['Resource_Type'])

def calculate_idle_time_by_store_type(df, work_hours_per_day=8):
    """
    Calculate total cleaning hours and idle time for each store type,
    based on the given productivity and store size values.

    Parameters:
        df (pd.DataFrame): The input DataFrame with store information.
        work_hours_per_day (int): Number of work hours per day. Default is 8 hours.

    Returns:
        pd.DataFrame: A DataFrame with total cleaning hours, idle time, and idle time percentage for each store type.
    """
    # Step 1: Calculate cleaning time for each store
    df['Cleaning Hours Per Store'] = df['Store_Size_SQFT'] / df['Productivity_SQ_F_PerHour']

    st.write(df)

    # Step 2: Group by 'Store Type' to get total cleaning hours for each store type
    cleaning_hours_by_store_type = df.groupby('Store_Type')['Cleaning Hours Per Store'].sum().reset_index()
    cleaning_hours_by_store_type.columns = ['Store_Type', 'Total Cleaning Hours']

    st.write(cleaning_hours_by_store_type)

    # Step 3: Calculate the number of stores for each store type
    store_counts = df.groupby('Store_Type').size().reset_index(name='Number of Stores')

    # Step 4: Merge the store counts with the total cleaning hours
    cleaning_hours_by_store_type = cleaning_hours_by_store_type.merge(store_counts, on='Store_Type')

    # Step 5: Calculate total available hours (assuming 8 hours per day per store)
    cleaning_hours_by_store_type['Total Available Hours'] = cleaning_hours_by_store_type[
                                                                'Number of Stores'] * work_hours_per_day

    # Convert 'Total Available Hours' and 'Total Cleaning Hours' to numeric
    cleaning_hours_by_store_type['Total Available Hours'] = pd.to_numeric(
        cleaning_hours_by_store_type['Total Available Hours'], errors='coerce'
    )
    cleaning_hours_by_store_type['Total Cleaning Hours'] = pd.to_numeric(
        cleaning_hours_by_store_type['Total Cleaning Hours'], errors='coerce'
    )

    # Step 6: Calculate idle time (Total Available Hours - Total Cleaning Hours)
    cleaning_hours_by_store_type['Idle Time'] = cleaning_hours_by_store_type['Total Available Hours'] - \
                                                cleaning_hours_by_store_type['Total Cleaning Hours']

    # Step 7: Calculate idle time percentage
    cleaning_hours_by_store_type['Idle Time Percentage'] = (cleaning_hours_by_store_type['Idle Time'] /
                                                            cleaning_hours_by_store_type['Total Available Hours']) * 100

    return cleaning_hours_by_store_type

def plot_store_locations(df):
    """
    Plot the store locations on a map using latitude and longitude.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        None: Displays the map in Streamlit.
    """
    # Create a map centered at the average latitude and longitude
    map_center = [df['latitude'].mean(), df['longitude'].mean()]
    store_map = folium.Map(location=map_center, zoom_start=6)

    # Add small red circle markers to the map
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,  # Size of the circle
            color='red',  # Border color
            fill=True,
            fill_color='red',  # Fill color
            fill_opacity=0.6
        ).add_to(store_map)

    # Display the map in Streamlit
    folium_static(store_map)

def perform_clustering(df, eps_miles):
    """
    Perform DBSCAN clustering on the store locations based on distance.

    Parameters:
        df (pd.DataFrame): The input DataFrame with 'Latitude' and 'Longitude' columns.
        eps_miles (float): The maximum distance (in miles) for two points to be considered in the same neighborhood.

    Returns:
        np.ndarray: Cluster labels for each point.
    """
    # Convert miles to kilometers (1 mile = 1.60934 kilometers)
    eps_km = eps_miles * 1.60934

    # DBSCAN requires distance in radians for geographic data
    earth_radius_km = 6371.0
    eps_rad = eps_km / earth_radius_km

    # Convert latitude and longitude to radians
    coords = np.radians(df[['latitude', 'longitude']])

    # Perform DBSCAN clustering
    db = DBSCAN(eps=eps_rad, min_samples=2, metric='haversine').fit(coords)
    labels = db.labels_

    return labels


def plot_clusters(df, labels):
    """
    Plot clusters on a map using folium.

    Parameters:
        df (pd.DataFrame): The input DataFrame with 'Latitude' and 'Longitude' columns.
        labels (np.ndarray): Cluster labels for each point.

    Returns:
        None: Displays the map in Streamlit.
    """
    # Create a map centered at the average latitude and longitude
    map_center = [df['latitude'].mean(), df['longitude'].mean()]
    store_map = folium.Map(location=map_center, zoom_start=6)

    # Generate color map for clusters
    unique_labels = set(labels)
    colors = [
        f"#{hex(np.random.randint(0, 256))[2:]}{hex(np.random.randint(0, 256))[2:]}{hex(np.random.randint(0, 256))[2:]}"
        for _ in range(len(unique_labels))]

    # Plot each point with a color based on its cluster label
    for idx, row in df.iterrows():
        label = labels[idx]
        color = colors[label] if label != -1 else 'black'  # Black for noise points
        store_ref = row.get('Karcher reference', 'Unknown')
        postcode = row.get('Postcode', 'Unknown')
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=f"Cluster {label}, Store Ref: {store_ref}, Postcode: {postcode}" if label != -1 else f"Noise, Store Ref: {store_ref}, Postcode: {postcode}"
        ).add_to(store_map)

    # Display the map in Streamlit
    folium_static(store_map)

def estimate_people_and_vans(store_size, productivity, shift_duration=8, van_capacity=4):
    """
    Estimate the number of people and vans needed for servicing a store.

    Parameters:
    - store_size (float): Size of the store in square feet.
    - productivity (float): Productivity in square feet per hour per person.
    - shift_duration (int, optional): Duration of a single shift in hours (default is 8).
    - van_capacity (int, optional): Capacity of each van in terms of number of people (default is 4).

    Returns:
    - num_people (int): Estimated number of people required.
    - num_vans (int): Estimated number of vans required.
    """
    # Calculate total hours needed to clean the store
    total_hours_needed = store_size / productivity

    # Calculate number of people required, based on shift duration
    num_people = int(np.ceil(total_hours_needed / shift_duration))

    # Calculate number of vans required, based on van capacity
    num_vans = int(np.ceil(num_people / van_capacity))

    return num_people, num_vans


def map_by_type(df):
    # Streamlit App
    st.write("Resource Type Map Visualization")

    # Check for required columns
    required_columns = ['Resource_Type', 'latitude', 'longitude', 'Karcher reference']
    if not all(col in df.columns for col in required_columns):
        st.error(f"Dataset must contain the following columns: {', '.join(required_columns)}")
        return

    # Dropdown for Resource_Type selection
    resource_type = st.selectbox("Select Resource Type:", options=df['Resource_Type'].unique())

    # Filter dataset based on selection
    filtered_data = df[df['Resource_Type'] == resource_type]

    # Handle case where no data matches the selected resource type
    if filtered_data.empty:
        st.warning(f"No data available for the selected Resource Type: {resource_type}")
        return

    # Create a Folium map centered at the mean latitude and longitude of filtered data
    mean_lat = filtered_data['latitude'].mean()
    mean_lon = filtered_data['longitude'].mean()
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=8)

    # Add points to the map with custom icons
    for _, row in filtered_data.iterrows():
        # Determine the icon based on Resource_Type
        if row['Resource_Type'] == "Mobile":
            icon = folium.Icon(icon="car", prefix="fa", color="blue")  # Car icon for Mobile
        elif row['Resource_Type'] == "Mobile+Robots":
            icon = folium.Icon(icon="robot", prefix="fa", color="red")  # Robot icon for Mobile+Robots
        elif row['Resource_Type'] == "Fixed":
            icon = folium.Icon(icon="building", prefix="fa", color="green")  # Building icon for Fixed
        else:
            icon = folium.Icon(icon="info-sign", color="gray")  # Default icon for other resource types

        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Store: {row['Karcher reference']}<br>Resource Type: {row['Resource_Type']}",
            tooltip=f"Store: {row['Karcher reference']}",
            icon=icon
        ).add_to(m)

    # Display the map
    st.write(f"Displaying points for Resource Type: {resource_type}")
    folium_static(m, width=700, height=500)

def nearest_neighbor_tsp(df, start_index=0, num_stores=100):
    """
    Approximate TSP solution using the Nearest Neighbor algorithm for a given number of stores.
    """
    # Limit to specified number of stores (default to 100)
    subset_df = df.iloc[:num_stores].reset_index(drop=True)

    # Initialize
    unvisited = set(range(len(subset_df)))
    path = [start_index]
    unvisited.remove(start_index)
    current_index = start_index

    while unvisited:
        # Find the nearest neighbor
        nearest_index = min(unvisited, key=lambda i: np.hypot(
            subset_df.iloc[i]['latitude'] - subset_df.iloc[current_index]['latitude'],
            subset_df.iloc[i]['longitude'] - subset_df.iloc[current_index]['longitude']
        ))
        # Update path
        path.append(nearest_index)
        unvisited.remove(nearest_index)
        current_index = nearest_index

    return path

def get_route_via_osrm(lat1, lon1, lat2, lon2, retries=3):
    """
    Get the driving route from point A (lat1, lon1) to point B (lat2, lon2) using OSRM, with retry logic.
    """
    url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
    for attempt in range(retries):
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200 and 'routes' in data and data['routes']:
            route_geometry = data['routes'][0]['geometry']['coordinates']
            return route_geometry
        else:
            print(f"Error fetching route from {lat1},{lon1} to {lat2},{lon2}, attempt {attempt + 1}")
            time.sleep(1)  # Delay before retrying

    return None  # Return None if all retries fail


def visualize_route_on_map(df, optimal_path):
    # Create a map centered at the average latitude and longitude
    map_center = [df['latitude'].mean(), df['longitude'].mean()]
    store_map = folium.Map(location=map_center, zoom_start=6)

    # Plot the stores as markers
    for index in optimal_path:
        row = df.iloc[index]
        store_ref = row.get('Karcher reference', 'Unknown')
        postcode = row.get('Postcode', 'Unknown')
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Store {index}, Store Ref: {store_ref}, Postcode: {postcode}",
            icon=folium.Icon(color='green', icon='fa-shopping-basket', prefix='fa')
        ).add_to(store_map)

    # Draw the road routes between stores
    for i in range(len(optimal_path) - 1):
        start_index = optimal_path[i]
        end_index = optimal_path[i + 1]
        start_lat, start_lon = df.iloc[start_index]['latitude'], df.iloc[start_index]['longitude']
        end_lat, end_lon = df.iloc[end_index]['latitude'], df.iloc[end_index]['longitude']

        # Get the route via OSRM
        route = get_route_via_osrm(start_lat, start_lon, end_lat, end_lon)

        if route:
            folium.PolyLine(
                locations=[(lat, lon) for lon, lat in route],  # Flip to (lat, lon)
                color='red',
                weight=5,
                opacity=0.7
            ).add_to(store_map)

    return store_map

def get_coordinates_from_postcode(postcode):
    """
    Converts a postcode to latitude and longitude.

    Parameters:
    - postcode (str): The postcode to convert.

    Returns:
    - (float, float): Latitude and longitude if found, otherwise (None, None).
    """
    geolocator = Nominatim(user_agent="store_locator")
    location = geolocator.geocode(postcode)
    if location:
        return location.latitude, location.longitude
    else:
        st.error("Unable to find coordinates for the given postcode.")
        return None, None

def get_nearby_stores(current_lat, current_lon, stores_df, distance_threshold, max_stores):
    """
    Recommend a list of nearby stores within a given distance threshold.
    Excludes stores that are exactly 0 miles from the current location.
    """
    distances = []
    for _, row in stores_df.iterrows():
        store_lat = row['latitude']
        store_lon = row['longitude']
        distance = geodesic((current_lat, current_lon), (store_lat, store_lon)).miles
        distances.append(distance)

    stores_df['Distance'] = distances
    # Filter stores greater than 0 miles away and within the distance threshold
    nearby_stores = stores_df[(stores_df['Distance'] > 0) & (stores_df['Distance'] <= distance_threshold)]
    nearby_stores = nearby_stores.sort_values(by='Distance').head(max_stores)

    return nearby_stores[['Karcher reference', 'latitude', 'longitude', 'Distance']]

def get_osrm_route(start_lat, start_lon, end_lat, end_lon):
    """
    Get road route between two points using OSRM.

    Parameters:
    - start_lat (float): Latitude of the start location.
    - start_lon (float): Longitude of the start location.
    - end_lat (float): Latitude of the end location.
    - end_lon (float): Longitude of the end location.

    Returns:
    - list of (lat, lon) tuples representing the route.
    """
    osrm_url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson"
    response = requests.get(osrm_url)

    if response.status_code == 200:
        data = response.json()
        route = data['routes'][0]['geometry']['coordinates']
        # OSRM returns coordinates as (lon, lat), so we need to reverse them
        route = [(lat, lon) for lon, lat in route]
        return route
    else:
        st.error("Error fetching route from OSRM.")
        return []

def plot_map_with_routes(predicted_resource, current_lat, current_lon, nearby_stores):
    """
    Plots a map with road routes from the current location to each nearby store.

    Parameters:
    - current_lat (float): Latitude of the current location.
    - current_lon (float): Longitude of the current location.
    - nearby_stores (DataFrame): DataFrame of nearby stores with 'Latitude', 'Longitude', and 'Store Id'.

    Returns:
    - folium.Map: Folium map object with markers and routes.
    """
    st.write("Predicted Resource ", predicted_resource)
    m = folium.Map(location=[current_lat, current_lon], zoom_start=12)
    folium.Marker(
        [current_lat, current_lon],
        popup="Current Location",
        icon=folium.Icon(color='darkred', icon='shopping-basket', prefix='fa')
    ).add_to(m)

    for _, row in nearby_stores.iterrows():
        store_lat = row['latitude']
        store_lon = row['longitude']
        distance = row['Distance']

        # Fetch the actual road route from OSRM
        route = get_osrm_route(current_lat, current_lon, store_lat, store_lon)

        # Plot the route on the map if found
        if route:
            folium.PolyLine(
                locations=route,
                color="blue",
                weight=2.5,
                opacity=0.8
            ).add_to(m)

        # Determine icon properties based on predicted_resource and Resource_Type
        if predicted_resource == "Mobile":
            icon = folium.Icon(icon="car", prefix="fa", color="purple")  # Blue car icon
        elif predicted_resource == "Mobile+Robots":
            icon = folium.Icon(icon="robot", prefix="fa", color="cadetblue")  # Red robot icon for Mobile+Robots
        else:
            icon = folium.Icon(color='green', icon='building',
                               prefix='fa')  # Default green shopping basket icon

        # Add a marker for each nearby store
        folium.Marker(
            [store_lat, store_lon],
            popup=f"Store ID: {row['Karcher reference']}, Distance: {distance:.2f} miles",
            icon=icon
        ).add_to(m)

    return m