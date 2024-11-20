import pandas as pd
import streamlit as st
import resource_op as ro
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import create_model as cm
from streamlit_folium import folium_static


def load_models():
    """
    Load saved models for cost prediction and resource scheduling.
    """
    if os.path.exists("recent_cost_model.pkl") and os.path.exists("recent_resource_model.pkl"):
        cost_model = joblib.load("recent_cost_model.pkl")
        resource_model = joblib.load("recent_resource_model.pkl")
        return cost_model, resource_model
    else:
        cost_model, resource_model = cm.train_model()
        return cost_model, resource_model

def predict_with_models(cost_model, resource_model, test_values):
    """
    Use the loaded models to make predictions on test values.
    """
    # Extract features for prediction
    features = [
        'Store_Size_SQFT','Productivity_SQ_F_PerHour','Demand', 'Priority_Level'
    ]

    # Convert input values to a DataFrame
    test_data = pd.DataFrame([test_values], columns=features)

    # Predict cleaning cost
    predicted_cost = cost_model.predict(test_data)[0]

    # Predict resource type
    predicted_resource_encoded = resource_model.predict(test_data)[0]
    predicted_resource = {1: 'Mobile', 2: 'Mobile+Robots', 3: 'Fixed'}.get(predicted_resource_encoded, "Unknown")

    return predicted_cost, predicted_resource

def visualize_cost_savings_waterfall(comparison_results):
    # Create waterfall chart for cost comparison
    fig = go.Figure(go.Waterfall(
        name="Cost Comparison",
        orientation="v",
        measure=["absolute", "relative", "total"],
        x=["Fixed Cost", "Savings", "Mobile Cost"],
        y=[comparison_results['Fixed Cost'],
           -comparison_results['Cost Savings'],
           comparison_results['Mobile Cost']],
        text=["Fixed Cost", "Savings", "Mobile Cost"],
        decreasing={"marker": {"color": "green"}},
        increasing={"marker": {"color": "red"}},
        totals={"marker": {"color": "blue"}}
    ))

    fig.update_layout(
        title="Waterfall Chart of Cost Savings",
        xaxis_title="Cost Type",
        yaxis_title="Cost (in currency)",
        height=500,
        width=800
    )

    return fig


def show_basic_statistics():
    st.title("Coop Resource Optimisation")
    st.write("Analyze and filter Coop store data.")

    # Step 1: Load the data
    file_path = "recent_coop_dataset.csv"
    if file_path:
        df = ro.load_data(file_path)
        st.write("Data Loaded:")
        st.dataframe(df.head())

    analysis_options = [
        "Store Type Distribution",
        "Resource Type Distribution",
        "Crosstab of Store Type vs. Solution Type"
    ]
    selected_option = st.selectbox("Select Analysis Type:", analysis_options)

    # Show the selected analysis
    if selected_option == "Store Type Distribution":
        st.write("Store Type Distribution:")
        st.bar_chart(ro.store_type_distribution(df))

    elif selected_option == "Resource Type Distribution":
        st.write("Resource Type Distribution:")
        st.bar_chart(ro.resource_type_distribution(df))

    elif selected_option == "Crosstab of Store Type vs. Solution Type":
        crosstab = ro.store_resource_crosstab(df)
        st.write("Crosstab:")
        st.dataframe(crosstab)

    # Step 2: Show idle time analysis by store type
    idle_time_results = ro.calculate_idle_time_by_store_type(df)
    st.write("Idle Time Analysis by Store Type:")
    st.dataframe(idle_time_results)

    # Bar chart for total cleaning hours and idle time
    # Create a bar chart using Plotly
    fig = px.bar(
        idle_time_results,
        x='Store_Type',
        y=['Total Cleaning Hours', 'Idle Time'],
        title="Total Cleaning Hours vs Idle Time by Store Type",
        labels={'value': 'Hours', 'variable': 'Type'}
    )

    # Update the layout to increase the height
    fig.update_layout(
        height=500,  # Set the height of the chart
        width=800  # Optional: Set the width of the chart
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)

    # Pie chart for idle time percentage
    st.write("Idle Time Percentage by Store Type:")

    # Bar chart for total cleaning hours, idle time, and total available hours
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=idle_time_results['Store_Type'],
        y=idle_time_results['Total Cleaning Hours'],
        name='Total Cleaning Hours',
        marker_color='blue'
    ))
    fig.add_trace(go.Bar(
        x=idle_time_results['Store_Type'],
        y=idle_time_results['Idle Time'],
        name='Idle Time',
        marker_color='red'
    ))

    # Adding total available hours as a line for reference
    fig.add_trace(go.Scatter(
        x=idle_time_results['Store_Type'],
        y=idle_time_results['Total Available Hours'],
        mode='lines+markers',
        name='Total Available Hours',
        line=dict(color='green', width=2)
    ))

    fig.update_layout(
        title='Total Cleaning Hours vs Idle Time by Store Type',
        xaxis_title='Store_Type',
        yaxis_title='Hours',
        barmode='group'
    )

    st.plotly_chart(fig)

    st.write("Visualising store locations on map")
    if st.checkbox("Show Store Locations on Map"):
        ro.plot_store_locations(df)
        # Get clustering parameters
        eps_miles = st.slider("Set the distance threshold (miles) for clustering", min_value=0.1, max_value=50.0,
                              value=10.0, step=0.1)

        # Perform clustering
        labels = ro.perform_clustering(df, eps_miles)
        df['Cluster'] = labels  # Add cluster labels to the DataFrame

        # Plot clusters
        st.write(f"Clusters based on a {eps_miles} mile threshold:")
        ro.plot_clusters(df, labels)

    if st.button("Show Optimal Cleaning Path with Road Routes"):
        # Get the optimal path (for simplicity, let's assume Nearest Neighbor is used)
        optimal_path = ro.nearest_neighbor_tsp(df, start_index=0, num_stores=100)
        # Visualize the path using Folium
        road_route_map = ro.visualize_route_on_map(df, optimal_path)
        # Display the map in Streamlit
        folium_static(road_route_map)

    if st.checkbox("View map by Resource Type"):
        ro.map_by_type(df)

    st.write("Loaded models for prediction")

    # Load models
    cost_model, resource_model = load_models()

    if cost_model and resource_model:
        st.subheader("Provide Test Values for Prediction")

        # Input fields for test values
        store_size = st.number_input("Store Size (SQFT)", min_value=1000, max_value=10000, step=1)
        productivity = st.number_input("Productivity (SQ/F Per Hour)", min_value=float(100), max_value=float(4000),
                                       step=0.1)
        demand = st.slider("Demand", min_value=1, max_value=9, step=1)
        priority_level = st.selectbox("Priority Level", options=[1, 2, 3],
                                      format_func=lambda x: {1: "Low", 2: "Medium", 3: "High"}[x])
        # Get postcode from user
        postcode = st.text_input("Enter Postcode for Current Location")
        distance_threshold = st.number_input("Maximum Distance for Nearby Stores (in miles)", min_value=1,
                                             value=5,
                                             step=1)
        # Get number of stores to recommend from user
        max_stores = st.number_input("Number of Nearby Stores to Show", min_value=1, max_value=10, value=3, step=1)

        # Prepare test values
        test_values = [store_size, productivity, demand, priority_level]

        predicted_resource = "Fixed"

        # Predict and display results
        if st.button("Predict"):
            predicted_cost, predicted_resource = predict_with_models(cost_model, resource_model, test_values)

            st.subheader("Prediction Results")
            st.write(f"Predicted Cleaning Cost: £{predicted_cost:.2f}")
            st.write(f"Recommended Resource Type: {predicted_resource}")

            if(predicted_resource != "Fixed"):
                # Estimate the number of people and vans needed
                num_people, num_vans = ro.estimate_people_and_vans(store_size, productivity)

                # Display the estimates
                st.write("Estimated Resource Requirements:")
                st.write(f"Number of People Needed: {num_people}")
                st.write(f"Number of Vans Needed: {num_vans}")

            if postcode:
                current_lat, current_lon = ro.get_coordinates_from_postcode(postcode)
                st.write(current_lat)
                st.write(current_lon)
                if current_lat is not None and current_lon is not None:
                    input_data = pd.DataFrame({
                        'Store_Size_SQFT': [store_size],
                        'Productivity_SQ_F_PerHour': [productivity],
                        'Demand_Score': [demand],
                        'Priority_Level': [priority_level]
                    })

                nearby_stores = ro.get_nearby_stores(current_lat, current_lon, df, distance_threshold, max_stores)
                st.write("Recommended Nearby Stores:")
                st.dataframe(nearby_stores)

                if not nearby_stores.empty:
                    st.write("Map with Routes to Nearby Stores:")
                    m = ro.plot_map_with_routes(predicted_resource, current_lat, current_lon, nearby_stores)
                    folium_static(m)

if __name__ == "__main__":
    show_basic_statistics()