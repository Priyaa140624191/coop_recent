import pandas as pd
import streamlit as st
import os
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
import plotly.graph_objects as go

def train_model():
    # Step 1: Load the data
    file_path = "recent_coop_dataset.csv"
    if file_path:
        df = pd.read_csv(file_path, dtype={"Karcher reference": str})

        # Encode 'Store_Type' for model compatibility
        df['Store_Type'] = df['Store_Type'].map({'Convenience Store': 1, 'Supermarket': 2, 'Superstore': 3})

        # Encode 'Solution_Type' for model compatibility
        df['Solution_Type'] = df['Solution_Type'].map({
            'Mobile Solution': 1,
            'Static Solution': 2,
            'Static Solution with Robotics': 3
        })

        # Clean 'Cost_Per_Hour_Labour_Only' column
        df['Cost_Per_Hour_Labour_Only'] = df['Cost_Per_Hour_Labour_Only'].str.replace('£', '', regex=False).astype(
            float)

        # Introduce controlled random noise
        np.random.seed(42)  # Ensure reproducibility
        noise = np.random.normal(5, 15, size=len(df))  # Mean = 0, Std Dev = 10

        # Adjust Total_Estimated_Cost to include slight variation
        df['Total_Estimated_Cost'] = (
                                             df['Estimated_Service_Time_Hours'] * df['Cost_Per_Hour_Labour_Only']
                                     ) + noise

        # # Define cost prediction target
        # df['Total_Estimated_Cost'] = df['Estimated_Service_Time_Hours'] * df['Cost_Per_Hour_Labour_Only']

        # # Encode Resource_Type for classification (dummy encoding example for now)
        # df['Resource_Type_Encoded'] = df['Solution_Type'] - 1  # Temporary mapping

        # Encode 'Priority_Level' for model compatibility
        df['Priority_Level'] = df['Priority_Level'].map({'Low': 1, 'Medium': 2, 'High': 3})

        df['Resource_Type_Encoded'] = df['Resource_Type'].map({'Mobile': 1, 'Mobile+Robots': 2, 'Fixed': 3})

        # Define features for both models
        features = ['Store_Size_SQFT', 'Productivity_SQ_F_PerHour', 'Demand', 'Priority_Level']

        # Cost Prediction Target (Regression)
        y_cost = df['Total_Estimated_Cost']

        # Resource Scheduling Target (Classification)
        y_resource = df['Resource_Type_Encoded']

        # Split data for cost prediction
        X_train_cost, X_test_cost, y_train_cost, y_test_cost = train_test_split(
            df[features], y_cost, test_size=0.2, random_state=42
        )

        # Split data for resource scheduling
        X_train_resource, X_test_resource, y_train_resource, y_test_resource = train_test_split(
            df[features], y_resource, test_size=0.2, random_state=42
        )

        # Check if saved models exist
        if os.path.exists("recent_cost_model.pkl") and os.path.exists("recent_resource_model.pkl"):
            # Load the saved models
            cost_model = joblib.load("recent_cost_model.pkl")
            resource_model = joblib.load("recent_resource_model.pkl")
        else:
            # Train models if not already saved
            cost_model = RandomForestRegressor(random_state=42)
            cost_model.fit(X_train_cost, y_train_cost)
            joblib.dump(cost_model, "recent_cost_model.pkl")  # Save cost model

            resource_model = RandomForestClassifier(random_state=42)
            resource_model.fit(X_train_resource, y_train_resource)
            joblib.dump(resource_model, "recent_resource_model.pkl")  # Save resource model

            st.write("Trained and saved models.")

        # Predict cost on test data
        predicted_cost = cost_model.predict(X_test_cost)

        # Predict resource type on test data
        predicted_resource = resource_model.predict(X_test_resource)

        # Evaluate cost prediction model
        mse_cost = mean_squared_error(y_test_cost, predicted_cost)
        rmse_cost = np.sqrt(mse_cost)
        st.write(f"Root Mean Squared Error (RMSE) for Cost Prediction: £{rmse_cost:.2f}")

        # Evaluate resource scheduling model
        accuracy_resource = accuracy_score(y_test_resource, predicted_resource)
        st.write(f"Accuracy for Resource Scheduling Prediction: {accuracy_resource:.2%}")

        # Sample of predictions for verification
        sample_predictions = pd.DataFrame({
            "Actual Cost": y_test_cost.reset_index(drop=True)[:10],
            "Predicted Cost": predicted_cost[:10],
            "Actual Resource Type": y_test_resource.reset_index(drop=True)[:10].map(
                {1: 'Mobile', 2: 'Mobile+Robots', 3: 'Fixed'}),
            "Predicted Resource Type": pd.Series(predicted_resource[:10]).map(
                {1: 'Mobile', 2: 'Mobile+Robots', 3: 'Fixed'})
        })

        st.write("Sample Predictions for Verification:")
        st.dataframe(sample_predictions)

        # Create Plotly line chart
        fig = go.Figure()

        # Add Actual Cost line
        fig.add_trace(go.Scatter(x=sample_predictions.index, y=sample_predictions['Actual Cost'],
                                 mode='lines+markers', name='Actual Cost'))

        # Add Predicted Cost line
        fig.add_trace(go.Scatter(x=sample_predictions.index, y=sample_predictions['Predicted Cost'],
                                 mode='lines+markers', name='Predicted Cost'))

        # Update layout
        fig.update_layout(title="Actual vs Predicted Cost",
                          xaxis_title="Test Sample Index",
                          yaxis_title="Cost",
                          template="plotly_white")

        # Display Plotly chart with a unique key
        st.plotly_chart(fig, key="cost_prediction_chart")

        return cost_model, resource_model

