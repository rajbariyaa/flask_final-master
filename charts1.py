import streamlit as st
import pandas as pd
import numpy as np
import datetime
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

warnings.filterwarnings('ignore')

# ---------- Data Loading ----------
@st.cache_data
def load_and_prepare_data():
    try:
        df = pd.read_csv("full_data.csv")
        df = df.dropna(subset=['ARRIVAL_DELAY'])
        df['FLIGHT_DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']], errors='coerce')  # New
        return df
    except FileNotFoundError:
        st.error("CSV file not found.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ---------- Preprocessing ----------
def preprocess_features(df):
    time_cols = ['SCHEDULED_DEPARTURE', 'SCHEDULED_ARRIVAL']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    y = df['ARRIVAL_DELAY'].fillna(0)

    exclude = [
        'DEPARTURE_DELAY', 'ARRIVAL_DELAY', 'ARRIVAL_TIME',
        'FLIGHT_DATE'  # 👈 Exclude datetime column
    ]
    features = [col for col in df.columns if col not in exclude]
    X = df[features].fillna(0)

    categorical_cols = X.select_dtypes(include=['object']).columns
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    return X, y, encoders, features


# ---------- Model Training ----------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train.columns

# ---------- Prediction Prep ----------
def create_prediction_input(inputs, df, encoders, feature_columns):
    date_obj = inputs['date']

    route_data = df[
        (df['ORIGIN_AIRPORT'].str.upper() == inputs['origin']) &
        (df['DESTINATION_AIRPORT'].str.upper() == inputs['dest']) &
        (df['AIRLINE'].str.upper() == inputs['airline']) &
        (df['MONTH'] == date_obj.month)
        ]

    if route_data.empty:
        route_data = df[
            (df['AIRLINE'].str.upper() == inputs['airline']) &
            (df['MONTH'] == date_obj.month)
            ]

    input_dict = {
        'YEAR': date_obj.year,
        'MONTH': date_obj.month,
        'DAY': date_obj.day,
        'DAY_OF_WEEK': date_obj.weekday() + 1,
        'SCHEDULED_DEPARTURE': inputs['scheduled_departure'],
        'SCHEDULED_ARRIVAL': inputs['scheduled_arrival'],
        'ORIGIN_AIRPORT': inputs['origin'],
        'DESTINATION_AIRPORT': inputs['dest'],
        'AIRLINE': inputs['airline']
    }

    numeric_cols = route_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in feature_columns and col not in input_dict:
            avg_val = route_data[col].mean()
            input_dict[col] = avg_val if not pd.isna(avg_val) else 0

    input_df = pd.DataFrame([input_dict])

    for col, encoder in encoders.items():
        if col in input_df.columns:
            value = str(input_df[col].iloc[0])
            if value in encoder.classes_:
                input_df[col] = encoder.transform([value])[0]
            else:
                input_df[col] = encoder.transform([encoder.classes_[0]])[0]

    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    return input_df

# ---------- Weather-Based Delay Reasoning ----------
def get_weather_delay_reason(df, origin, dest, date):
    subset = df[
        (df['ORIGIN_AIRPORT'].str.upper() == origin.upper()) &
        (df['DESTINATION_AIRPORT'].str.upper() == dest.upper()) &
        (df['MONTH'] == date.month)
        ]

    if subset.empty:
        return "No weather data available for this route and month."

    origin_weather = subset['origin_weather_desc'].dropna().astype(str) if 'origin_weather_desc' in subset else pd.Series(dtype=str)
    dest_weather = subset['dest_weather_desc'].dropna().astype(str) if 'dest_weather_desc' in subset else pd.Series(dtype=str)

    weather_data = pd.concat([origin_weather, dest_weather], ignore_index=True)

    if weather_data.empty:
        return "Weather data is missing or insufficient for analysis."

    weather_keywords = {
        'rain': '🌧️ Rain',
        'snow': '❄️ Snow',
        'storm': '⛈️ Storms',
        'fog': '🌫️ Fog or Mist',
        'mist': '🌫️ Fog or Mist'
    }

    total_entries = len(weather_data)
    keyword_counts = {key: 0 for key in weather_keywords}

    for desc in weather_data:
        lowered = desc.lower()
        for key in keyword_counts:
            if key in lowered:
                keyword_counts[key] += 1

    fog_mist_count = keyword_counts['fog'] + keyword_counts['mist']
    weather_summary = []

    for key, label in weather_keywords.items():
        if key == 'mist':
            continue
        count = fog_mist_count if key == 'fog' else keyword_counts[key]
        percent = (count / total_entries) * 100 if total_entries else 0
        if count > 0:
            weather_summary.append(f"{label} affected {percent:.1f}% of flights.")

    if not weather_summary:
        weather_summary.append("☀️ Clear weather dominated during this period.")

    return "\n".join(weather_summary)

# ---------- Airport Weather Summary ----------
def get_airport_weather_summary(df, airport_code, month, prefix):
    airport_cols_map = {
        "origin": "ORIGIN_AIRPORT",
        "dest": "DESTINATION_AIRPORT"
    }
    ap_col = airport_cols_map.get(prefix, f"{prefix.upper()}_AIRPORT")

    weather_desc_col = f'{prefix}_weather_desc'
    temp_col = f'{prefix}_temperature'
    wind_col = f'{prefix}_wind_speed'

    try:
        subset = df[(df[ap_col].str.upper() == airport_code.upper()) & (df['MONTH'] == month)]
    except KeyError:
        return "⚠️ Weather data unavailable."

    if subset.empty:
        return "⚠️ Weather data unavailable."

    descs = subset[weather_desc_col].dropna().unique()
    desc_text = ", ".join(descs[:3]) + ("..." if len(descs) > 3 else "")

    temp_c = subset[temp_col].dropna().mean()
    temp_f = (temp_c * 9 / 5 + 32) if pd.notna(temp_c) else None
    wind = subset[wind_col].dropna().mean()

    temp_str = f"{temp_c:.1f}°C / {temp_f:.1f}°F" if pd.notna(temp_c) else "N/A"
    wind_str = f"{wind:.1f} m/s" if pd.notna(wind) else "N/A"

    return f"Conditions: {desc_text}\nAvg Temp: {temp_str}\nAvg Wind: {wind_str}"

# ---------- Generate HTML Charts ----------
def generate_html_charts(df, user_inputs, prediction):
    """Generate all charts and save them to an HTML file"""
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Top 10 Airports with Highest Average Arrival Delays",
            "Monthly Delay Trends", 
            "Delay Causes - Focus on Weather",
            "Airline Performance Comparison",
            "Daily Delay Patterns",
            "Route-Specific Analysis"
        ),
        specs=[
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "pie"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "bar"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Chart 1: Top 10 Airports with Highest Average Arrival Delays
    delay_by_airport = df.groupby("ORIGIN_AIRPORT")["ARRIVAL_DELAY"].mean().sort_values(ascending=False).head(10)
    fig.add_trace(
        go.Bar(
            x=delay_by_airport.index,
            y=delay_by_airport.values,
            name="Avg Delay (min)",
            marker_color='lightcoral'
        ),
        row=1, col=1
    )
    
    # Chart 2: Monthly Delay Trends
    if 'FLIGHT_DATE' in df.columns:
        monthly_delay = df.dropna(subset=["FLIGHT_DATE"]).groupby(df["FLIGHT_DATE"].dt.to_period("M"))["ARRIVAL_DELAY"].mean()
        monthly_delay.index = monthly_delay.index.to_timestamp()
        fig.add_trace(
            go.Scatter(
                x=monthly_delay.index,
                y=monthly_delay.values,
                mode='lines+markers',
                name="Monthly Avg Delay",
                line=dict(color='steelblue', width=3)
            ),
            row=1, col=2
        )
    
    # Chart 3: Delay Causes - Focus on Weather
    delay_cause_cols = {
        "Air System": "AIR_SYSTEM_DELAY",
        "Security": "SECURITY_DELAY", 
        "Airline": "AIRLINE_DELAY",
        "Late Aircraft": "LATE_AIRCRAFT_DELAY",
        "Weather": "WEATHER_DELAY"
    }
    
    cause_sums = {}
    for label, col in delay_cause_cols.items():
        if col in df.columns:
            cause_sums[label] = df[col].sum()
    
    if cause_sums:
        labels = list(cause_sums.keys())
        values = list(cause_sums.values())
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink', 'lightcoral']
        
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                name="Delay Causes",
                marker=dict(colors=colors)
            ),
            row=2, col=1
        )
    
    # Chart 4: Airline Performance Comparison
    airline_delay = df.groupby("AIRLINE")["ARRIVAL_DELAY"].mean().sort_values(ascending=False).head(10)
    fig.add_trace(
        go.Bar(
            x=airline_delay.index,
            y=airline_delay.values,
            name="Airline Avg Delay",
            marker_color='lightsteelblue'
        ),
        row=2, col=2
    )
    
    # Chart 5: Daily Delay Patterns
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    daily_delay = df.groupby("DAY_OF_WEEK")["ARRIVAL_DELAY"].mean()
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    fig.add_trace(
        go.Scatter(
            x=day_names,
            y=daily_delay.values,
            mode='lines+markers',
            name="Daily Pattern",
            line=dict(color='orange', width=3)
        ),
        row=3, col=1
    )
    
    # Chart 6: Route-Specific Analysis (if user made a prediction)
    if user_inputs:
        route_data = df[
            (df['ORIGIN_AIRPORT'].str.upper() == user_inputs['origin']) &
            (df['DESTINATION_AIRPORT'].str.upper() == user_inputs['dest'])
        ]
        if not route_data.empty:
            monthly_route_delay = route_data.groupby("MONTH")["ARRIVAL_DELAY"].mean()
            fig.add_trace(
                go.Bar(
                    x=monthly_route_delay.index,
                    y=monthly_route_delay.values,
                    name=f"{user_inputs['origin']}-{user_inputs['dest']} Route",
                    marker_color='mediumseagreen'
                ),
                row=3, col=2
            )
    
    # Update layout
    fig.update_layout(
        height=1200,
        showlegend=True,
        title_text="Flight Delay Analytics Dashboard",
        title_x=0.5,
        title_font_size=24
    )
    
    # Add prediction result as annotation if available
    if user_inputs and prediction is not None:
        prediction_text = f"Prediction: {prediction:.1f} min {'delay' if prediction > 0 else 'early'}"
        fig.add_annotation(
            text=prediction_text,
            xref="paper", yref="paper",
            x=0.5, y=0.98,
            showarrow=False,
            font=dict(size=16, color="red" if prediction > 0 else "green"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Flight Delay Analytics Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                text-align: center;
                background-color: #2E86AB;
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
            }}
            .prediction-box {{
                background-color: {'#ffebee' if prediction and prediction > 0 else '#e8f5e8'};
                border: 2px solid {'#f44336' if prediction and prediction > 0 else '#4caf50'};
                border-radius: 10px;
                padding: 15px;
                margin: 20px 0;
                text-align: center;
            }}
            .chart-container {{
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🛫 Flight Delay Analytics Dashboard</h1>
            <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    """
    
    if user_inputs and prediction is not None:
        html_content += f"""
        <div class="prediction-box">
            <h2>Prediction Result for {user_inputs['airline']} Flight</h2>
            <h3>{user_inputs['origin']} → {user_inputs['dest']} on {user_inputs['date']}</h3>
            <p style="font-size: 18px; font-weight: bold;">
                {'🔺 Predicted delay: ' + str(abs(prediction)) + ' minutes' if prediction > 0 
                 else '✅ Predicted early/on-time arrival: ' + str(abs(prediction)) + ' minutes early'}
            </p>
        </div>
        """
    
    html_content += """
        <div class="chart-container">
            <div id="charts"></div>
        </div>
        <script>
    """
    
    # Add the plotly figure
    html_content += f"""
            var figure = {fig.to_json()};
            Plotly.plot('charts', figure.data, figure.layout);
        </script>
    </body>
    </html>
    """
    
    # Save to file
    filename = f"flight_delay_dashboard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return filename

# ---------- Streamlit Interface ----------
def main():
    st.set_page_config(page_title="Flight Delay Predictor", layout="centered")
    st.title("🛫 Flight Delay Prediction System")
    st.markdown("Enter flight information below to predict arrival delays.")

    with st.spinner("Loading data and model..."):
        df = load_and_prepare_data()
        if df is None:
            return
        X, y, encoders, features = preprocess_features(df)
        model, feature_columns = train_model(X, y)

    st.subheader("✍️ Enter Flight Details")
    with st.form("prediction_form"):
        origin = st.text_input("Origin Airport Code (e.g., JFK)").upper()
        dest = st.text_input("Destination Airport Code (e.g., LAX)").upper()
        airline = st.text_input("Airline Code (e.g., AA)").upper()
        date = st.date_input("Flight Date", datetime.date.today())
        scheduled_departure = st.number_input("Scheduled Departure (HHMM)", value=1400, step=100)
        scheduled_arrival = st.number_input("Scheduled Arrival (HHMM)", value=1600, step=100)
        submitted = st.form_submit_button("Predict Delay")

    if submitted:
        if not origin or not dest or not airline:
            st.error("Please provide origin, destination, and airline codes.")
            return

        user_inputs = {
            "origin": origin,
            "dest": dest,
            "airline": airline,
            "date": date,
            "scheduled_departure": scheduled_departure,
            "scheduled_arrival": scheduled_arrival
        }

        with st.spinner("Predicting delay..."):
            input_df = create_prediction_input(user_inputs, df, encoders, feature_columns)
            prediction = model.predict(input_df)[0]

        st.subheader(f"🕑 Prediction Result for {airline}")
        if prediction > 0:
            st.warning(f"🔺 Predicted delay: {prediction:.1f} minutes")
            if prediction > 15:
                st.error("⚠️ Significant delay expected. Consider alternate plans.")
        else:
            st.success(f"✅ Predicted early/ontime arrival: {abs(prediction):.1f} minutes early")

        st.subheader("📍 Airport Weather Conditions")
        origin_weather_text = get_airport_weather_summary(df, origin, date.month, "origin")
        dest_weather_text = get_airport_weather_summary(df, dest, date.month, "dest")
        st.text_area("Origin", origin_weather_text, height=100)
        st.text_area("Destination", dest_weather_text, height=100)

        st.subheader("🌦️ Weather Impact Summary")
        weather_reason = get_weather_delay_reason(df, origin, dest, date)
        st.text_area("Weather Delay Reason", weather_reason, height=100)

        # ---------- Generate HTML Charts ----------
        st.subheader("📊 Generating Analytics Dashboard")
        with st.spinner("Creating comprehensive analytics dashboard..."):
            try:
                html_filename = generate_html_charts(df, user_inputs, prediction)
                st.success(f"✅ Dashboard generated successfully!")
                st.info(f"📄 HTML file saved as: **{html_filename}**")
                st.markdown("The dashboard includes:")
                st.markdown("""
                - 📈 Top 10 airports with highest delays
                - 📅 Monthly delay trends
                - 🌦️ Weather-related delay causes breakdown
                - ✈️ Airline performance comparison
                - 📊 Daily delay patterns
                - 🎯 Route-specific analysis (for your selected route)
                """)
                
                # Provide download option
                try:
                    with open(html_filename, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    st.download_button(
                        label="💾 Download Dashboard HTML",
                        data=html_content,
                        file_name=html_filename,
                        mime="text/html"
                    )
                except Exception as e:
                    st.error(f"Error reading HTML file for download: {e}")
                    
            except Exception as e:
                st.error(f"Error generating dashboard: {e}")

    # Option to generate dashboard without prediction
    st.markdown("---")
    if st.button("📊 Generate Dashboard (No Prediction Required)"):
        with st.spinner("Creating analytics dashboard..."):
            try:
                html_filename = generate_html_charts(df, None, None)
                st.success(f"✅ Dashboard generated successfully!")
                st.info(f"📄 HTML file saved as: **{html_filename}**")
                
                # Provide download option
                try:
                    with open(html_filename, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    st.download_button(
                        label="💾 Download Dashboard HTML",
                        data=html_content,
                        file_name=html_filename,
                        mime="text/html"
                    )
                except Exception as e:
                    st.error(f"Error reading HTML file for download: {e}")
                    
            except Exception as e:
                st.error(f"Error generating dashboard: {e}")

if __name__ == "__main__":
    main()