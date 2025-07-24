from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import numpy as np
import pickle
import datetime
import requests
import warnings
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import plotly.express as px
import plotly.utils
import json
import os

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# Global variables to store model components
model = None
encoders = None
feature_columns = None
airports_df = None

def load_model_safe():
    """Safely load the saved model and components"""
    try:
        with open('models/test02.pkl', 'rb') as f:
            model_package = pickle.load(f)

        return (
            model_package['model'],
            model_package['encoders'],
            model_package['feature_columns'],
            model_package.get('created_date', 'Unknown'),
            model_package.get('version', 'Unknown')
        )
    except FileNotFoundError:
        print("Model file 'models/test02.pkl' not found. Please ensure the model file is in the correct directory.")
        return None, None, None, None, None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None, None, None, None

def load_airports_data_safe():
    """Safely load airports data"""
    try:
        airports_df = pd.read_csv("airports.csv")
        if airports_df.empty:
            print("Airports data file is empty. Please check the file content.")
            return None
        return airports_df
    except FileNotFoundError:
        print("Airports data file 'airports.csv' not found. This file is required for airport validation.")
        return None
    except Exception as e:
        print(f"Error loading airports data: {str(e)}")
        return None

def validate_airports(origin_code, dest_code, airports_df):
    """Validate that both airports exist in the airports database"""
    if airports_df is None:
        raise ValueError("Airports database not loaded. Cannot validate airports.")
    
    missing_airports = []
    
    if origin_code not in airports_df['IATA_CODE'].values:
        missing_airports.append(f"Origin airport '{origin_code}'")
    
    if dest_code not in airports_df['IATA_CODE'].values:
        missing_airports.append(f"Destination airport '{dest_code}'")
    
    if missing_airports:
        error_msg = f"The following airports are not found in the database: {', '.join(missing_airports)}"
        available_airports = sorted(airports_df['IATA_CODE'].unique())
        raise ValueError(f"{error_msg}\n\nAvailable airports: {', '.join(available_airports[:20])}{'...' if len(available_airports) > 20 else ''}")
    
    return True

def get_weather_forecast(iata_code, date_str, api_key, airports_df):
    """Get weather forecast from Visual Crossing API"""
    if not api_key:
        return None, "no_api_key"

    try:
        # Get coordinates for the airport from the database
        if airports_df is None or airports_df.empty:
            return None, "no_airports_data"
        
        if iata_code not in airports_df['IATA_CODE'].values:
            return None, "unknown_airport"
        
        airport_row = airports_df[airports_df['IATA_CODE'] == iata_code].iloc[0]
        lat, lon = airport_row['LATITUDE'], airport_row['LONGITUDE']

        # Visual Crossing API call
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{date_str}"

        params = {
            'key': api_key,
            'unitGroup': 'metric',
            'include': 'days',
            'elements': 'temp,humidity,pressure,windspeed,winddir,cloudcover,visibility,conditions,precip,snow'
        }

        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()

        data = response.json()

        if 'days' not in data or len(data['days']) == 0:
            return None, "no_data"

        day_data = data['days'][0]

        weather_features = {
            'temperature': day_data.get('temp', 20.0),
            'humidity': day_data.get('humidity', 50.0),
            'pressure': day_data.get('pressure', 1013.25),
            'wind_speed': day_data.get('windspeed', 10.0),
            'wind_direction': day_data.get('winddir', 180.0),
            'cloudiness': day_data.get('cloudcover', 25.0),
            'visibility': day_data.get('visibility', 10.0),
            'weather_desc': day_data.get('conditions', 'Clear'),
            'precipitation': day_data.get('precip', 0.0),
            'snow': day_data.get('snow', 0.0)
        }

        return weather_features, "success"

    except Exception as e:
        return None, f"error: {str(e)}"

def analyze_weather_impact(weather_data, location_name):
    """Analyze weather conditions and identify delay-causing factors"""
    if not weather_data:
        return [], "LOW"
    
    delay_factors = []
    risk_level = "LOW"
    
    # Define thresholds for weather conditions that typically cause delays
    thresholds = {
        'high_wind': 40,        # km/h
        'low_visibility': 5,    # km
        'heavy_rain': 10,       # mm
        'snow_present': 0.5,    # cm
        'extreme_cold': -10,    # °C
        'extreme_heat': 35,     # °C
    }
    
    # Check each weather factor
    if weather_data['wind_speed'] > thresholds['high_wind']:
        severity = "HIGH" if weather_data['wind_speed'] > 60 else "MEDIUM"
        delay_factors.append({
            'factor': 'High Wind Speed',
            'value': f"{weather_data['wind_speed']:.1f} km/h",
            'threshold': f">{thresholds['high_wind']} km/h",
            'severity': severity,
            'impact': 'Can cause landing/takeoff delays and turbulence',
            'location': location_name
        })
        if severity == "HIGH":
            risk_level = "HIGH"
        elif risk_level == "LOW":
            risk_level = "MEDIUM"
    
    if weather_data['visibility'] < thresholds['low_visibility']:
        severity = "HIGH" if weather_data['visibility'] < 2 else "MEDIUM"
        delay_factors.append({
            'factor': 'Poor Visibility',
            'value': f"{weather_data['visibility']:.1f} km",
            'threshold': f"<{thresholds['low_visibility']} km",
            'severity': severity,
            'impact': 'Requires instrument approaches, reduces airport capacity',
            'location': location_name
        })
        if severity == "HIGH":
            risk_level = "HIGH"
        elif risk_level == "LOW":
            risk_level = "MEDIUM"
    
    if weather_data['precipitation'] > thresholds['heavy_rain']:
        severity = "HIGH" if weather_data['precipitation'] > 25 else "MEDIUM"
        delay_factors.append({
            'factor': 'Heavy Precipitation',
            'value': f"{weather_data['precipitation']:.1f} mm",
            'threshold': f">{thresholds['heavy_rain']} mm",
            'severity': severity,
            'impact': 'Reduces runway capacity, increases stopping distances',
            'location': location_name
        })
        if severity == "HIGH":
            risk_level = "HIGH"
        elif risk_level == "LOW":
            risk_level = "MEDIUM"
    
    if weather_data['snow'] > thresholds['snow_present']:
        severity = "HIGH" if weather_data['snow'] > 5 else "MEDIUM"
        delay_factors.append({
            'factor': 'Snow Conditions',
            'value': f"{weather_data['snow']:.1f} cm",
            'threshold': f">{thresholds['snow_present']} cm",
            'severity': severity,
            'impact': 'Requires de-icing, runway clearing, major delays possible',
            'location': location_name
        })
        risk_level = "HIGH"  # Snow always high risk
    
    if weather_data['temperature'] < thresholds['extreme_cold']:
        severity = "HIGH" if weather_data['temperature'] < -20 else "MEDIUM"
        delay_factors.append({
            'factor': 'Extreme Cold',
            'value': f"{weather_data['temperature']:.1f}°C",
            'threshold': f"<{thresholds['extreme_cold']}°C",
            'severity': severity,
            'impact': 'Requires extensive de-icing, equipment issues possible',
            'location': location_name
        })
        if severity == "HIGH":
            risk_level = "HIGH"
        elif risk_level == "LOW":
            risk_level = "MEDIUM"
    
    if weather_data['temperature'] > thresholds['extreme_heat']:
        severity = "MEDIUM"
        delay_factors.append({
            'factor': 'Extreme Heat',
            'value': f"{weather_data['temperature']:.1f}°C",
            'threshold': f">{thresholds['extreme_heat']}°C",
            'severity': severity,
            'impact': 'Reduces aircraft performance, weight restrictions possible',
            'location': location_name
        })
        if risk_level == "LOW":
            risk_level = "MEDIUM"
    
    # Check for severe weather conditions
    severe_conditions = ['thunderstorm', 'storm', 'fog', 'mist', 'freezing', 'blizzard', 'hail']
    if any(condition in weather_data['weather_desc'].lower() for condition in severe_conditions):
        severity = "HIGH" if any(condition in weather_data['weather_desc'].lower() 
                               for condition in ['thunderstorm', 'blizzard', 'freezing']) else "MEDIUM"
        delay_factors.append({
            'factor': 'Severe Weather Conditions',
            'value': weather_data['weather_desc'],
            'threshold': 'Clear/Partly Cloudy preferred',
            'severity': severity,
            'impact': 'Various operational restrictions and safety concerns',
            'location': location_name
        })
        if severity == "HIGH":
            risk_level = "HIGH"
        elif risk_level == "LOW":
            risk_level = "MEDIUM"
    
    return delay_factors, risk_level

def create_prediction_input(inputs, encoders, feature_columns, api_key, airports_df):
    """Create input dataframe for prediction and return weather data"""
    try:
        date_obj = datetime.datetime.strptime(inputs['date_str'], "%Y-%m-%d")
    except ValueError:
        return None, "Invalid date format", None, None

    # Create base input
    input_dict = {
        'YEAR': date_obj.year,
        'MONTH': date_obj.month,
        'DAY': date_obj.day,
        'SCHEDULED_DEPARTURE': inputs['scheduled_departure'],
        'SCHEDULED_ARRIVAL': inputs['scheduled_arrival'],
        'ORIGIN_AIRPORT': inputs['origin'],
        'DESTINATION_AIRPORT': inputs['dest'],
        'AIRLINE': inputs['airline']
    }

    # Add weather data if available
    today = datetime.date.today()
    prediction_date = date_obj.date()
    use_forecast = prediction_date >= today and api_key is not None

    origin_weather = None
    dest_weather = None

    if use_forecast and airports_df is not None:
        # Get weather forecasts
        origin_weather, _ = get_weather_forecast(inputs['origin'], inputs['date_str'], api_key, airports_df)
        dest_weather, _ = get_weather_forecast(inputs['dest'], inputs['date_str'], api_key, airports_df)

        if origin_weather:
            for key, value in origin_weather.items():
                if key != 'weather_desc':  # Skip non-numeric
                    input_dict[f'origin_{key}'] = value

        if dest_weather:
            for key, value in dest_weather.items():
                if key != 'weather_desc':  # Skip non-numeric
                    input_dict[f'dest_{key}'] = value

    # Fill missing features with defaults
    for col in feature_columns:
        if col not in input_dict:
            input_dict[col] = 0

    # Create DataFrame
    input_df = pd.DataFrame([input_dict])

    # Encode categorical features
    for col, encoder in encoders.items():
        if col in input_df.columns:
            value = str(input_df[col].iloc[0])
            if value in encoder.classes_:
                input_df[col] = encoder.transform([value])[0]
            else:
                input_df[col] = encoder.transform([encoder.classes_[0]])[0]

    # Ensure all required features are present
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    return input_df, "success", origin_weather, dest_weather

def create_delay_visualization(prediction, airline, route):
    """Create a visualization for the delay prediction"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{airline} {route}<br>Delay Prediction (minutes)"},
        delta={'reference': 0},
        gauge={
            'axis': {'range': [None, 120]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 15], 'color': "lightgreen"},
                {'range': [15, 30], 'color': "yellow"},
                {'range': [30, 60], 'color': "orange"},
                {'range': [60, 120], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 30
            }
        }
    ))

    fig.update_layout(
        height=400,
        font={'color': "darkblue", 'family': "Arial"}
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)



# Routes
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/contact')
def contact():
    """Contact page"""
    return render_template('contact.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page with prediction form"""
    airlines = ["AA", "UA", "DL", "WN", "AS", "B6", "F9", "NK", "G4", "SY"]
    return render_template('dashboard.html', airlines=airlines)

@app.route('/departure_heatmap')
def departure_heatmap():
    """Departure heatmap visualization page"""
    return render_template('departure_delays.html')

@app.route('/arrival_heatmap')
def arrival_heatmap():
    """Arrival heatmap visualization page"""
    return render_template('arrival_delays.html')

@app.route('/charts')
def charts():
    """Charts and analytics page"""
    return render_template('charts.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    global model, encoders, feature_columns, airports_df
    
    try:
        # Get form data
        origin = request.form['origin'].upper().strip()
        dest = request.form['dest'].upper().strip()
        airline = request.form['airline'].upper().strip()
        flight_date = request.form['flight_date']
        departure_time = request.form['departure_time']
        arrival_time = request.form['arrival_time']
        
        # Validate inputs
        if len(origin) != 3 or len(dest) != 3:
            flash("Airport codes must be exactly 3 letters!", "error")
            return redirect(url_for('dashboard'))
        
        if origin == dest:
            flash("Origin and destination cannot be the same!", "error")
            return redirect(url_for('dashboard'))
        
        # Validate airports exist in database
        validate_airports(origin, dest, airports_df)
        
        # Parse time inputs
        dep_hour, dep_min = map(int, departure_time.split(':'))
        arr_hour, arr_min = map(int, arrival_time.split(':'))
        scheduled_departure = dep_hour * 100 + dep_min
        scheduled_arrival = arr_hour * 100 + arr_min
        
        # Prepare input data
        inputs = {
            'origin': origin,
            'dest': dest,
            'airline': airline,
            'date_str': flight_date,
            'scheduled_departure': scheduled_departure,
            'scheduled_arrival': scheduled_arrival
        }
        
        # Get prediction with weather data
        api_key = 'KZG5KUC6LL62Z5LHDDZ3TTGVC'  # Replace with your API key
        input_df, status, origin_weather, dest_weather = create_prediction_input(
            inputs, encoders, feature_columns, api_key, airports_df
        )
        
        if input_df is None:
            flash(f"Error creating prediction input: {status}", "error")
            return redirect(url_for('dashboard'))
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Analyze weather impact
        origin_factors, origin_risk = analyze_weather_impact(origin_weather, f"Origin ({origin})")
        dest_factors, dest_risk = analyze_weather_impact(dest_weather, f"Destination ({dest})")
        
        # Determine overall risk
        all_factors = origin_factors + dest_factors
        overall_risk = "LOW"
        if origin_risk == "HIGH" or dest_risk == "HIGH":
            overall_risk = "HIGH"
        elif origin_risk == "MEDIUM" or dest_risk == "MEDIUM":
            overall_risk = "MEDIUM"
        
        # Create visualization
        chart_json = create_delay_visualization(prediction, airline, f"{origin} → {dest}")
        
        # Format flight date for display
        flight_date_obj = datetime.datetime.strptime(flight_date, "%Y-%m-%d")
        formatted_date = flight_date_obj.strftime('%B %d, %Y')
        
        return render_template('results.html',
                             prediction=prediction,
                             airline=airline,
                             origin=origin,
                             dest=dest,
                             flight_date=formatted_date,
                             departure_time=departure_time,
                             arrival_time=arrival_time,
                             chart_json=chart_json,
                             origin_weather=origin_weather,
                             dest_weather=dest_weather,
                             origin_factors=origin_factors,
                             dest_factors=dest_factors,
                             overall_risk=overall_risk)
        
    except ValueError as e:
        flash(str(e), "error")
        return redirect(url_for('dashboard'))
    except Exception as e:
        flash(f"Error making prediction: {str(e)}", "error")
        return redirect(url_for('dashboard'))

@app.route('/api/airports')
def get_airports():
    """API endpoint to get airports data"""
    if airports_df is not None:
        airports_list = airports_df[['IATA_CODE', 'AIRPORT', 'CITY', 'STATE']].to_dict('records')
        return jsonify(airports_list)
    return jsonify([])

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
        
        # Check if airports data is loaded
        if airports_df is None:
            return jsonify({'status': 'error', 'message': 'Airports data not loaded'}), 500
        
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'airports_count': len(airports_df),
            'timestamp': datetime.datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Initialize the model and data when the app starts
def initialize_app():
    global model, encoders, feature_columns, airports_df
    
    # Load model
    model, encoders, feature_columns, created_date, version = load_model_safe()
    if model is None:
        print("Failed to load model. Please check model file.")
        return False
    
    # Load airports data
    airports_df = load_airports_data_safe()
    if airports_df is None:
        print("Failed to load airports data. Please check airports file.")
        return False
    
    print(f"Model loaded successfully: {created_date}, Version: {version}")
    print(f"Airports loaded: {len(airports_df)} airports")
    return True

if __name__ == '__main__':
    if initialize_app():
        app.run(debug=True, port=5000)
    else:
        print(" Failed to initialize application.")