from flask import Flask, render_template
from xgboost import XGBClassifier
import xgboost as xgb
import yfinance as yahooFinance
import pandas as pd
import numpy as np
import os

my_dir = os.path.dirname(__file__)

def get_prediction(ticker):
    history = yahooFinance.Ticker(ticker)
    data = history.history(period="1y")

    data['Change'] = (data['Close'] - data['Open']) / data['Open'] 
    data['Date'] = pd.to_datetime(data.index)
    data['day_of_year'] = data['Date'].dt.dayofyear

    model = xgb.XGBClassifier()
    model_path = os.path.join(my_dir, 'xgb_model_classification_' + ticker + '.json')
    model.load_model('xgb_model_classification_' + ticker + '.json')

    N = len(data)
    W = 15
    n_attributes = 4
    X = np.zeros((N - W, n_attributes * W + 1))
    y = data['Change'].copy()
    y = y[W:].to_numpy()

    for i in range(W, N):
        for j in range(W):
            X[i - W, j * n_attributes + 0] = abs(data['High'].iloc[i - j - 1] - data['Low'].iloc[i - j - 1])/data['Low'].iloc[i - j - 1]
            X[i - W, j * n_attributes + 1] = data['Volume'].iloc[i - j - 1]
            X[i - W, j * n_attributes + 2] = data['Change'].iloc[i - j - 1]
        X[i - W, -1] = data['day_of_year'].iloc[i]
    y_pred = model.predict(X)
    return 'Up' if y_pred[-1] == 1 else 'Down'

def get_recent_price(ticker):
    history = yahooFinance.Ticker(ticker)
    data = history.history(period="1mo")
    data['Date'] = pd.to_datetime(data.index).date.astype(str)
    data = data[['Date', 'Close']].values.tolist()

    # Start the HTML table
    html_code = "<table border='1'>\n"

    # Add a header row
    html_code += "<tr><th>Date</th><th>Value</th></tr>\n"

    # Add data rows
    for date, value in data:
        html_code += f"<tr><td>{date}</td><td>{value:.2f}</td></tr>\n"

    # Close the table
    html_code += "</table>"

    return html_code

# Initialize the Flask application
app = Flask(__name__)

# Define a route for the main page
@app.route('/')
def index():
    # Render an HTML template
    return render_template('index.html', qqq_pred = get_prediction('QQQ'), qqq_table = get_recent_price('QQQ'), spy_pred = get_prediction('SPY'), spy_table = get_recent_price('SPY'))

# Define additional routes as needed
@app.route('/about')
def about():
    return 'About Page'

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
