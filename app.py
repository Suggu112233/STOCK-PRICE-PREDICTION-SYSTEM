import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from datetime import datetime as dt, timedelta
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go  
import plotly.express as px
from model import prediction  # Ensure this is correct
import joblib
import numpy as np

# Helper function to create stock price figure
def get_stock_price_fig(df):
    fig = px.line(df, x="Date", y=["Close", "Open"], title="Closing and Opening Price vs Date")
    return fig

# Helper function to create exponential moving average figure
def get_more(df):
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.scatter(df, x="Date", y="EWA_20", title="Exponential Moving Average vs Date")
    fig.update_traces(mode='lines+markers')
    return fig

# Dash app initialization
app = dash.Dash(__name__, external_stylesheets=["https://fonts.googleapis.com/css2?family=Roboto&display=swap"])
server = app.server

# Layout of the Dash app
app.layout = html.Div(
    [
        html.Div(
            [
                html.P("Welcome to the Stock Dash App!", className="start"),
                html.Div([
                    html.P("Input stock code: "),
                    html.Div([
                        dcc.Input(id="dropdown_tickers", type="text"),
                        html.Button("Submit", id='submit'),
                    ], className="form")
                ], className="input-place"),
                html.Div([
                    dcc.DatePickerRange(
                        id='my-date-picker-range',
                        min_date_allowed=dt(1995, 8, 5),
                        max_date_allowed=dt.now(),
                        initial_visible_month=dt.now(),
                        end_date=dt.now().date()
                    ),
                ], className="date"),
                html.Div([
                    html.Button("Stock Price", className="stock-btn", id="stock"),
                    html.Button("Indicators", className="indicators-btn", id="indicators"),
                    dcc.Input(id="n_days", type="text", placeholder="number of days"),
                    html.Button("Forecast", className="forecast-btn", id="forecast")
                ], className="buttons"),
            ], className="nav"),

        html.Div(
            [
                html.Div(
                    [html.Img(id="logo"), html.P(id="ticker")],
                    className="header"),
                html.Div(id="description", className="description_ticker"),
                html.Div([], id="graphs-content"),
                html.Div([], id="main-content"),
                html.Div([], id="forecast-content")
            ], className="content"),
    ],
    className="container"
)

# Callback to update company info
@app.callback(
    [
        Output("description", "children"),
        Output("logo", "src"),
        Output("ticker", "children"),
        Output("stock", "n_clicks"),
        Output("indicators", "n_clicks"),
        Output("forecast", "n_clicks")
    ],
    [Input("submit", "n_clicks")],
    [State("dropdown_tickers", "value")]
)
def update_data(n, val):
    if n is None or val is None:
        return "Hey there! Please enter a legitimate stock code to get details.", " ", ".", None, None, None
    ticker = yf.Ticker(val)
    info = ticker.info
    return info.get('longBusinessSummary', 'No summary available'), \
           info.get('logo_url', ''), \
           info.get('shortName', 'Unknown ticker'), None, None, None

# Callback for stock price graphs
@app.callback(
    Output("graphs-content", "children"),
    [
        Input("stock", "n_clicks"),
        Input('my-date-picker-range', 'start_date'),
        Input('my-date-picker-range', 'end_date')
    ],
    [State("dropdown_tickers", "value")]
)
def stock_price(n, start_date, end_date, val):
    if n is None or val is None:
        raise PreventUpdate
    df = yf.download(val, start=start_date, end=end_date) if start_date else yf.download(val)
    df.reset_index(inplace=True)
    df['Date'] = df.index  # Ensure 'Date' column exists
    fig = get_stock_price_fig(df)
    return [dcc.Graph(figure=fig)]

# Callback for indicators
@app.callback(
    Output("main-content", "children"),
    [
        Input("indicators", "n_clicks"),
        Input('my-date-picker-range', 'start_date'),
        Input('my-date-picker-range', 'end_date')
    ],
    [State("dropdown_tickers", "value")]
)
def indicators(n, start_date, end_date, val):
    if n is None or val is None:
        raise PreventUpdate
    df_more = yf.download(val, start=start_date, end=end_date) if start_date else yf.download(val)
    df_more.reset_index(inplace=True)
    df_more['Date'] = df_more.index  # Ensure 'Date' column exists
    fig = get_more(df_more)
    return [dcc.Graph(figure=fig)]

# Callback for forecast
@app.callback(
    Output("forecast-content", "children"),
    [Input("forecast", "n_clicks")],
    [State("n_days", "value"), State("dropdown_tickers", "value")]
)
def forecast(n, n_days, val):
    if n is None or val is None:
        raise PreventUpdate

    try:
        fig = prediction(val, int(n_days) + 1)
        
        # Check if fig is a string (error message)
        if isinstance(fig, str):
            return [
                html.Div(
                    children=fig,
                    style={"color": "red", "font-weight": "bold"}
                )
            ]
        
        return [dcc.Graph(figure=fig)]
    
    except Exception as e:
        return [
            html.Div(
                children=f"An error occurred: {e}",
                style={"color": "red", "font-weight": "bold"}
            )
        ]

# Function to predict and compare stock prices
def predict_and_compare(stock, n_days):
    model = joblib.load('path_to_your_model.pkl')
    end_date = dt.today()
    start_date = end_date - timedelta(days=n_days)

    stock_data = yf.download(stock, start=start_date, end=end_date)
    actual_prices = stock_data['Close'].values

    predictions = []
    for i in range(len(actual_prices)):
        if i < model.input_length:
            predictions.append(np.nan)
        else:
            input_data = actual_prices[i-model.input_length:i].reshape(1, -1)
            predicted_price = model.predict(input_data)
            predictions.append(predicted_price[0])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=actual_prices, mode='lines', name='Actual Prices'))
    fig.add_trace(go.Scatter(x=stock_data.index, y=predictions, mode='lines', name='Predicted Prices'))
    fig.update_layout(title=f'Actual vs Predicted Stock Prices for {stock}', xaxis_title='Date', yaxis_title='Stock Price')

    return fig

# Running the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
