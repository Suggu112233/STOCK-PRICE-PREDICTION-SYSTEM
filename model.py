def prediction(stock, n_days):
    import yfinance as yf
    import numpy as np
    from datetime import date, timedelta
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVR
    import plotly.graph_objs as go

    try:
        # Validate n_days input
        if n_days is None:
            return "Error: The number of days (n_days) is missing. Please provide a valid integer."
        
        # Convert n_days to an integer if it's not already
        try:
            n_days = int(n_days)
        except ValueError:
            return "Error: The number of days must be a valid integer."
        
        # Load the stock data for the past month
        df = yf.download(stock, period='1mo')
        if df.empty or len(df) < 2:
            return "Not enough data to make predictions."
        
        df.reset_index(inplace=True)

        # Generate a list of day indices for feature engineering
        days = list(range(len(df)))
        df['Day'] = df.index

        # Define the features (X) and target (Y)
        X = np.array(days).reshape(-1, 1)  # Reshape to ensure compatibility with sklearn
        Y = df['Close'].values.reshape(-1, 1)

        # Ensure sufficient data for training and testing
        if len(X) <= 1:
            return "Insufficient data points for prediction."
        if len(X) < 10:
            return "Not enough data to perform train-test split. Need at least 10 data points."
        
        # Split the data into training and testing sets, with a seed for random_state
        x_train, x_test, y_train, _ = train_test_split(X, Y, test_size=0.1, shuffle=False, random_state=42)
        
        if len(x_train) == 0 or len(x_test) == 0:
            return "The dataset is too small to perform train-test split."

        # Define and train the SVR model
        svr_model = SVR(kernel='rbf', C=1, epsilon=0.1, gamma=0.1)
        y_train = y_train.ravel()  # Flatten the target array for fitting
        svr_model.fit(x_train, y_train)

        # Prepare the days for which to predict future prices
        last_day = x_test[-1][0]
        output_days = np.array(range(last_day + 1, last_day + n_days)).reshape(-1, 1)

        # Generate future dates starting from today
        future_dates = [date.today() + timedelta(days=i) for i in range(1, n_days + 1)]

        # Predict future prices using the trained model
        predicted_prices = svr_model.predict(output_days)

        # Create a Plotly figure to visualize the predicted prices
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predicted_prices,
            mode='lines+markers',
            name='Predicted Prices'
        ))
        
        # Update the layout of the figure
        fig.update_layout(
            title=f"Predicted Close Price for the Next {n_days} Days",
            xaxis_title="Date",
            yaxis_title="Close Price",
        )

        return fig
    
    except Exception as e:
        return f"An error occurred: {e}"