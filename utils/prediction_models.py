import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from pmdarima.arima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    # Alternative implementation or placeholder

class ESPredictionModels:
    """
    A class that implements various forecasting models for Earned Schedule prediction.
    """
    
    def __init__(self):
        self.data = None
        self.time_series = None
        self.models = {}
        self.forecasts = {}
        self.metrics = {}
        self.best_model = None
        
    def load_data(self, data, time_column='Time', spi_t_column='SPI(t)', es_column='ES', at_column='AT', pd_column='PD'):
        """
        Load data for prediction models.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing time-phased project data
        time_column : str
            Column name for time periods
        spi_t_column : str
            Column name for SPI(t) values
        es_column : str
            Column name for ES values
        at_column : str
            Column name for AT values
        pd_column : str
            Column name for PD values
        """
        self.data = data.copy()
        self.time_series = {
            'time': data[time_column].values,
            'spi_t': data[spi_t_column].values if spi_t_column in data.columns else None,
            'es': data[es_column].values if es_column in data.columns else None,
            'at': data[at_column].values if at_column in data.columns else None,
            'pd': data[pd_column].values[0] if pd_column in data.columns else None
        }
        
    def _create_features(self, series, lags=3):
        """
        Create lagged features for ML models.
        
        Parameters:
        -----------
        series : np.array
            Time series to create features from
        lags : int
            Number of lag periods to include
            
        Returns:
        --------
        X : np.array
            Feature matrix
        y : np.array
            Target vector
        """
        X = []
        y = []
        
        for i in range(lags, len(series)):
            X.append(series[i-lags:i])
            y.append(series[i])
            
        return np.array(X), np.array(y)
        
    def fit_linear_regression(self, target='es', forecast_periods=10):
        """
        Fit linear regression model on time series data.
        
        Parameters:
        -----------
        target : str
            Target variable to forecast ('es' or 'spi_t')
        forecast_periods : int
            Number of periods to forecast
            
        Returns:
        --------
        dict: Forecast results
        """
        if self.time_series is None:
            raise ValueError("Data must be loaded first")
            
        series = self.time_series['es'] if target == 'es' else self.time_series['spi_t']
        time = self.time_series['time']
        
        if series is None or time is None:
            raise ValueError(f"{target} or time data is missing")
            
        # Reshape for sklearn
        X = time.reshape(-1, 1)
        y = series
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        
        # Make predictions
        last_time = time[-1]
        future_times = np.array([last_time + i + 1 for i in range(forecast_periods)]).reshape(-1, 1)
        forecast = model.predict(future_times)
        
        # Calculate metrics
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        # Store results
        self.models['linear_regression'] = model
        self.forecasts['linear_regression'] = {
            'times': np.concatenate([time, future_times.flatten()]),
            'values': np.concatenate([y_pred, forecast]),
            'future_times': future_times.flatten(),
            'future_values': forecast,
            'target': target
        }
        self.metrics['linear_regression'] = {
            'r2': r2,
            'mse': mse,
            'mae': mae,
            'formula': f"{target} = {model.coef_[0]:.4f} * time + {model.intercept_:.4f}"
        }
        
        return self.forecasts['linear_regression']
    
    def fit_exponential_smoothing(self, target='es', forecast_periods=10, seasonal_periods=None):
        """
        Fit exponential smoothing model on time series data.
        
        Parameters:
        -----------
        target : str
            Target variable to forecast ('es' or 'spi_t')
        forecast_periods : int
            Number of periods to forecast
        seasonal_periods : int or None
            Number of periods in a seasonal cycle (None for non-seasonal data)
            
        Returns:
        --------
        dict: Forecast results
        """
        if self.time_series is None:
            raise ValueError("Data must be loaded first")
            
        series = self.time_series['es'] if target == 'es' else self.time_series['spi_t']
        time = self.time_series['time']
        
        if series is None or time is None:
            raise ValueError(f"{target} or time data is missing")
            
        # Convert to pandas Series for statsmodels
        ts = pd.Series(series, index=pd.RangeIndex(start=0, stop=len(series)))
        
        # Determine model type based on data properties
        if len(ts) < 4:  # Very short series
            model_type = 'simple'
        else:
            model_type = 'holt'
            
        if seasonal_periods is not None and len(ts) >= 2 * seasonal_periods:
            model_type = 'holt_winters'
        
        # Fit model based on type
        if model_type == 'simple':
            model = ExponentialSmoothing(ts, trend=None, seasonal=None)
        elif model_type == 'holt':
            model = ExponentialSmoothing(ts, trend='add', seasonal=None)
        else:  # holt_winters
            model = ExponentialSmoothing(ts, trend='add', seasonal='add', seasonal_periods=seasonal_periods)
            
        fitted_model = model.fit(optimized=True)
        
        # Make predictions
        forecast = fitted_model.forecast(forecast_periods)
        
        # Calculate metrics
        y_pred = fitted_model.fittedvalues
        if len(y_pred) < len(series):  # Handle potential NaN values at start
            y_pred = np.concatenate([np.full(len(series) - len(y_pred), np.nan), y_pred])
            
        # Remove NaN for metric calculation
        mask = ~np.isnan(y_pred)
        r2 = r2_score(series[mask], y_pred[mask]) if np.sum(mask) > 1 else np.nan
        mse = mean_squared_error(series[mask], y_pred[mask]) if np.sum(mask) > 0 else np.nan
        mae = mean_absolute_error(series[mask], y_pred[mask]) if np.sum(mask) > 0 else np.nan
        
        # Store results
        self.models['exp_smoothing'] = fitted_model
        self.forecasts['exp_smoothing'] = {
            'times': np.concatenate([time, np.array([time[-1] + i + 1 for i in range(forecast_periods)])]),
            'values': np.concatenate([y_pred, forecast.values]),
            'future_times': np.array([time[-1] + i + 1 for i in range(forecast_periods)]),
            'future_values': forecast.values,
            'target': target
        }
        self.metrics['exp_smoothing'] = {
            'r2': r2,
            'mse': mse,
            'mae': mae,
            'model_type': model_type,
            'parameters': {
                'smoothing_level': fitted_model.params.get('smoothing_level', None),
                'smoothing_trend': fitted_model.params.get('smoothing_trend', None),
                'smoothing_seasonal': fitted_model.params.get('smoothing_seasonal', None)
            }
        }
        
        return self.forecasts['exp_smoothing']
    
    def fit_arima(self, target='es', forecast_periods=10, order=None):
        """
        Fit ARIMA model on time series data.
        
        Parameters:
        -----------
        target : str
            Target variable to forecast ('es' or 'spi_t')
        forecast_periods : int
            Number of periods to forecast
        order : tuple or None
            ARIMA order (p,d,q). If None, will be determined automatically
            
        Returns:
        --------
        dict: Forecast results
        """
        if self.time_series is None:
            raise ValueError("Data must be loaded first")
            
        series = self.time_series['es'] if target == 'es' else self.time_series['spi_t']
        time = self.time_series['time']
        
        if series is None or time is None:
            raise ValueError(f"{target} or time data is missing")
            
        # Convert to pandas Series for statsmodels
        ts = pd.Series(series, index=pd.RangeIndex(start=0, stop=len(series)))
        
        # If order not specified, use a simple default
        if order is None:
            if len(ts) <= 3:  # Very short series
                order = (1, 0, 0)  # Simple AR(1)
            else:
                # Try to find optimal order
                try:
                    # Find best order based on AIC
                    best_aic = float('inf')
                    best_order = (1, 0, 0)
                    
                    for p in range(3):
                        for d in range(2):
                            for q in range(3):
                                try:
                                    model = ARIMA(ts, order=(p, d, q))
                                    results = model.fit()
                                    if results.aic < best_aic:
                                        best_aic = results.aic
                                        best_order = (p, d, q)
                                except:
                                    continue
                    
                    order = best_order
                except:
                    # Fallback to default
                    order = (1, 0, 0)
        
        # Fit ARIMA model
        try:
            model = ARIMA(ts, order=order)
            fitted_model = model.fit()
            
            # Generate in-sample predictions
            y_pred = fitted_model.fittedvalues.values
            
            # Forecast future values
            forecast = fitted_model.forecast(steps=forecast_periods)
            
            # Calculate metrics
            if len(y_pred) < len(series):  # Handle potential missing values at start
                y_pred = np.concatenate([np.full(len(series) - len(y_pred), np.nan), y_pred])
                
            # Remove NaN for metric calculation
            mask = ~np.isnan(y_pred)
            r2 = r2_score(series[mask], y_pred[mask]) if np.sum(mask) > 1 else np.nan
            mse = mean_squared_error(series[mask], y_pred[mask]) if np.sum(mask) > 0 else np.nan
            mae = mean_absolute_error(series[mask], y_pred[mask]) if np.sum(mask) > 0 else np.nan
            
            # Store results
            self.models['arima'] = fitted_model
            self.forecasts['arima'] = {
                'times': np.concatenate([time, np.array([time[-1] + i + 1 for i in range(forecast_periods)])]),
                'values': np.concatenate([y_pred, forecast]),
                'future_times': np.array([time[-1] + i + 1 for i in range(forecast_periods)]),
                'future_values': forecast,
                'target': target
            }
            self.metrics['arima'] = {
                'r2': r2,
                'mse': mse,
                'mae': mae,
                'order': order,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            }
            
            return self.forecasts['arima']
            
        except Exception as e:
            print(f"ARIMA model fitting failed: {e}")
            return None
    
    def fit_machine_learning(self, model_type='random_forest', target='es', forecast_periods=10, lags=3):
        """
        Fit ML model (Random Forest or Gradient Boosting) on time series data with lagged features.
        
        Parameters:
        -----------
        model_type : str
            Type of ML model ('random_forest' or 'gradient_boosting')
        target : str
            Target variable to forecast ('es' or 'spi_t')
        forecast_periods : int
            Number of periods to forecast
        lags : int
            Number of lag periods to include as features
            
        Returns:
        --------
        dict: Forecast results
        """
        if self.time_series is None:
            raise ValueError("Data must be loaded first")
            
        series = self.time_series['es'] if target == 'es' else self.time_series['spi_t']
        time = self.time_series['time']
        
        if series is None or time is None:
            raise ValueError(f"{target} or time data is missing")
            
        if len(series) <= lags:
            raise ValueError(f"Need at least {lags+1} observations for ML models with {lags} lags")
        
        # Create lagged features
        X, y = self._create_features(series, lags)
        
        # Select model type
        if model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model_key = 'random_forest'
        else:  # gradient_boosting
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model_key = 'gradient_boosting'
        
        # Fit model
        model.fit(X, y)
        
        # Make in-sample predictions
        y_pred = model.predict(X)
        
        # Prepare data for forecasting
        forecast = []
        last_window = series[-lags:].copy()
        
        # Forecast future periods one by one
        for _ in range(forecast_periods):
            next_pred = model.predict([last_window])[0]
            forecast.append(next_pred)
            # Update window for next prediction
            last_window = np.append(last_window[1:], next_pred)
        
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        # Create times array for all predictions (including those we couldn't predict due to lags)
        all_y_pred = np.full(len(series), np.nan)
        all_y_pred[lags:] = y_pred  # Only filled from lag+1 onwards
        
        # Store results
        self.models[model_key] = model
        self.forecasts[model_key] = {
            'times': np.concatenate([time, np.array([time[-1] + i + 1 for i in range(forecast_periods)])]),
            'values': np.concatenate([all_y_pred, forecast]),
            'future_times': np.array([time[-1] + i + 1 for i in range(forecast_periods)]),
            'future_values': np.array(forecast),
            'target': target
        }
        self.metrics[model_key] = {
            'r2': r2,
            'mse': mse,
            'mae': mae,
            'lags': lags,
            'feature_importance': dict(zip([f"lag_{i+1}" for i in range(lags)], model.feature_importances_)) if hasattr(model, 'feature_importances_') else None
        }
        
        return self.forecasts[model_key]
    
    def predict_completion_date(self, model_key='linear_regression', pd=None, current_date=None, period_unit='month'):
        """
        Predict project completion date based on the forecasted ES.
        
        Parameters:
        -----------
        model_key : str
            Key of the model to use for prediction
        pd : float or None
            Planned Duration. If None, use the one from time_series
        current_date : datetime or None
            Current date. If None, use today's date
        period_unit : str
            Unit of time periods ('day', 'week', 'month')
            
        Returns:
        --------
        tuple: (completion_date, confidence_interval)
        """
        if model_key not in self.forecasts or self.forecasts[model_key]['target'] != 'es':
            raise ValueError(f"Model {model_key} not found or not forecasting ES")
            
        # Get planned duration
        if pd is None:
            pd = self.time_series['pd']
        if pd is None:
            raise ValueError("Planned Duration (PD) must be provided")
        
        # Get forecasted ES values
        forecast = self.forecasts[model_key]
        future_times = forecast['future_times']
        future_es = forecast['future_values']
        
        # Find when ES reaches PD
        for i, es in enumerate(future_es):
            if es >= pd:
                # Interpolate to find more precise completion time
                if i == 0:
                    # ES already reached PD in the first forecasted period
                    completion_time = future_times[0]
                else:
                    # Linear interpolation
                    t_prev = future_times[i-1]
                    t_curr = future_times[i]
                    es_prev = future_es[i-1]
                    es_curr = es
                    
                    # Interpolate: t = t_prev + (pd - es_prev) / (es_curr - es_prev) * (t_curr - t_prev)
                    if es_curr == es_prev:  # Avoid division by zero
                        completion_time = t_prev
                    else:
                        completion_time = t_prev + (pd - es_prev) / (es_curr - es_prev) * (t_curr - t_prev)
                        
                # Convert to date if current_date is provided
                if current_date is not None:
                    if period_unit == 'day':
                        delta = timedelta(days=completion_time - self.time_series['time'][-1])
                    elif period_unit == 'week':
                        delta = timedelta(weeks=completion_time - self.time_series['time'][-1])
                    else:  # month
                        # Approximate months as 30.44 days
                        delta = timedelta(days=30.44 * (completion_time - self.time_series['time'][-1]))
                        
                    completion_date = current_date + delta
                    
                    # Simple confidence interval (±10% of remaining time)
                    remaining_time = completion_time - self.time_series['time'][-1]
                    confidence_interval = timedelta(days=0.1 * remaining_time * (30.44 if period_unit == 'month' else 7 if period_unit == 'week' else 1))
                    
                    return (completion_date, confidence_interval)
                else:
                    # Return time periods
                    confidence_interval = 0.1 * (completion_time - self.time_series['time'][-1])
                    return (completion_time, confidence_interval)
        
        # If ES never reaches PD in the forecast horizon
        return (None, None)
    
    def find_best_model(self, metric='mae', target='es'):
        """
        Find the best model based on the specified metric.
        
        Parameters:
        -----------
        metric : str
            Metric to use for comparison ('mae', 'mse', or 'r2')
        target : str
            Target variable ('es' or 'spi_t')
            
        Returns:
        --------
        str: Key of the best model
        """
        best_score = float('inf') if metric in ['mae', 'mse'] else -float('inf')
        best_model = None
        
        for key, metrics in self.metrics.items():
            if key in self.forecasts and self.forecasts[key]['target'] == target:
                score = metrics.get(metric)
                if score is not None and not np.isnan(score):
                    if (metric in ['mae', 'mse'] and score < best_score) or (metric == 'r2' and score > best_score):
                        best_score = score
                        best_model = key
        
        self.best_model = best_model
        return best_model
    
    def get_all_completion_forecasts(self, pd=None, current_date=None, period_unit='month'):
        """
        Get completion forecasts from all models forecasting ES.
        
        Parameters:
        -----------
        pd : float or None
            Planned Duration. If None, use the one from time_series
        current_date : datetime or None
            Current date. If None, use today's date
        period_unit : str
            Unit of time periods ('day', 'week', 'month')
            
        Returns:
        --------
        dict: Dictionary with model keys and completion forecasts
        """
        forecasts = {}
        
        for key in self.forecasts:
            if self.forecasts[key]['target'] == 'es':
                try:
                    completion, confidence = self.predict_completion_date(key, pd, current_date, period_unit)
                    forecasts[key] = (completion, confidence)
                except Exception as e:
                    print(f"Error predicting completion with {key}: {e}")
                    forecasts[key] = (None, None)
        
        return forecasts
    
    def plot_forecasts(self, target='es', include_models=None, figsize=(12, 6)):
        """
        Plot forecasts from all or selected models.
        
        Parameters:
        -----------
        target : str
            Target variable ('es' or 'spi_t')
        include_models : list or None
            List of model keys to include. If None, include all models forecasting target
        figsize : tuple
            Figure size
            
        Returns:
        --------
        fig, ax: Figure and axis objects
        """
        # Set up plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot original data
        original_time = self.time_series['time']
        original_series = self.time_series['es'] if target == 'es' else self.time_series['spi_t']
        ax.plot(original_time, original_series, 'ko-', label='Actual', linewidth=2)
        
        # Determine which models to include
        if include_models is None:
            include_models = [k for k in self.forecasts if self.forecasts[k]['target'] == target]
        else:
            include_models = [k for k in include_models if k in self.forecasts and self.forecasts[k]['target'] == target]
        
        # Colors for different models
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
        
        # Plot each forecast
        for i, key in enumerate(include_models):
            forecast = self.forecasts[key]
            color = colors[i % len(colors)]
            
            # Split into historical and future
            n_actual = len(original_time)
            historical_times = forecast['times'][:n_actual]
            historical_values = forecast['values'][:n_actual]
            future_times = forecast['times'][n_actual:]
            future_values = forecast['values'][n_actual:]
            
            # Plot predictions on historical data (dashed)
            mask = ~np.isnan(historical_values)
            if np.any(mask):
                ax.plot(historical_times[mask], historical_values[mask], '--', color=color, alpha=0.7)
                
            # Plot forecasts (solid)
            ax.plot(future_times, future_values, '-', color=color, linewidth=2, label=f"{key.replace('_', ' ').title()}")
        
        # Add planned duration line if plotting ES
        if target == 'es' and self.time_series['pd'] is not None:
            pd_value = self.time_series['pd']
            ax.axhline(y=pd_value, linestyle=':', color='black', alpha=0.7, label='Planned Duration')
        
        # Add schedule performance reference line if plotting SPI(t)
        if target == 'spi_t':
            ax.axhline(y=1.0, linestyle=':', color='black', alpha=0.7, label='On Schedule (SPI(t)=1)')
        
        # Customize plot
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Earned Schedule' if target == 'es' else 'Schedule Performance Index (time)')
        ax.set_title(f"{'Earned Schedule' if target == 'es' else 'SPI(t)'} Forecast")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return fig, ax
