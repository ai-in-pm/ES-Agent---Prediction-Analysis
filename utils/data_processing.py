import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ESDataProcessor:
    """
    A class for processing and validating Earned Schedule project data.
    """
    
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        self.periods = None
        self.period_unit = None
        self.has_manual_inputs = False
        
    def load_excel(self, file_path, sheet_name=None):
        """
        Load data from an Excel spreadsheet.
        
        Parameters:
        -----------
        file_path : str
            Path to the Excel file
        sheet_name : str or None
            Name of the sheet to load. If None, tries to find the sheet with data
            
        Returns:
        --------
        pd.DataFrame: The processed data
        """
        # Read Excel file
        if sheet_name is None:
            # Try to find a suitable sheet
            xls = pd.ExcelFile(file_path)
            for sheet in xls.sheet_names:
                # Skip sheets with 'description' or similar in the name
                if 'description' in sheet.lower() or 'readme' in sheet.lower():
                    continue
                # Try to load this sheet
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet)
                    # Check if this looks like project data
                    potential_columns = ['PV', 'EV', 'AC', 'Date', 'Time', 'Period']
                    if any(col in df.columns for col in potential_columns):
                        sheet_name = sheet
                        break
                except Exception:
                    continue
                    
            if sheet_name is None:
                # No suitable sheet found, try to read all sheets
                try:
                    all_sheets = pd.read_excel(file_path, sheet_name=None)
                    if all_sheets:
                        # Use the first sheet
                        sheet_name = list(all_sheets.keys())[0]
                        df = all_sheets[sheet_name]
                    else:
                        raise ValueError("No sheets found in the Excel file")
                except Exception as e:
                    raise ValueError(f"Error reading Excel file: {e}")
            else:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            # Load specified sheet
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        self.raw_data = df
        
        # Try to identify the structure and extract relevant data
        return self._process_excel_data(df)
    
    def _process_excel_data(self, df):
        """
        Process the raw Excel data to extract project metrics.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw data from Excel
            
        Returns:
        --------
        pd.DataFrame: The processed data with required columns
        """
        # First, try to identify time-related columns
        time_columns = [col for col in df.columns if any(s in str(col).lower() for s in ['time', 'period', 'date', 'month', 'week'])]
        
        # Look for standard EVM columns
        pv_columns = [col for col in df.columns if 'pv' in str(col).lower() or 'planned value' in str(col).lower()]
        ev_columns = [col for col in df.columns if 'ev' in str(col).lower() or 'earned value' in str(col).lower()]
        ac_columns = [col for col in df.columns if 'ac' in str(col).lower() or 'actual cost' in str(col).lower()]
        
        # Look for ES-specific columns
        es_columns = [col for col in df.columns if 'es' in str(col).lower() or 'earned schedule' in str(col).lower()]
        spi_t_columns = [col for col in df.columns if 'spi(t)' in str(col).lower() or 'spi-t' in str(col).lower() or 'spi_t' in str(col).lower()]
        pd_columns = [col for col in df.columns if 'pd' in str(col).lower() or 'planned duration' in str(col).lower()]
        at_columns = [col for col in df.columns if 'at' in str(col).lower() or 'actual time' in str(col).lower()]
        
        # If we couldn't find columns by name, look for patterns in data
        if not time_columns and not pv_columns and not ev_columns:
            # Look for numeric columns that could be data
            numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
            
            if numeric_columns:
                # If there are numeric columns, use them and try to determine the structure
                # Assume first numeric column might be time/period
                if len(numeric_columns) >= 1:
                    time_columns = [numeric_columns[0]]
                # If we have more numeric columns, assume they could be PV, EV, AC
                if len(numeric_columns) >= 4:
                    pv_columns = [numeric_columns[1]]
                    ev_columns = [numeric_columns[2]]
                    ac_columns = [numeric_columns[3]]
        
        # If we still don't have enough information, try another approach
        if not (time_columns and pv_columns and ev_columns):
            # Look for tables within the workbook - sometimes data is in tables with headers in rows
            potential_headers = df.apply(lambda row: row.astype(str).str.contains('pv|planned|ev|earned|ac|actual', case=False).any(), axis=1)
            header_rows = potential_headers[potential_headers].index.tolist()
            
            if header_rows:
                # Use the first identified header row
                header_row = header_rows[0]
                df_reheaded = pd.DataFrame(df.values[header_row+1:], columns=df.values[header_row])
                
                # Try to find columns again
                time_columns = [col for col in df_reheaded.columns if any(s in str(col).lower() for s in ['time', 'period', 'date', 'month', 'week'])]
                pv_columns = [col for col in df_reheaded.columns if 'pv' in str(col).lower() or 'planned value' in str(col).lower()]
                ev_columns = [col for col in df_reheaded.columns if 'ev' in str(col).lower() or 'earned value' in str(col).lower()]
                ac_columns = [col for col in df_reheaded.columns if 'ac' in str(col).lower() or 'actual cost' in str(col).lower()]
                
                if time_columns and pv_columns and ev_columns:
                    df = df_reheaded
        
        # If we still don't have the required columns, raise an error
        if not time_columns or not pv_columns or not ev_columns:
            raise ValueError("Could not identify required columns (time, PV, EV) in the Excel file")
        
        # Select the first identified column for each metric
        time_col = time_columns[0]
        pv_col = pv_columns[0]
        ev_col = ev_columns[0]
        ac_col = ac_columns[0] if ac_columns else None
        es_col = es_columns[0] if es_columns else None
        spi_t_col = spi_t_columns[0] if spi_t_columns else None
        pd_col = pd_columns[0] if pd_columns else None
        at_col = at_columns[0] if at_columns else None
        
        # Create a new dataframe with the required columns
        data = {}
        
        # Extract time/period data
        time_data = df[time_col].copy()
        
        # If time is a date, convert to period numbers
        if pd.api.types.is_datetime64_any_dtype(time_data) or pd.api.types.is_string_dtype(time_data):
            try:
                time_data = pd.to_datetime(time_data)
                self.period_unit = 'date'
                
                # Calculate periods based on frequency (assumed to be monthly if not daily or weekly)
                first_date = time_data.min()
                
                # Determine the period unit - daily, weekly, or monthly
                day_diffs = [(date - first_date).days for date in time_data if not pd.isna(date)]
                if max(day_diffs) > 0:
                    # Check if periods are approximately weekly
                    if all(abs(diff % 7) < 2 for diff in day_diffs if diff > 0):
                        self.period_unit = 'week'
                        periods = [(date - first_date).days / 7 + 1 for date in time_data]
                    # Check if periods are approximately monthly
                    elif all(abs(diff % 30) < 5 for diff in day_diffs if diff > 0):
                        self.period_unit = 'month'
                        periods = [(date - first_date).days / 30.44 + 1 for date in time_data]
                    else:
                        self.period_unit = 'day'
                        periods = [(date - first_date).days + 1 for date in time_data]
                else:
                    # Default to months if we can't determine
                    self.period_unit = 'month'
                    periods = range(1, len(time_data) + 1)
                    
                data['Date'] = time_data
                data['Time'] = periods
            except:
                # If conversion fails, treat as numeric periods
                self.period_unit = 'period'
                data['Time'] = pd.to_numeric(time_data, errors='coerce')
        else:
            # Numeric periods
            self.period_unit = 'period'
            data['Time'] = pd.to_numeric(time_data, errors='coerce')
        
        # Extract other data
        data['PV'] = pd.to_numeric(df[pv_col], errors='coerce')
        data['EV'] = pd.to_numeric(df[ev_col], errors='coerce')
        
        if ac_col:
            data['AC'] = pd.to_numeric(df[ac_col], errors='coerce')
            
        if es_col:
            data['ES'] = pd.to_numeric(df[es_col], errors='coerce')
        
        if spi_t_col:
            data['SPI(t)'] = pd.to_numeric(df[spi_t_col], errors='coerce')
            
        if pd_col:
            pd_value = pd.to_numeric(df[pd_col], errors='coerce')
            if isinstance(pd_value, pd.Series):
                # If PD is a series, take the first non-NaN value
                pd_value = pd_value.dropna().iloc[0] if not pd_value.dropna().empty else None
            data['PD'] = pd_value
        
        if at_col:
            at_value = pd.to_numeric(df[at_col], errors='coerce')
            if isinstance(at_value, pd.Series):
                # If AT is a series, take the last non-NaN value
                at_value = at_value.dropna().iloc[-1] if not at_value.dropna().empty else None
            data['AT'] = at_value
        
        # Create DataFrame
        processed_df = pd.DataFrame(data)
        
        # Drop rows with NaN in critical columns
        processed_df = processed_df.dropna(subset=['Time', 'PV', 'EV'])
        
        # Sort by time
        processed_df = processed_df.sort_values('Time').reset_index(drop=True)
        
        self.processed_data = processed_df
        self.periods = processed_df['Time'].max()
        
        return processed_df
    
    def update_with_manual_input(self, manual_input):
        """
        Update the dataset with manual input data.
        
        Parameters:
        -----------
        manual_input : dict
            Dictionary with manual input values. Keys should match column names in processed_data
            
        Returns:
        --------
        pd.DataFrame: The updated processed data
        """
        if self.processed_data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        # Create a copy of the processed data
        updated_data = self.processed_data.copy()
        
        # Get the time period for the manual input
        if 'Time' in manual_input:
            time_period = manual_input['Time']
        elif self.periods is not None:
            # If time period is not provided, use the next period
            time_period = self.periods + 1
        else:
            time_period = 1
            
        # Check if the time period already exists in the data
        if time_period in updated_data['Time'].values:
            # Update existing row
            row_idx = updated_data[updated_data['Time'] == time_period].index[0]
            for key, value in manual_input.items():
                if key in updated_data.columns:
                    updated_data.at[row_idx, key] = value
        else:
            # Create a new row
            new_row = {col: np.nan for col in updated_data.columns}
            new_row['Time'] = time_period
            
            # Add date if applicable
            if 'Date' in updated_data.columns:
                if 'Date' in manual_input:
                    new_row['Date'] = manual_input['Date']
                else:
                    # Calculate a new date based on the period unit
                    last_date = updated_data['Date'].iloc[-1]
                    if self.period_unit == 'day':
                        new_row['Date'] = last_date + timedelta(days=1)
                    elif self.period_unit == 'week':
                        new_row['Date'] = last_date + timedelta(weeks=1)
                    else:  # month or period
                        # Approximate a month as 30.44 days
                        new_row['Date'] = last_date + timedelta(days=30.44)
            
            # Update with manual input values
            for key, value in manual_input.items():
                if key in new_row:
                    new_row[key] = value
            
            # Append the new row
            updated_data = updated_data.append(new_row, ignore_index=True)
        
        # Sort by time
        updated_data = updated_data.sort_values('Time').reset_index(drop=True)
        
        self.processed_data = updated_data
        self.periods = updated_data['Time'].max()
        self.has_manual_inputs = True
        
        return updated_data
    
    def get_latest_metrics(self):
        """
        Get the latest project metrics from the data.
        
        Returns:
        --------
        dict: Dictionary with latest metric values
        """
        if self.processed_data is None or len(self.processed_data) == 0:
            return {}
            
        # Get the latest row
        latest = self.processed_data.iloc[-1].to_dict()
        
        # Add derived metrics if not present
        if 'PD' not in latest or pd.isna(latest['PD']):
            # If PD is not provided, estimate it from the data
            if 'PV' in latest:
                # Rough estimate: assume last PV is 100% of planned budget
                latest['PD'] = self.processed_data['Time'].max()
        
        if 'AT' not in latest or pd.isna(latest['AT']):
            # If AT is not provided, use the latest time period
            latest['AT'] = latest['Time']
        
        if 'ES' not in latest or pd.isna(latest['ES']):
            # If ES is not provided, we'll need to calculate it using the ES calculator
            pass
        
        return latest
    
    def validate_data(self):
        """
        Validate the processed data for inconsistencies or issues.
        
        Returns:
        --------
        tuple: (is_valid, issues)
        """
        if self.processed_data is None or len(self.processed_data) == 0:
            return (False, ["No data loaded or empty dataset"])
            
        issues = []
        
        # Check for missing values in critical columns
        for col in ['Time', 'PV', 'EV']:
            if col in self.processed_data.columns and self.processed_data[col].isna().any():
                issues.append(f"Missing values in {col} column")
        
        # Check for decreasing PV (should be non-decreasing)
        if 'PV' in self.processed_data.columns and len(self.processed_data) > 1:
            if not (self.processed_data['PV'].diff().dropna() >= 0).all():
                issues.append("Planned Value (PV) should be non-decreasing over time")
        
        # Check for unrealistic EV values (EV should not exceed PV by too much)
        if 'PV' in self.processed_data.columns and 'EV' in self.processed_data.columns:
            if (self.processed_data['EV'] > self.processed_data['PV'] * 1.1).any():
                issues.append("Some Earned Value (EV) values exceed Planned Value (PV) by more than 10%")
        
        # Check if we have enough data for meaningful analysis
        if len(self.processed_data) < 3:
            issues.append("Limited data available (less than 3 periods), forecasts may be less reliable")
        
        return (len(issues) == 0, issues)
