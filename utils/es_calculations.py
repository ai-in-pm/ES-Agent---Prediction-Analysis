import numpy as np
import pandas as pd

class EarnedScheduleCalculator:
    """
    A class to calculate Earned Schedule (ES) metrics.
    ES is an advanced project control technique that extends traditional Earned Value Management (EVM)
    by measuring schedule performance in time units.
    """
    
    def __init__(self):
        self.data = None
        self.pv = None
        self.ev = None
        self.ac = None
        self.time = None
        self.pd = None  # Planned Duration
        self.at = None  # Actual Time
        self.es = None  # Earned Schedule
        self.ed = None  # Earned Duration
        
    def load_data(self, data, time_column='Time', pv_column='PV', ev_column='EV', ac_column='AC'):
        """
        Load data from DataFrame.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing time-phased project data
        time_column : str
            Column name for time periods
        pv_column : str
            Column name for Planned Value
        ev_column : str
            Column name for Earned Value
        ac_column : str
            Column name for Actual Cost (optional)
        """
        self.data = data.copy()
        self.time = self.data[time_column].values
        self.pv = self.data[pv_column].values
        self.ev = self.data[ev_column].values
        if ac_column in self.data.columns:
            self.ac = self.data[ac_column].values
        
    def set_parameters(self, at=None, pd=None, ed=None, es=None):
        """
        Set the ES parameters manually.
        
        Parameters:
        -----------
        at : float
            Actual Time (current time period)
        pd : float
            Planned Duration (total planned project duration)
        ed : float
            Earned Duration (if known)
        es : float
            Earned Schedule (if known)
        """
        if at is not None:
            self.at = at
        if pd is not None:
            self.pd = pd
        if ed is not None:
            self.ed = ed
        if es is not None:
            self.es = es
    
    def calculate_es(self):
        """
        Calculate Earned Schedule (ES).
        ES is the time at which the current EV should have been achieved according to the plan.
        """
        if self.es is not None:
            return self.es
            
        if self.data is None or self.ev is None or self.pv is None:
            raise ValueError("Data must be loaded first or ES must be provided manually")
            
        current_ev = self.ev[-1]  # Latest EV
        
        # Find the time period where PV is just greater than current EV
        for i in range(len(self.pv)):
            if self.pv[i] >= current_ev:
                if i == 0:
                    self.es = 0
                else:
                    # Linear interpolation to find ES
                    t_prev = self.time[i-1]
                    t_curr = self.time[i]
                    pv_prev = self.pv[i-1]
                    pv_curr = self.pv[i]
                    
                    # Linear interpolation: ES = t_prev + (current_ev - pv_prev) / (pv_curr - pv_prev) * (t_curr - t_prev)
                    if pv_curr == pv_prev:  # Avoid division by zero
                        self.es = t_prev
                    else:
                        self.es = t_prev + (current_ev - pv_prev) / (pv_curr - pv_prev) * (t_curr - t_prev)
                return self.es
                
        # If EV exceeds all PV values, ES is the max time
        self.es = self.time[-1]
        return self.es
        
    def calculate_spi_t(self):
        """
        Calculate Schedule Performance Index in time units: SPI(t) = ES / AT
        SPI(t) > 1 indicates ahead of schedule (ES > AT)
        SPI(t) < 1 indicates behind schedule (ES < AT)
        """
        if self.es is None:
            self.calculate_es()
            
        if self.at is None:
            if self.data is not None and len(self.time) > 0:
                self.at = self.time[-1]  # Use latest time period as AT
            else:
                raise ValueError("Actual Time (AT) must be provided or calculated from data")
                
        if self.at == 0:  # Avoid division by zero
            return 1.0
            
        return self.es / self.at
    
    def calculate_c(self):
        """
        Calculate C = ES / PD (fraction of the total duration earned, essentially percent time complete)
        """
        if self.es is None:
            self.calculate_es()
            
        if self.pd is None:
            raise ValueError("Planned Duration (PD) must be provided")
            
        return self.es / self.pd
        
    def calculate_tspi(self):
        """
        Calculate To Complete Schedule Performance Index: TSPI = (PD - ES) / (PD - AT)
        This is the efficiency required for the remaining duration to complete on time.
        """
        if self.es is None:
            self.calculate_es()
            
        if self.at is None or self.pd is None:
            raise ValueError("Actual Time (AT) and Planned Duration (PD) must be provided")
            
        denominator = self.pd - self.at
        
        if denominator == 0:  # Avoid division by zero
            if self.pd > self.es:
                # We've reached PD but haven't completed the work
                return float('inf')  # Impossible to recover
            else:
                # We've completed on or ahead of schedule
                return 1.0
                
        return (self.pd - self.es) / denominator
        
    def calculate_ieac_t(self, performance_factor=None):
        """
        Calculate Independent Estimate at Completion in time units (IEAC(t)).
        
        Parameters:
        -----------
        performance_factor : float or str
            Performance factor to use for calculation. Options:
            - None: Use current SPI(t)
            - 1.0: Assume future performance = planned (PF=1)
            - float: Use specified value
        
        Returns:
        --------
        float: Estimated total duration
        """
        if self.es is None:
            self.calculate_es()
            
        if self.at is None or self.pd is None:
            raise ValueError("Actual Time (AT) and Planned Duration (PD) must be provided")
            
        if performance_factor is None:
            # Use current SPI(t)
            performance_factor = self.calculate_spi_t()
        elif performance_factor == 1.0 or performance_factor == "PF=1":
            # Assume future performance = planned
            performance_factor = 1.0
            
        if performance_factor == 0:  # Avoid division by zero
            return float('inf')
            
        return self.at + (self.pd - self.es) / performance_factor
        
    def is_recovery_likely(self, threshold=1.1):
        """
        Determine if schedule recovery is likely based on TSPI and threshold.
        Typically, TSPI > 1.1 indicates recovery is unlikely.
        
        Parameters:
        -----------
        threshold : float
            Threshold for TSPI above which recovery is unlikely (default: 1.1)
            
        Returns:
        --------
        bool: True if recovery is likely, False otherwise
        """
        tspi = self.calculate_tspi()
        return tspi <= threshold
        
    def calculate_all_metrics(self):
        """
        Calculate all ES metrics and return them as a dictionary.
        
        Returns:
        --------
        dict: Dictionary containing all calculated ES metrics
        """
        if self.es is None:
            self.calculate_es()
            
        spi_t = self.calculate_spi_t()
        c = self.calculate_c()
        tspi = self.calculate_tspi()
        ieac_t_current = self.calculate_ieac_t()
        ieac_t_pf1 = self.calculate_ieac_t(1.0)
        recovery_likely = self.is_recovery_likely()
        
        return {
            'ES': self.es,
            'AT': self.at,
            'PD': self.pd,
            'ED': self.ed if self.ed is not None else self.es,
            'SPI(t)': spi_t,
            'C': c,
            'TSPI': tspi,
            'IEAC(t) - Current Performance': ieac_t_current,
            'IEAC(t) - PF=1': ieac_t_pf1,
            'Recovery Likely': recovery_likely
        }
