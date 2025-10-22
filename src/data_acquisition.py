"""
Data Acquisition Module for CATIA
Fetches real-time and historical climate and socioeconomic data.
Implements robust error handling and data validation.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import API_CONFIG, DATA_CONFIG, LOGGING_CONFIG

# Configure logging
logging.basicConfig(level=LOGGING_CONFIG["level"], format=LOGGING_CONFIG["format"])
logger = logging.getLogger(__name__)

# ============================================================================
# DATA ACQUISITION CLASS
# ============================================================================

class DataAcquisition:
    """Handles data fetching from multiple sources with error handling."""
    
    def __init__(self, use_mock_data: bool = True):
        """
        Initialize DataAcquisition.
        
        Args:
            use_mock_data: If True, use mock data; otherwise fetch from APIs
        """
        self.use_mock_data = use_mock_data
        self.session = self._create_session()
        logger.info(f"DataAcquisition initialized (mock_data={use_mock_data})")
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy."""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def fetch_climate_data(self, region: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch climate data for a region.
        
        Args:
            region: Geographic region (e.g., "US_Gulf_Coast")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with climate variables
        """
        if self.use_mock_data:
            return self._generate_mock_climate_data(region, start_date, end_date)
        
        try:
            logger.info(f"Fetching climate data for {region} from {start_date} to {end_date}")
            # Real API call would go here
            # For now, return mock data
            return self._generate_mock_climate_data(region, start_date, end_date)
        except Exception as e:
            logger.error(f"Error fetching climate data: {e}")
            raise
    
    def fetch_socioeconomic_data(self, region: str) -> pd.DataFrame:
        """
        Fetch socioeconomic data for a region.
        
        Args:
            region: Geographic region
        
        Returns:
            DataFrame with socioeconomic variables
        """
        if self.use_mock_data:
            return self._generate_mock_socioeconomic_data(region)
        
        try:
            logger.info(f"Fetching socioeconomic data for {region}")
            # Real API call would go here
            return self._generate_mock_socioeconomic_data(region)
        except Exception as e:
            logger.error(f"Error fetching socioeconomic data: {e}")
            raise
    
    def fetch_historical_events(self, region: str, event_type: str) -> pd.DataFrame:
        """
        Fetch historical catastrophe events.
        
        Args:
            region: Geographic region
            event_type: Type of event (hurricane, flood, wildfire, etc.)
        
        Returns:
            DataFrame with historical events
        """
        if self.use_mock_data:
            return self._generate_mock_historical_events(region, event_type)
        
        try:
            logger.info(f"Fetching {event_type} events for {region}")
            return self._generate_mock_historical_events(region, event_type)
        except Exception as e:
            logger.error(f"Error fetching historical events: {e}")
            raise
    
    # ========================================================================
    # MOCK DATA GENERATION
    # ========================================================================
    
    def _generate_mock_climate_data(self, region: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate realistic mock climate data."""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start=start, end=end, freq='D')
        
        n = len(dates)
        data = {
            'date': dates,
            'temperature': np.random.normal(20, 5, n),  # Celsius
            'precipitation': np.abs(np.random.normal(5, 10, n)),  # mm
            'wind_speed': np.abs(np.random.normal(10, 5, n)),  # km/h
            'sea_level_pressure': np.random.normal(1013, 5, n),  # hPa
            'humidity': np.clip(np.random.normal(65, 15, n), 0, 100),  # %
            'region': region
        }
        
        df = pd.DataFrame(data)
        logger.info(f"Generated mock climate data: {len(df)} records for {region}")
        return df
    
    def _generate_mock_socioeconomic_data(self, region: str) -> pd.DataFrame:
        """Generate realistic mock socioeconomic data."""
        data = {
            'region': [region],
            'population_density': [np.random.uniform(50, 500)],  # people/km²
            'gdp_per_capita': [np.random.uniform(5000, 50000)],  # USD
            'infrastructure_index': [np.random.uniform(0.3, 0.9)],  # 0-1 scale
            'poverty_rate': [np.random.uniform(0.05, 0.3)]  # 0-1 scale
        }
        
        df = pd.DataFrame(data)
        logger.info(f"Generated mock socioeconomic data for {region}")
        return df
    
    def _generate_mock_historical_events(self, region: str, event_type: str) -> pd.DataFrame:
        """Generate realistic mock historical event data."""
        n_events = np.random.randint(5, 20)
        years = np.random.choice(range(2000, 2024), n_events)
        
        data = {
            'year': years,
            'event_type': event_type,
            'region': region,
            'magnitude': np.random.uniform(1, 10, n_events),  # Saffir-Simpson scale
            'loss_usd': np.random.lognormal(15, 2, n_events),  # Log-normal distribution
            'affected_population': np.random.randint(1000, 1000000, n_events)
        }
        
        df = pd.DataFrame(data).sort_values('year')
        logger.info(f"Generated {len(df)} mock {event_type} events for {region}")
        return df
    
    # ========================================================================
    # DATA VALIDATION
    # ========================================================================
    
    def validate_data(self, df: pd.DataFrame, data_type: str = "climate") -> Tuple[pd.DataFrame, Dict]:
        """
        Validate and clean data.
        
        Args:
            df: DataFrame to validate
            data_type: Type of data (climate, socioeconomic, events)
        
        Returns:
            Tuple of (cleaned_df, validation_report)
        """
        report = {
            'original_rows': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'outliers_removed': 0,
            'data_type': data_type
        }
        
        # Check for missing values
        if DATA_CONFIG["data_validation"]["check_missing_values"]:
            df = df.dropna()
            report['rows_after_missing_removal'] = len(df)
        
        # Check for outliers
        if DATA_CONFIG["data_validation"]["check_outliers"]:
            threshold = DATA_CONFIG["data_validation"]["outlier_threshold"]
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                mean = df[col].mean()
                std = df[col].std()
                mask = np.abs((df[col] - mean) / std) <= threshold
                outliers = len(df) - mask.sum()
                report['outliers_removed'] += outliers
                df = df[mask]
        
        logger.info(f"Data validation report: {report}")
        return df, report

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def fetch_all_data(region: str, use_mock: bool = True) -> Dict:
    """
    Fetch all required data for a region.
    
    Args:
        region: Geographic region
        use_mock: Use mock data if True
    
    Returns:
        Dictionary with climate, socioeconomic, and historical event data
    """
    da = DataAcquisition(use_mock_data=use_mock)
    
    # Fetch data
    climate_data = da.fetch_climate_data(region, "2020-01-01", "2023-12-31")
    socioeconomic_data = da.fetch_socioeconomic_data(region)
    historical_events = da.fetch_historical_events(region, "hurricane")
    
    # Validate data
    climate_data, _ = da.validate_data(climate_data, "climate")
    socioeconomic_data, _ = da.validate_data(socioeconomic_data, "socioeconomic")
    historical_events, _ = da.validate_data(historical_events, "events")
    
    return {
        'climate': climate_data,
        'socioeconomic': socioeconomic_data,
        'historical_events': historical_events
    }

if __name__ == "__main__":
    # Example usage
    data = fetch_all_data("US_Gulf_Coast", use_mock=True)
    print("\nClimate Data Sample:")
    print(data['climate'].head())
    print("\nSocioeconomic Data:")
    print(data['socioeconomic'])
    print("\nHistorical Events Sample:")
    print(data['historical_events'].head())

