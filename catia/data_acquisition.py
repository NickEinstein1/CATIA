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

from catia.config import API_CONFIG, DATA_CONFIG, LOGGING_CONFIG, PERIL_CONFIG, DEFAULT_PERILS

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
        """Generate realistic mock historical event data based on peril configuration."""
        peril_config = PERIL_CONFIG.get(event_type, {})

        # Determine number of events based on peril frequency
        base_freq = peril_config.get('frequency_base', 0.5)
        years_of_history = 24  # 2000-2024
        expected_events = int(base_freq * years_of_history)
        n_events = np.random.randint(max(3, expected_events - 5), expected_events + 10)

        # Generate years with seasonality consideration
        seasonality = peril_config.get('seasonality', list(range(1, 13)))
        years = np.random.choice(range(2000, 2024), n_events)
        months = np.random.choice(seasonality, n_events)

        # Get severity parameters from config
        severity_params = peril_config.get('severity_params', {'mu': 15, 'sigma': 2})

        # Generate magnitude based on peril type
        if event_type == 'hurricane':
            magnitude = np.random.uniform(1, 5, n_events)  # Saffir-Simpson 1-5
        elif event_type == 'earthquake':
            magnitude = np.random.uniform(4, 9, n_events)  # Richter scale
        elif event_type == 'flood':
            magnitude = np.random.uniform(1, 5, n_events)  # Flood severity 1-5
        elif event_type == 'wildfire':
            magnitude = np.random.uniform(1, 5, n_events)  # Fire severity 1-5
        else:
            magnitude = np.random.uniform(1, 10, n_events)

        data = {
            'year': years,
            'month': months,
            'event_type': event_type,
            'region': region,
            'magnitude': magnitude,
            'loss_usd': np.random.lognormal(severity_params['mu'], severity_params['sigma'], n_events),
            'affected_population': np.random.randint(1000, 1000000, n_events),
            'peril_name': peril_config.get('name', event_type.title())
        }

        df = pd.DataFrame(data).sort_values(['year', 'month'])
        logger.info(f"Generated {len(df)} mock {event_type} events for {region}")
        return df

    def fetch_multi_peril_events(self, region: str, perils: List[str] = None) -> pd.DataFrame:
        """
        Fetch historical events for multiple peril types.

        Args:
            region: Geographic region
            perils: List of peril types (uses DEFAULT_PERILS if None)

        Returns:
            DataFrame with historical events for all perils
        """
        perils = perils or DEFAULT_PERILS
        all_events = []

        for peril in perils:
            if peril in PERIL_CONFIG:
                events = self.fetch_historical_events(region, peril)
                all_events.append(events)
            else:
                logger.warning(f"Unknown peril type: {peril}")

        if all_events:
            combined = pd.concat(all_events, ignore_index=True)
            combined = combined.sort_values(['year', 'month']).reset_index(drop=True)
            logger.info(f"Fetched {len(combined)} total events across {len(perils)} perils for {region}")
            return combined

        return pd.DataFrame()
    
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
        
        # Check for outliers (only if we have enough rows for statistics)
        if DATA_CONFIG["data_validation"]["check_outliers"] and len(df) > 2:
            threshold = DATA_CONFIG["data_validation"]["outlier_threshold"]
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                mean = df[col].mean()
                std = df[col].std()
                # Skip outlier detection if std is 0 (all values identical)
                if std > 0:
                    mask = np.abs((df[col] - mean) / std) <= threshold
                    outliers = len(df) - mask.sum()
                    report['outliers_removed'] += outliers
                    df = df[mask]
        
        logger.info(f"Data validation report: {report}")
        return df, report

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def fetch_all_data(region: str, use_mock: bool = True, perils: List[str] = None) -> Dict:
    """
    Fetch all required data for a region.

    Args:
        region: Geographic region
        use_mock: Use mock data if True
        perils: List of peril types to fetch (uses DEFAULT_PERILS if None)

    Returns:
        Dictionary with climate, socioeconomic, and historical event data
    """
    da = DataAcquisition(use_mock_data=use_mock)
    perils = perils or DEFAULT_PERILS

    # Fetch data
    climate_data = da.fetch_climate_data(region, "2020-01-01", "2023-12-31")
    socioeconomic_data = da.fetch_socioeconomic_data(region)

    # Fetch multi-peril historical events
    historical_events = da.fetch_multi_peril_events(region, perils)

    # Also fetch individual peril data for backward compatibility
    events_by_peril = {}
    for peril in perils:
        events_by_peril[peril] = da.fetch_historical_events(region, peril)

    # Validate data
    climate_data, _ = da.validate_data(climate_data, "climate")
    socioeconomic_data, _ = da.validate_data(socioeconomic_data, "socioeconomic")
    historical_events, _ = da.validate_data(historical_events, "events")

    return {
        'climate': climate_data,
        'socioeconomic': socioeconomic_data,
        'historical_events': historical_events,
        'events_by_peril': events_by_peril,
        'perils_analyzed': perils
    }


def fetch_single_peril_data(region: str, peril: str, use_mock: bool = True) -> Dict:
    """
    Fetch data for a single peril type (backward compatible).

    Args:
        region: Geographic region
        peril: Peril type (hurricane, flood, wildfire, earthquake)
        use_mock: Use mock data if True

    Returns:
        Dictionary with climate, socioeconomic, and historical event data
    """
    da = DataAcquisition(use_mock_data=use_mock)

    climate_data = da.fetch_climate_data(region, "2020-01-01", "2023-12-31")
    socioeconomic_data = da.fetch_socioeconomic_data(region)
    historical_events = da.fetch_historical_events(region, peril)

    climate_data, _ = da.validate_data(climate_data, "climate")
    socioeconomic_data, _ = da.validate_data(socioeconomic_data, "socioeconomic")
    historical_events, _ = da.validate_data(historical_events, "events")

    return {
        'climate': climate_data,
        'socioeconomic': socioeconomic_data,
        'historical_events': historical_events,
        'peril': peril
    }


if __name__ == "__main__":
    # Example usage with multi-peril support
    print("=" * 60)
    print("Multi-Peril Data Acquisition Demo")
    print("=" * 60)

    data = fetch_all_data("US_Gulf_Coast", use_mock=True)

    print(f"\nPerils Analyzed: {data['perils_analyzed']}")
    print(f"\nClimate Data: {len(data['climate'])} records")
    print(f"Socioeconomic Data: {len(data['socioeconomic'])} records")
    print(f"Total Historical Events: {len(data['historical_events'])} events")

    print("\nEvents by Peril Type:")
    for peril, events in data['events_by_peril'].items():
        print(f"  {peril}: {len(events)} events")

    print("\nSample Events (All Perils):")
    print(data['historical_events'][['year', 'event_type', 'magnitude', 'loss_usd']].head(10))

