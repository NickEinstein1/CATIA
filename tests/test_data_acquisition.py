"""
Unit tests for data acquisition module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_acquisition import DataAcquisition, fetch_all_data

class TestDataAcquisition:
    """Test cases for DataAcquisition class."""
    
    @pytest.fixture
    def da(self):
        """Create DataAcquisition instance."""
        return DataAcquisition(use_mock_data=True)
    
    def test_initialization(self, da):
        """Test DataAcquisition initialization."""
        assert da.use_mock_data == True
        assert da.session is not None
    
    def test_fetch_climate_data(self, da):
        """Test climate data fetching."""
        df = da.fetch_climate_data("US_Gulf_Coast", "2020-01-01", "2023-12-31")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'temperature' in df.columns
        assert 'precipitation' in df.columns
        assert 'wind_speed' in df.columns
        assert 'sea_level_pressure' in df.columns
        assert 'humidity' in df.columns
    
    def test_fetch_socioeconomic_data(self, da):
        """Test socioeconomic data fetching."""
        df = da.fetch_socioeconomic_data("US_Gulf_Coast")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'population_density' in df.columns
        assert 'gdp_per_capita' in df.columns
        assert 'infrastructure_index' in df.columns
        assert 'poverty_rate' in df.columns
    
    def test_fetch_historical_events(self, da):
        """Test historical events fetching."""
        df = da.fetch_historical_events("US_Gulf_Coast", "hurricane")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'year' in df.columns
        assert 'event_type' in df.columns
        assert 'loss_usd' in df.columns
    
    def test_validate_data(self, da):
        """Test data validation."""
        df = da.fetch_climate_data("US_Gulf_Coast", "2020-01-01", "2023-12-31")
        cleaned_df, report = da.validate_data(df, "climate")
        
        assert isinstance(cleaned_df, pd.DataFrame)
        assert isinstance(report, dict)
        assert 'original_rows' in report
        assert 'missing_values' in report
        assert 'outliers_removed' in report
    
    def test_fetch_all_data(self):
        """Test fetching all data."""
        data = fetch_all_data("US_Gulf_Coast", use_mock=True)
        
        assert 'climate' in data
        assert 'socioeconomic' in data
        assert 'historical_events' in data
        assert len(data['climate']) > 0
        assert len(data['socioeconomic']) > 0
        assert len(data['historical_events']) > 0
    
    def test_climate_data_ranges(self, da):
        """Test climate data is within reasonable ranges."""
        df = da.fetch_climate_data("US_Gulf_Coast", "2020-01-01", "2023-12-31")
        
        # Temperature: -50 to 60 Celsius
        assert df['temperature'].min() >= -50
        assert df['temperature'].max() <= 60
        
        # Precipitation: 0 to 500 mm
        assert df['precipitation'].min() >= 0
        assert df['precipitation'].max() <= 500
        
        # Wind speed: 0 to 200 km/h
        assert df['wind_speed'].min() >= 0
        assert df['wind_speed'].max() <= 200
        
        # Humidity: 0 to 100%
        assert df['humidity'].min() >= 0
        assert df['humidity'].max() <= 100
    
    def test_socioeconomic_data_ranges(self, da):
        """Test socioeconomic data is within reasonable ranges."""
        df = da.fetch_socioeconomic_data("US_Gulf_Coast")
        
        # Population density: 0 to 10000 people/km²
        assert df['population_density'].min() >= 0
        assert df['population_density'].max() <= 10000
        
        # GDP per capita: 0 to 200000 USD
        assert df['gdp_per_capita'].min() >= 0
        assert df['gdp_per_capita'].max() <= 200000
        
        # Infrastructure index: 0 to 1
        assert df['infrastructure_index'].min() >= 0
        assert df['infrastructure_index'].max() <= 1
        
        # Poverty rate: 0 to 1
        assert df['poverty_rate'].min() >= 0
        assert df['poverty_rate'].max() <= 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

