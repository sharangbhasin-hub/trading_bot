"""
Data Freshness Manager - Ensures all data is current
"""
from datetime import datetime, timedelta
from typing import Dict, Optional
import streamlit as st

class DataFreshnessManager:
    """Manages data freshness across the application"""
    
    def __init__(self):
        self.max_age_seconds = {
            'options_chain': 60,      # 1 minute
            'spot_price': 30,         # 30 seconds
            'historical_data': 300,   # 5 minutes
            'analysis': 60,           # 1 minute
            'trend_analysis': 120     # 2 minutes
        }
    
    def is_data_fresh(self, data_type: str, timestamp: Optional[datetime]) -> bool:
        """Check if data is still fresh"""
        if timestamp is None:
            return False
        
        max_age = self.max_age_seconds.get(data_type, 60)
        age_seconds = (datetime.now() - timestamp).total_seconds()
        
        return age_seconds < max_age
    
    def mark_data_stale(self, data_type: str):
        """Force data to be refreshed"""
        key = f'{data_type}_timestamp'
        if key in st.session_state:
            del st.session_state[key]
    
    def mark_all_stale(self):
        """Force all data to refresh"""
        for data_type in self.max_age_seconds.keys():
            self.mark_data_stale(data_type)
    
    def get_data_age(self, timestamp: Optional[datetime]) -> Optional[int]:
        """Get age of data in seconds"""
        if timestamp is None:
            return None
        return int((datetime.now() - timestamp).total_seconds())
