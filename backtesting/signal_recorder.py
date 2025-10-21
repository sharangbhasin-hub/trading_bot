"""
Signal Recorder
Captures and stores all signals generated during backtest
"""
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SignalRecorder:
    """
    Records all signals generated during backtesting
    """
    
    def __init__(self):
        """Initialize signal recorder"""
        self.signals = []
        self.signal_count = 0
        
    def record_signal(self, timestamp, signal_data):
        """
        Record a signal
        
        Args:
            timestamp: DateTime of signal generation
            signal_data: Dict with signal details from strategy
        """
        signal = {
            'signal_id': self.signal_count,
            'timestamp': timestamp,
            'date': timestamp.strftime('%Y-%m-%d'),
            'time': timestamp.strftime('%H:%M'),
            'strategy_name': signal_data.get('strategy_name'),
            'signal_type': signal_data.get('signal'),  # CALL or PUT
            'confidence': signal_data.get('confidence'),
            'entry_price': signal_data.get('entry_price'),
            'stop_loss': signal_data.get('stop_loss'),
            'target': signal_data.get('target'),
            'tier': signal_data.get('tier'),
            'reasoning': signal_data.get('reasoning', []),
            'candlestick_pattern': signal_data.get('candlestick_pattern'),
            'setup_complete': signal_data.get('setup_complete', False)
        }
        
        # Calculate R:R ratio
        if signal['entry_price'] and signal['stop_loss'] and signal['target']:
            risk = abs(signal['entry_price'] - signal['stop_loss'])
            reward = abs(signal['target'] - signal['entry_price'])
            signal['rr_ratio'] = reward / risk if risk > 0 else 0
        else:
            signal['rr_ratio'] = 0
        
        self.signals.append(signal)
        self.signal_count += 1
        
        logger.debug(f"Recorded signal #{self.signal_count}: {signal['strategy_name']} {signal['signal_type']} @ {signal['entry_price']}")
        
        return signal['signal_id']
    
    def get_signals_for_date(self, date_str):
        """Get all signals for a specific date"""
        return [s for s in self.signals if s['date'] == date_str]
    
    def get_signals_by_strategy(self, strategy_name):
        """Get all signals for a specific strategy"""
        return [s for s in self.signals if s['strategy_name'] == strategy_name]
    
    def get_signal_by_id(self, signal_id):
        """Get signal by ID"""
        for signal in self.signals:
            if signal['signal_id'] == signal_id:
                return signal
        return None
    
    def to_dataframe(self):
        """Convert signals to DataFrame"""
        if not self.signals:
            return pd.DataFrame()
        
        # Flatten reasoning (list to string)
        df_signals = []
        for sig in self.signals:
            sig_copy = sig.copy()
            sig_copy['reasoning'] = ' | '.join(sig_copy['reasoning']) if sig_copy['reasoning'] else ''
            df_signals.append(sig_copy)
        
        df = pd.DataFrame(df_signals)
        return df
    
    def get_summary_stats(self):
        """Get summary statistics of all signals"""
        if not self.signals:
            return {}
        
        df = self.to_dataframe()
        
        stats = {
            'total_signals': len(self.signals),
            'call_signals': len(df[df['signal_type'] == 'CALL']),
            'put_signals': len(df[df['signal_type'] == 'PUT']),
            'avg_confidence': df['confidence'].mean(),
            'avg_rr_ratio': df['rr_ratio'].mean(),
            'signals_by_tier': df.groupby('tier').size().to_dict(),
            'signals_by_strategy': df.groupby('strategy_name').size().to_dict(),
            'signals_by_date': df.groupby('date').size().to_dict()
        }
        
        return stats
    
    def reset(self):
        """Clear all recorded signals"""
        self.signals = []
        self.signal_count = 0
        logger.info("Signal recorder reset")
