from strategies.base_strategy import BaseStrategy
from detectors.crt_detector import CRTDetector
from detectors.keylevel_detector import KeyLevelDetector
from detectors.tbs_detector import TBSDetector

class StrategyCRTTBS(BaseStrategy):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.crt_detector = CRTDetector()
        self.keylevel_detector = KeyLevelDetector()
        self.tbs_detector = TBSDetector()
        self.state = 'HTF_SCANNING'
        self.htf_setup = None
        
    def generate_signals(self, df_htf, df_ltf):
        """
        Main strategy logic following state machine:
        HTF_SCANNING → HTF_SETUP_COMPLETE → LTF_MONITORING → 
        TBS_CONFIRMED → MODEL1_CONFIRMED → ENTRY_TRIGGERED
        """
        if self.state == 'HTF_SCANNING':
            return self._scan_htf(df_htf)
        elif self.state == 'LTF_MONITORING':
            return self._monitor_ltf(df_ltf)
        # ... state machine implementation
    
    def _scan_htf(self, df_htf):
        # Step 1: Find CRT candles
        crt_candles = self.crt_detector.detect_crt_candles(df_htf)
        
        if crt_candles.empty:
            return None
        
        # Step 2: Check for key levels
        for idx, crt in crt_candles.iterrows():
            keylevels = self.keylevel_detector.detect_all_keylevels(
                df_htf, crt_index=idx
            )
            
            if keylevels:
                # Step 3: Calculate 50% level
                tp1 = self.crt_detector.calculate_50_percent_level(
                    crt['high'], crt['low']
                )
                
                # Store HTF setup
                self.htf_setup = {
                    'crt_high': crt['high'],
                    'crt_low': crt['low'],
                    'tp1': tp1,
                    'tp2': crt['low'],  # For sells
                    'keylevels': keylevels,
                    'direction': 'sell'  # Determined by keylevel type
                }
                
                self.state = 'LTF_MONITORING'
                return None
        
        return None
    
    def _monitor_ltf(self, df_ltf):
        # Step 4: Wait for TBS
        tbs = self.tbs_detector.detect_tbs_pattern(
            df_ltf, 
            reference_level=self.htf_setup['crt_high'],
            direction=self.htf_setup['direction']
        )
        
        if not tbs:
            return None
        
        self.state = 'TBS_CONFIRMED'
        
        # Step 5: Find Model #1
        model1 = self.tbs_detector.detect_model1(
            df_ltf[tbs['index']:],
            direction=self.htf_setup['direction']
        )
        
        if not model1:
            return None
        
        self.state = 'MODEL1_CONFIRMED'
        
        # Step 6: Check for entry trigger
        return self._check_entry_trigger(df_ltf, model1)
    
    def _check_entry_trigger(self, df_ltf, model1):
        # Entry condition: Close beyond Model #1
        last_candle = df_ltf.iloc[-1]
        
        if self.htf_setup['direction'] == 'sell':
            if last_candle['close'] < model1['low']:
                return self._create_sell_signal(last_candle, model1)
        
        return None
    
    def _create_sell_signal(self, entry_candle, model1):
        # Calculate stop loss (exact extreme)
        stop_loss = max(
            self.htf_setup['tbs_high'],
            model1['high']
        )
        
        # Calculate risk-reward
        entry_price = entry_candle['close']
        risk = stop_loss - entry_price
        reward_tp1 = entry_price - self.htf_setup['tp1']
        reward_tp2 = entry_price - self.htf_setup['tp2']
        
        avg_reward = (reward_tp1 * 0.5) + (reward_tp2 * 0.5)
        rr_ratio = avg_reward / risk
        
        # RR filter
        if rr_ratio < self.config['min_rr_ratio']:
            return None
        
        return {
            'direction': 'sell',
            'entry': entry_price,
            'stop_loss': stop_loss,
            'tp1': self.htf_setup['tp1'],
            'tp2': self.htf_setup['tp2'],
            'position_scaling': [0.5, 0.5],  # Close 50% at each TP
            'move_to_breakeven': 'after_tp1',
            'rr_ratio': rr_ratio
        }
