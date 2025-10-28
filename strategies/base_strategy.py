"""
Base Strategy Class - All strategies inherit from this
Professional Edition with Unified Stop-Loss Framework
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Tuple
import pandas as pd
from detectors.market_regime_detector import MarketRegimeDetector
from utils.dataframe_validator import DataFrameValidator


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        
        # ========== UNIFIED STOP-LOSS PARAMETERS ==========
        # These parameters ensure consistency across ALL stop calculation methods
        # Professional standard ranges based on institutional trading practices
        
        # ATR Multiplier Range (for confidence-adaptive stops)
        self.atr_multiplier_high = 2.2  # Used for low confidence (wider stops)
        self.atr_multiplier_low = 1.7   # Used for high confidence (tighter stops)
        
        # Fallback stop distance (when ATR unavailable)
        self.min_stop_pct = 0.005  # 0.5% of price (intraday standard)
        
        # Risk:Reward ratios by market regime
        self.rr_trending = 3.0   # Aggressive targets in trends
        self.rr_ranging = 2.0    # Conservative targets in ranges
        self.rr_neutral = 2.5    # Standard targets in neutral markets
        
        # Volatility thresholds (as % of price)
        self.high_volatility_threshold = 1.5  # Above = high volatility
        self.low_volatility_threshold = 0.5   # Below = low volatility
        
        # Volatility R:R adjustments
        self.high_vol_adjustment = 0.5   # Add to R:R in high volatility
        self.low_vol_adjustment = -0.5   # Subtract from R:R in low volatility
        
        # ========== EXISTING PARAMETERS (Backward Compatible) ==========
        self.min_confidence = 40       # Minimum confidence to show signal
        self.retest_required = False   # Whether retest is required
        self.min_risk_reward = 1.5     # Minimum R:R ratio (updated to professional standard)
        
        # Market analysis tools
        self.market_regime_detector = MarketRegimeDetector()
        self.df_validator = DataFrameValidator()

    def is_valid_trading_time(self, timestamp) -> bool:
        """
        Check if current time is suitable for trading
        Universal filter - works for any index
        
        Evidence from backtests:
        - 29-31% WR in closing hour (15:15-15:30) across ALL 4 tests
        - Lower liquidity and wider spreads at market close
        
        Args:
            timestamp: Can be datetime object, pandas Timestamp, or string
            
        Returns:
            bool: True if time is valid for trading
        """
        # Handle None case
        if timestamp is None:
            return True  # Allow if no timestamp provided
            
        # Convert string to datetime if needed
        if isinstance(timestamp, str):
            from datetime import datetime
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                return True  # Allow if can't parse
        
        # Extract hour and minute
        try:
            hour = timestamp.hour
            minute = timestamp.minute
        except AttributeError:
            return True  # Allow if no hour/minute attributes
        
        # FILTER: Block closing hour (15:15-15:30)
        # Evidence: 29-31% WR = guaranteed losses
        if hour == 15 and minute >= 15:
            return False
        
        return True
    
    @abstractmethod
    def analyze(self,
                df_5min: pd.DataFrame,
                df_15min: pd.DataFrame,
                df_1h: pd.DataFrame,
                df_4h: pd.DataFrame,
                spot_price: float,
                support: float,
                resistance: float,
                overall_trend: str) -> Dict:
        """
        Analyze market and return signal
        
        Returns dict with:
        {
            'signal': 'CALL' | 'PUT' | 'NO_TRADE',
            'confidence': 0-100,
            'entry_price': float,
            'stop_loss': float,
            'target': float,
            'reasoning': List[str],
            'setup_detected': bool,
            'retest_confirmed': bool,
            'candlestick_pattern': str | None
        }
        """
        pass
    
    def is_tradeable(self, result: Dict, timestamp=None) -> bool:
        """
        Check if signal meets trading criteria
        
        Args:
            result: Strategy result dict
            timestamp: Current timestamp to validate trading time
            
        Returns:
            bool: True if signal is tradeable
        """
        if result['signal'] == 'NO_TRADE':
            return False
        
        if result['confidence'] < self.min_confidence:
            return False
        
        if self.retest_required and not result['retest_confirmed']:
            return False
        
        # Time-based filter
        if timestamp is not None:
            if not self.is_valid_trading_time(timestamp):
                return False
        
        return True
    
    def calculate_risk_reward(self, entry: float, stop_loss: float, target: float) -> float:
        """
        Calculate risk:reward ratio
        
        Args:
            entry: Entry price
            stop_loss: Stop loss price
            target: Target price
            
        Returns:
            Risk:Reward ratio (e.g., 2.0 means 1:2 ratio)
        """
        risk = abs(entry - stop_loss)
        reward = abs(target - entry)
        
        if risk == 0:
            return 0.0
        
        return round(reward / risk, 2)
    
    def validate_risk_reward(self, result: Dict) -> Dict:
        """
        Validate and update result with R:R ratio
        
        Args:
            result: Strategy result dict
            
        Returns:
            Updated result dict with R:R validation
        """
        if result['signal'] == 'NO_TRADE':
            return result
        
        # Calculate R:R ratio
        rr_ratio = self.calculate_risk_reward(
            result['entry_price'],
            result['stop_loss'],
            result['target']
        )
        
        # Add R:R to result
        result['risk_reward_ratio'] = rr_ratio
        
        # Validate minimum R:R
        if rr_ratio < self.min_risk_reward:
            result['signal'] = 'NO_TRADE'
            result['confidence'] = 0
            result['reasoning'].append(
                f'❌ Risk:Reward too low: {rr_ratio:.2f}:1 (need minimum {self.min_risk_reward}:1)'
            )
        else:
            result['reasoning'].append(
                f'✅ Risk:Reward ratio: {rr_ratio:.2f}:1'
            )
        
        return result

    # ========== UNIFIED HELPER METHODS ==========
    
    def _get_atr_multiplier(self, confidence: float) -> float:
        """
        ✅ SINGLE SOURCE OF TRUTH: Calculate ATR multiplier based on confidence
        
        This ensures ALL stop calculation methods use consistent multipliers.
        
        Logic:
        - Higher confidence (e.g., 100) → Tighter stops (1.7x ATR)
        - Lower confidence (e.g., 40) → Wider stops (2.2x ATR)
        
        Args:
            confidence: Signal confidence (0-100)
            
        Returns:
            ATR multiplier (1.7-2.2x range)
        """
        # Formula: 2.2 - (confidence / 200)
        # At confidence=100: 2.2 - 0.5 = 1.7x (tight)
        # At confidence=40:  2.2 - 0.2 = 2.0x (wider)
        # At confidence=0:   2.2 - 0   = 2.2x (widest)
        return self.atr_multiplier_high - (confidence / 200)
    
    def _get_dynamic_rr_multiplier(self, 
                                   atr: Optional[float], 
                                   entry_price: float, 
                                   market_regime: str) -> float:
        """
        ✅ UNIFIED R:R CALCULATOR: Used by all stop calculation methods
        
        Adjusts risk:reward ratio based on:
        1. Market regime (trending vs ranging)
        2. Volatility (high vs low)
        
        Args:
            atr: Average True Range (optional)
            entry_price: Entry price
            market_regime: 'TRENDING' | 'RANGING' | 'NEUTRAL'
            
        Returns:
            Risk:Reward multiplier (e.g., 2.5 means 1:2.5 ratio)
        """
        # Step 1: Base R:R from market regime
        if market_regime == 'TRENDING':
            base_rr = self.rr_trending  # 3.0 - aggressive in trends
        elif market_regime == 'RANGING':
            base_rr = self.rr_ranging   # 2.0 - conservative in ranges
        else:  # NEUTRAL
            base_rr = self.rr_neutral   # 2.5 - standard
        
        # Step 2: Adjust by volatility
        if atr and atr > 0:
            volatility_pct = (atr / entry_price) * 100
            
            if volatility_pct > self.high_volatility_threshold:
                # High volatility = wider targets (more room to move)
                base_rr += self.high_vol_adjustment
            elif volatility_pct < self.low_volatility_threshold:
                # Low volatility = tighter targets (less room to move)
                base_rr += self.low_vol_adjustment  # Note: this is negative
        
        return base_rr
    
    def _calculate_stop_distance(self, 
                                 atr: Optional[float], 
                                 entry_price: float, 
                                 confidence: float,
                                 fallback_distance: Optional[float] = None) -> float:
        """
        ✅ UNIFIED STOP DISTANCE CALCULATOR
        
        Calculates stop distance using ATR (primary) or fallback method.
        Used by all stop calculation methods for consistency.
        
        Args:
            atr: Average True Range (optional)
            entry_price: Entry price
            confidence: Signal confidence (0-100)
            fallback_distance: Optional fallback distance (e.g., zone size)
            
        Returns:
            Stop distance in price points
        """
        if atr and atr > 0:
            # Primary: ATR-based stop
            stop_multiplier = self._get_atr_multiplier(confidence)
            stop_distance = atr * stop_multiplier
        else:
            # Fallback: Use provided distance or percentage-based
            if fallback_distance and fallback_distance > 0:
                stop_distance = fallback_distance
            else:
                stop_distance = entry_price * self.min_stop_pct
        
        return stop_distance

    # ========== PRIMARY STOP CALCULATION METHOD ==========

    def calculate_simple_stops(self, 
                              entry_price: float, 
                              signal_type: str, 
                              support: float, 
                              resistance: float, 
                              atr: Optional[float] = None, 
                              confidence: float = 50,
                              market_regime: str = 'NEUTRAL') -> tuple:
        """
        ✅ PRIMARY METHOD: Professional hybrid approach (Market-Standard Logic)
        
        This is the RECOMMENDED method for all live trading strategies.
        
        Combines institutional trading practices:
        1. ATR-based stops (volatility-adaptive, confidence-adjusted)
        2. S/R-based targets (respects market structure)
        3. Dynamic fallback (volatility + market regime adjusted)
        
        Process:
        1. Calculate stop loss using ATR or fallback
        2. Try to use S/R levels as targets (if valid R:R)
        3. Fall back to dynamic targets if S/R invalid
        
        Args:
            entry_price: Entry price (spot price)
            signal_type: 'CALL' or 'PUT'
            support: Support level
            resistance: Resistance level
            atr: Average True Range (optional, for dynamic calculation)
            confidence: Signal confidence (0-100)
            market_regime: 'TRENDING' | 'RANGING' | 'NEUTRAL' (optional)
        
        Returns:
            tuple: (stop_loss, target)
            
        Examples:
            >>> # CALL signal with ATR
            >>> stop_loss, target = self.calculate_simple_stops(
            ...     entry_price=25990.70,
            ...     signal_type='CALL',
            ...     support=25800,
            ...     resistance=26200,
            ...     atr=78.5,
            ...     confidence=70,
            ...     market_regime='NEUTRAL'
            ... )
            >>> # Result: stop_loss=25912.73, target=26185.00
        """
        
        # ========== STEP 1: CALCULATE STOP LOSS ==========
        stop_distance = self._calculate_stop_distance(atr, entry_price, confidence)
        
        # ========== STEP 2: CALCULATE TARGET ==========
        if signal_type == 'CALL':
            # Stop loss below entry for CALL
            stop_loss = self._format_price(entry_price - stop_distance)
            
            # Try using resistance first (market structure)
            if resistance and resistance > entry_price:
                # Calculate R:R with resistance as target
                potential_reward = resistance - entry_price
                rr_ratio = potential_reward / stop_distance
                
                # Use resistance IF R:R is reasonable (1.5:1 minimum)
                if rr_ratio >= self.min_risk_reward:
                    target = resistance
                else:
                    # Resistance too close - use dynamic target
                    target = self._calculate_dynamic_target(
                        entry_price, stop_distance, atr, market_regime, 'CALL'
                    )
            else:
                # No valid resistance - use dynamic target
                target = self._calculate_dynamic_target(
                    entry_price, stop_distance, atr, market_regime, 'CALL'
                )
        
        else:  # PUT
            # Stop loss above entry for PUT
            stop_loss = self._format_price(entry_price + stop_distance)
            
            # Try using support first (market structure)
            if support and support < entry_price:
                # Calculate R:R with support as target
                potential_reward = entry_price - support
                rr_ratio = potential_reward / stop_distance
                
                # Use support IF R:R is reasonable (1.5:1 minimum)
                if rr_ratio >= self.min_risk_reward:
                    target = support
                else:
                    # Support too close - use dynamic target
                    target = self._calculate_dynamic_target(
                        entry_price, stop_distance, atr, market_regime, 'PUT'
                    )
            else:
                # No valid support - use dynamic target
                target = self._calculate_dynamic_target(
                    entry_price, stop_distance, atr, market_regime, 'PUT'
                )
        
        return (stop_loss, target)

    def _calculate_dynamic_target(self, 
                                  entry_price: float, 
                                  stop_distance: float, 
                                  atr: Optional[float], 
                                  market_regime: str,
                                  signal_type: str) -> float:
        """
        ✅ PROFESSIONAL STANDARD: Calculate target based on market regime and volatility
        
        This is how institutional traders adjust targets dynamically:
        - TRENDING markets = wider targets (3:1 R:R - more room to run)
        - RANGING markets = tighter targets (2:1 R:R - less room)
        - HIGH volatility = wider targets (accommodate larger swings)
        - LOW volatility = tighter targets (smaller expected moves)
        
        Args:
            entry_price: Entry price
            stop_distance: Calculated stop distance
            atr: Average True Range (optional)
            market_regime: 'TRENDING' | 'RANGING' | 'NEUTRAL'
            signal_type: 'CALL' or 'PUT'
        
        Returns:
            Target price
        """
        # Get dynamic R:R multiplier
        rr_multiplier = self._get_dynamic_rr_multiplier(atr, entry_price, market_regime)
        
        # Calculate target
        if signal_type == 'CALL':
            target = entry_price + (stop_distance * rr_multiplier)
        else:
            target = entry_price - (stop_distance * rr_multiplier)
        
        return self._format_price(target)

    # ========== SPECIALIZED STOP CALCULATION METHOD ==========

    def calculate_dynamic_stop_loss(self,
                                    zone_low: float,
                                    zone_high: float,
                                    direction: str,
                                    spot_price: float,
                                    atr: Optional[float] = None,
                                    confidence: float = 50) -> float:
        """
        ✅ SPECIALIZED METHOD: For zone-based strategies (Order Blocks, FVG)
        
        Use this when you have a specific zone (like Order Block or FVG) and need
        to calculate ONLY the stop loss (not the target).
        
        This method calculates stop loss using:
        1. ATR-based stop (primary)
        2. Zone size as fallback (if ATR unavailable)
        
        Args:
            zone_low: Bottom of setup zone (e.g., Order Block low)
            zone_high: Top of setup zone (e.g., Order Block high)
            direction: 'BULLISH' or 'BEARISH'
            spot_price: Current spot price (entry price)
            atr: Average True Range (optional, for dynamic calculation)
            confidence: Signal confidence (0-100)
            
        Returns:
            Stop loss price
            
        Examples:
            >>> # Bullish Order Block
            >>> stop_loss = self.calculate_dynamic_stop_loss(
            ...     zone_low=25850.00,
            ...     zone_high=25900.00,
            ...     direction='BULLISH',
            ...     spot_price=25990.70,
            ...     atr=78.5,
            ...     confidence=70
            ... )
            >>> # Result: stop_loss=25912.73
        """
        # Calculate zone size for fallback
        zone_size = abs(zone_high - zone_low)
        
        # Use unified stop distance calculator
        stop_distance = self._calculate_stop_distance(
            atr=atr,
            entry_price=spot_price,
            confidence=confidence,
            fallback_distance=zone_size  # Use zone size as fallback
        )
        
        # Ensure minimum stop distance (0.5% of price)
        min_distance = spot_price * self.min_stop_pct
        stop_distance = max(stop_distance, min_distance)
        
        # Calculate stop loss based on direction
        if direction == 'BULLISH':
            stop_loss = spot_price - stop_distance
        else:  # BEARISH
            stop_loss = spot_price + stop_distance
        
        return self._format_price(stop_loss)

    # ========== LEGACY/BACKTESTING METHOD ==========

    def calculate_atr_stops(self, 
                           entry_price: float, 
                           signal_type: str, 
                           confidence: float, 
                           replay_engine=None,
                           use_replay_logic: bool = False) -> Optional[Tuple[float, float, float]]:
        """
        ✅ LEGACY METHOD: For backward compatibility and backtesting
        
        This method is kept for:
        1. Backward compatibility with existing strategies using replay_engine
        2. Backtesting with historical replay_engine logic
        3. Research and comparison purposes
        
        For NEW strategies, use calculate_simple_stops() instead.
        
        Args:
            entry_price: Entry price
            signal_type: 'CALL' or 'PUT'
            confidence: Signal confidence (0-100)
            replay_engine: ReplayEngine instance (optional)
            use_replay_logic: If True, use replay_engine's logic (legacy)
                            If False, use unified BaseStrategy logic (RECOMMENDED)
        
        Returns:
            Tuple: (stop_loss, target, rr_ratio) or None if ATR unavailable
            
        Examples:
            >>> # Recommended: Use unified logic
            >>> result = self.calculate_atr_stops(
            ...     entry_price=25990.70,
            ...     signal_type='CALL',
            ...     confidence=70,
            ...     replay_engine=replay_engine,
            ...     use_replay_logic=False  # Use unified BaseStrategy logic
            ... )
            
            >>> # Legacy: Use replay_engine logic (for old backtests)
            >>> result = self.calculate_atr_stops(
            ...     entry_price=25990.70,
            ...     signal_type='CALL',
            ...     confidence=70,
            ...     replay_engine=replay_engine,
            ...     use_replay_logic=True  # Use old replay_engine logic
            ... )
        """
        if replay_engine is None:
            return None
        
        # Check if replay engine has required methods
        if not hasattr(replay_engine, 'calculate_atr'):
            return None
        
        # Calculate ATR from replay engine
        atr = replay_engine.calculate_atr(timeframe='5min', period=14)
        
        if atr is None or atr <= 0:
            return None
        
        # Choose calculation method
        if use_replay_logic:
            # ⚠️ LEGACY: Use replay_engine's logic (for backward compatibility)
            if not hasattr(replay_engine, 'get_atr_multiplier_for_signal'):
                # Fallback to unified logic if methods don't exist
                use_replay_logic = False
            else:
                atr_multiplier = replay_engine.get_atr_multiplier_for_signal(signal_type, confidence)
                rr_ratio = replay_engine.get_volatility_adjusted_rr(atr, entry_price)
        
        if not use_replay_logic:
            # ✅ RECOMMENDED: Use unified BaseStrategy logic
            atr_multiplier = self._get_atr_multiplier(confidence)
            rr_ratio = self._get_dynamic_rr_multiplier(atr, entry_price, 'NEUTRAL')
        
        # Calculate stop distance
        stop_distance = atr * atr_multiplier
        
        # Calculate stops and targets
        if signal_type == 'CALL':
            stop_loss = entry_price - stop_distance
            target = entry_price + (stop_distance * rr_ratio)
        else:  # PUT
            stop_loss = entry_price + stop_distance
            target = entry_price - (stop_distance * rr_ratio)
        
        return (
            self._format_price(stop_loss),
            self._format_price(target),
            round(rr_ratio, 2)
        )

    # ========== MARKET REGIME ANALYSIS ==========

    def check_market_regime(self, 
                           df: pd.DataFrame, 
                           current_idx: int, 
                           strategy_type: str) -> Tuple[bool, str]:
        """
        Check if current market regime is suitable for trading
        
        Different strategy types perform better in different market regimes:
        - TREND_FOLLOWING: Best in trending markets
        - BREAKOUT: Best in volatile/trending markets
        - MEAN_REVERSION: Best in ranging markets
        
        Args:
            df: Price dataframe
            current_idx: Current candle index
            strategy_type: 'TREND_FOLLOWING' | 'BREAKOUT' | 'MEAN_REVERSION'
        
        Returns:
            Tuple: (should_trade: bool, reason: str)
            
        Examples:
            >>> should_trade, reason = self.check_market_regime(
            ...     df=df_15min,
            ...     current_idx=len(df_15min)-1,
            ...     strategy_type='TREND_FOLLOWING'
            ... )
            >>> if not should_trade:
            ...     # Skip this strategy in current market regime
        """
        
        # Validate DataFrame first
        is_valid, errors = self.df_validator.validate_ohlc(df, strict=False, min_rows=20)
        if not is_valid:
            return (False, f"Invalid DataFrame: {errors}")
        
        # Validate index
        is_valid, error = self.df_validator.validate_index(df, current_idx, min_lookback=20)
        if not is_valid:
            return (False, error)
        
        # Detect market regime
        regime_info = self.market_regime_detector.detect_regime(df, current_idx)
        
        # Check if should trade in this regime
        should_trade, reason = self.market_regime_detector.should_trade_in_regime(
            regime_info, strategy_type
        )
        
        return (should_trade, reason)
    
    # ========== UTILITY METHODS ==========
    
    def _format_price(self, price: float) -> float:
        """
        Round price to 2 decimals
        
        Args:
            price: Price to format
            
        Returns:
            Formatted price (2 decimal places)
        """
        return round(price, 2)
