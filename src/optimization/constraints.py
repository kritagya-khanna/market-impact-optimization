import cvxpy as cp
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ConstraintConfig:
    total_shares: int            
    time_periods: int = 390   
    
    max_order_size: Optional[int] = None 
    min_order_size: Optional[int] = None    
    
    max_participation_rate: Optional[float] = None
    expected_volumes: Optional[np.ndarray] = None
    
    max_total_cost: Optional[float] = None         
    max_concentration: Optional[float] = None       
    
    completion_deadline: Optional[int] = None
    minimum_pace: Optional[float] = None

class ConstraintBuilder:
    
    def __init__(self, config: ConstraintConfig):
        self.config = config
        self.constraints = []
        
    def build_all_constraints(self, execution_variables: cp.Variable) -> List[cp.Constraint]:
        x = execution_variables
        constraints = []
        
        constraints.extend(self._build_fundamental_constraints(x))        
        constraints.extend(self._build_order_size_constraints(x))
        constraints.extend(self._build_participation_constraints(x))        
        constraints.extend(self._build_risk_constraints(x))        
        constraints.extend(self._build_timing_constraints(x))
        
        self.constraints = constraints
        return constraints
    
    def _build_fundamental_constraints(self, x: cp.Variable) -> List[cp.Constraint]:
        constraints = []
        constraints.append(cp.sum(x) == self.config.total_shares)        
        constraints.append(x >= 0)
        return constraints
    
    def _build_order_size_constraints(self, x: cp.Variable) -> List[cp.Constraint]:
        constraints = []
        
        if self.config.max_order_size is not None:
            constraints.append(x <= self.config.max_order_size)
        
        if self.config.min_order_size is not None and self.config.min_order_size > 0:
            pass
        
        return constraints
    
    def _build_participation_constraints(self, x: cp.Variable) -> List[cp.Constraint]:
        constraints = []
        
        if (self.config.max_participation_rate is not None and 
            self.config.expected_volumes is not None):
            
            max_rate = self.config.max_participation_rate
            expected_vols = self.config.expected_volumes
            
            if len(expected_vols) != self.config.time_periods:
                avg_vol = np.mean(expected_vols) if len(expected_vols) > 0 else 10000
                expected_vols = np.full(self.config.time_periods, avg_vol)
            
            max_execution_per_period = max_rate * expected_vols
            constraints.append(x <= max_execution_per_period)
        
        return constraints
    
    def _build_risk_constraints(self, x: cp.Variable) -> List[cp.Constraint]:
        constraints = []
        
        if self.config.max_concentration is not None:
            max_per_period = self.config.max_concentration * self.config.total_shares
            constraints.append(x <= max_per_period)
    
        return constraints
    
    def _build_timing_constraints(self, x: cp.Variable) -> List[cp.Constraint]:
        constraints = []
        
        if self.config.completion_deadline is not None:
            deadline = min(self.config.completion_deadline, self.config.time_periods)
            constraints.append(cp.sum(x[:deadline]) == self.config.total_shares)
            if deadline < self.config.time_periods:
                constraints.append(x[deadline:] == 0)
        
        return constraints
    
    def validate_config(self) -> List[str]:
        issues = []
        
        if self.config.total_shares <= 0:
            issues.append("total_shares must be positive")
        
        if self.config.time_periods <= 0:
            issues.append("time_periods must be positive")
        
        if (self.config.max_order_size is not None and 
            self.config.max_order_size * self.config.time_periods < self.config.total_shares):
            issues.append("max_order_size too small - cannot complete execution")
        
        if (self.config.max_concentration is not None and 
            self.config.max_concentration > 1.0):
            issues.append("max_concentration should be ≤ 1.0 (fraction of total)")
        
        if (self.config.completion_deadline is not None and 
            self.config.completion_deadline > self.config.time_periods):
            issues.append("completion_deadline exceeds time_periods")
        
        return issues

def create_basic_constraints(total_shares: int, 
                           max_order_size: int = 1000,
                           time_periods: int = 390) -> ConstraintConfig:
    return ConstraintConfig(
        total_shares=total_shares,
        time_periods=time_periods,
        max_order_size=max_order_size
    )

def create_risk_managed_constraints(total_shares: int,
                                  max_order_size: int = 1000,
                                  max_participation_rate: float = 0.10,
                                  max_concentration: float = 0.05,
                                  time_periods: int = 390) -> ConstraintConfig:
    return ConstraintConfig(
        total_shares=total_shares,
        time_periods=time_periods,
        max_order_size=max_order_size,
        max_participation_rate=max_participation_rate,
        max_concentration=max_concentration
    )