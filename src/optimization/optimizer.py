import cvxpy as cp
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime

from .objectives import ObjectiveBuilder
from .constraints import ConstraintBuilder, ConstraintConfig
from ..models.impact_models import BaseImpactModel

class OptimizationResult:
    
    def __init__(self, 
                 success: bool,
                 execution_schedule: Optional[np.ndarray] = None,
                 total_cost: Optional[float] = None,
                 solver_status: Optional[str] = None,
                 solve_time: Optional[float] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        
        self.success = success
        self.execution_schedule = execution_schedule
        self.total_cost = total_cost
        self.solver_status = solver_status
        self.solve_time = solve_time
        self.metadata = metadata or {}
    
    def to_dataframe(self) -> pd.DataFrame:
        if not self.success or self.execution_schedule is None:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'period': range(len(self.execution_schedule)),
            'shares': self.execution_schedule,
            'cumulative_shares': np.cumsum(self.execution_schedule)
        })
        
        if 'start_time' in self.metadata:
            start_time = self.metadata['start_time']
            df['time'] = pd.date_range(start=start_time, periods=len(df), freq='1min')
        
        return df
    
    def get_summary(self) -> Dict[str, Any]:
        if not self.success:
            return {
                'success': False,
                'solver_status': self.solver_status,
                'error': self.metadata.get('error', 'Unknown error')
            }
        
        schedule = self.execution_schedule
        return {
            'success': True,
            'total_shares': np.sum(schedule),
            'total_cost': self.total_cost,
            'periods_used': np.sum(schedule > 0),
            'max_order_size': np.max(schedule),
            'avg_order_size': np.mean(schedule[schedule > 0]),
            'solve_time_seconds': self.solve_time,
            'solver_status': self.solver_status
        }

class MarketImpactOptimizer:
    
    def __init__(self, 
                 impact_model: BaseImpactModel,
                 stock_symbol: str,
                 solver: str = 'SCS'):
        self.impact_model = impact_model
        self.stock_symbol = stock_symbol
        self.solver = solver
        self.objective_builder = ObjectiveBuilder(impact_model, stock_symbol)
    
    def optimize(self, 
                 constraint_config: ConstraintConfig,
                 verbose: bool = False) -> OptimizationResult:
        start_time = datetime.now()
        
        try:
            constraint_builder = ConstraintBuilder(constraint_config)
            issues = constraint_builder.validate_config()
            if issues:
                return OptimizationResult(
                    success=False,
                    solver_status='CONFIG_ERROR',
                    metadata={'error': f"Constraint validation failed: {issues}"}
                )
            
            n_periods = constraint_config.time_periods
            execution_vars = cp.Variable(n_periods, nonneg=True, name="execution")
            objective = self.objective_builder.build_objective(execution_vars)
            constraints = constraint_builder.build_all_constraints(execution_vars)
            problem = cp.Problem(cp.Minimize(objective), constraints)
            
            all_installed = cp.installed_solvers()
            preferred_solvers = ['OSQP', 'SCS', 'CLARABEL']
            available_solvers = [s for s in preferred_solvers if s in all_installed]
            
            if not available_solvers:
                available_solvers = all_installed
                
            if self.solver in available_solvers:
                solvers_to_try = [self.solver] + [s for s in available_solvers if s != self.solver]
            else:
                solvers_to_try = available_solvers
                
            last_status = None
            last_error = None
            
            for solver in solvers_to_try:
                try:
                    problem.solve(solver=solver, verbose=False)
                    last_status = problem.status
                    
                    if problem.status in ['optimal', 'optimal_inaccurate']:
                        break
                        
                except Exception as solver_error:
                    last_error = str(solver_error)
                    continue
            
            solve_time = (datetime.now() - start_time).total_seconds()
            
            if problem.status not in ['optimal', 'optimal_inaccurate']:
                error_msg = f"All solvers failed. Final status: {problem.status or last_status}"
                if last_error:
                    error_msg += f". Last error: {last_error}"
                    
                return OptimizationResult(
                    success=False,
                    solver_status=problem.status or last_status,
                    solve_time=solve_time,
                    metadata={
                        'error': error_msg,
                        'tried_solvers': solvers_to_try,
                        'last_error': last_error
                    }
                )
            
            execution_schedule = execution_vars.value
            total_cost = problem.value
            
            if execution_schedule is None:
                return OptimizationResult(
                    success=False,
                    solver_status='NO_SOLUTION',
                    solve_time=solve_time,
                    metadata={'error': 'Solver returned None for execution schedule'}
                )
            
            execution_schedule = np.maximum(execution_schedule, 0)
            execution_schedule = np.round(execution_schedule, 6)
            
            return OptimizationResult(
                success=True,
                execution_schedule=execution_schedule,
                total_cost=total_cost,
                solver_status=problem.status,
                solve_time=solve_time,
                metadata={
                    'stock': self.stock_symbol,
                    'model': self.impact_model.model_name,
                    'total_shares_executed': np.sum(execution_schedule),
                    'periods_used': np.sum(execution_schedule > 0.001)
                }
            )
            
        except Exception as e:
            solve_time = (datetime.now() - start_time).total_seconds()
            return OptimizationResult(
                success=False,
                solver_status='ERROR',
                solve_time=solve_time,
                metadata={'error': str(e)}
            )

def optimize_single_stock(stock: str,
                         total_shares: int,
                         impact_model: BaseImpactModel,
                         max_order_size: int = 1000,
                         time_periods: int = 390,
                         solver: str = 'SCS',
                         verbose: bool = False) -> OptimizationResult:
    constraints = ConstraintConfig(
        total_shares=total_shares,
        time_periods=time_periods,
        max_order_size=max_order_size
    )
    
    optimizer = MarketImpactOptimizer(impact_model, stock, solver=solver)
    return optimizer.optimize(constraints, verbose=verbose)