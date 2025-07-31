import cvxpy as cp
import numpy as np
from typing import Dict, Any
from ..models.impact_models import BaseImpactModel

class ObjectiveBuilder:

    def __init__(self, model: BaseImpactModel, stock_symbol: str):
        self.model = model
        self.stock_symbol = stock_symbol
        self.model_name = model.model_name
        
    def build_objective(self, execution_variables: cp.Variable) -> cp.Expression:
        if self.model_name == 'LinearModel':
            return self._build_linear_objective(execution_variables)
        elif self.model_name == 'SquareRootModel':
            return self._build_sqrt_objective(execution_variables)
        elif self.model_name == 'PowerLawModel':
            return self._build_power_objective(execution_variables)
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")
    
    def _build_linear_objective(self, x: cp.Variable) -> cp.Expression:
        beta = self.model.parameters['beta']
        return cp.sum(beta * x)
    
    def _build_sqrt_objective(self, x: cp.Variable) -> cp.Expression:
        alpha = self.model.parameters['alpha']
        return cp.sum(alpha * cp.power(x, 0.5))
    
    def _build_power_objective(self, x: cp.Variable) -> cp.Expression:
        alpha = self.model.parameters['alpha']
        beta = self.model.parameters['beta']
        
        if beta >= 1.0:
            return cp.sum(alpha * cp.power(x, beta))
        else:
            typical_size = 100
            linear_coeff = alpha * beta * (typical_size ** (beta - 1))
            return cp.sum(linear_coeff * x)
    
    def estimate_cost(self, execution_schedule: np.ndarray) -> float:
        individual_costs = self.model.calculate_impact(execution_schedule)
        return np.sum(individual_costs)
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'stock': self.stock_symbol,
            'model_name': self.model_name,
            'parameters': self.model.parameters,
        }


def create_objective(model: BaseImpactModel, stock_symbol: str) -> ObjectiveBuilder:
    return ObjectiveBuilder(model, stock_symbol)