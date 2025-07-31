import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from .parameter_manager import ParameterManager
from .impact_models import create_model, BaseImpactModel
import numpy as np

class ModelSelector:
    def __init__(self, parameter_manager: ParameterManager):
        self.parameter_manager = parameter_manager
        self.selection_strategies = {
            'best_r2': self._select_by_r2,
            'best_aic': self._select_by_aic,
            'optimization_friendly': self._select_optimization_friendly,
            'robust': self._select_robust
        }
    
    def select_model(self, 
                    stock: str, 
                    strategy: str = 'optimization_friendly') -> Tuple[str, BaseImpactModel]:
        if strategy not in self.selection_strategies:
            available = list(self.selection_strategies.keys())
            raise ValueError(f"Unknown strategy: {strategy}. Available: {available}")
        
        return self.selection_strategies[strategy](stock)
    
    def select_models_for_portfolio(self, 
                                  stocks: List[str], 
                                  strategy: str = 'optimization_friendly') -> Dict[str, Tuple[str, BaseImpactModel]]:
        selected_models = {}
        
        for stock in stocks:
            try:
                model_name, model_instance = self.select_model(stock, strategy)
                selected_models[stock] = (model_name, model_instance)
                
            except Exception as e:
                pass
        
        return selected_models
    
    def _select_by_r2(self, stock: str) -> Tuple[str, BaseImpactModel]:
        stock_params = self.parameter_manager.parameters[stock]
        
        best_r2 = -1
        best_model = None
        best_model_name = None
        
        for model_name, model_info in stock_params['models'].items():
            if model_info['valid'] and model_info['fitted']:
                r2 = model_info['performance']['r2']
                if r2 > best_r2:
                    best_r2 = r2
                    best_model_name = model_name
                    best_model = create_model(model_name, model_info['parameters'])
        
        if best_model is None:
            raise ValueError(f"No valid models found for {stock}")
        
        return best_model_name, best_model
    
    def _select_by_aic(self, stock: str) -> Tuple[str, BaseImpactModel]:
        stock_params = self.parameter_manager.parameters[stock]
        
        best_aic = float('inf')
        best_model = None
        best_model_name = None
        
        for model_name, model_info in stock_params['models'].items():
            if model_info['valid'] and model_info['fitted']:
                aic = model_info['performance']['aic']
                if aic < best_aic:
                    best_aic = aic
                    best_model_name = model_name
                    best_model = create_model(model_name, model_info['parameters'])
        
        if best_model is None:
            raise ValueError(f"No valid models found for {stock}")
        
        return best_model_name, best_model
    
    def _select_optimization_friendly(self, stock: str) -> Tuple[str, BaseImpactModel]:
        stock_params = self.parameter_manager.parameters[stock]
        models = stock_params['models']
        
        preference_order = ['Square Root', 'Linear', 'Power Law']
        
        for preferred_model in preference_order:
            if (preferred_model in models and 
                models[preferred_model]['valid'] and 
                models[preferred_model]['fitted']):
                
                model_info = models[preferred_model]
                
                if preferred_model in ['Square Root', 'Linear']:
                    model_instance = create_model(preferred_model, model_info['parameters'])
                    return preferred_model, model_instance

                elif preferred_model == 'Power Law':
                    beta = model_info['parameters'].get('beta', 0)
                    model_instance = create_model(preferred_model, model_info['parameters'])
                    return preferred_model, model_instance
        
        raise ValueError(f"No optimization-friendly models found for {stock}")
    
    def _select_robust(self, stock: str) -> Tuple[str, BaseImpactModel]:
        stock_params = self.parameter_manager.parameters[stock]
        
        candidates = []
        
        for model_name, model_info in stock_params['models'].items():
            if not (model_info['valid'] and model_info['fitted']):
                continue
            
            perf = model_info['performance']
            min_r2 = 0.3
            max_rmse = 0.01 
            
            if perf['r2'] >= min_r2 and perf['rmse'] <= max_rmse:
                complexity_penalty = {'Linear': 0, 'Square Root': 0.01, 'Power Law': 0.02}
                
                robustness_score = (
                    perf['r2'] +
                    (1.0 / (1.0 + perf['rmse'])) +
                    complexity_penalty.get(model_name, 0)
                )
                
                candidates.append((model_name, model_info, robustness_score))
        
        if not candidates:
            return self._select_by_r2(stock)

        best_candidate = max(candidates, key=lambda x: x[2])
        model_name, model_info, _ = best_candidate
        
        model_instance = create_model(model_name, model_info['parameters'])
        return model_name, model_instance
    
    def get_selection_summary(self, stocks: List[str], strategy: str = 'optimization_friendly') -> pd.DataFrame:
        summary_data = []
        
        for stock in stocks:
            try:
                model_name, model_instance = self.select_model(stock, strategy)
                params = self.parameter_manager.get_stock_parameters(stock, model_name)
                
                row = {
                    'Stock': stock,
                    'Selected_Model': model_name,
                    'Strategy': strategy,
                    'R²': params['performance']['r2'],
                    'RMSE': params['performance']['rmse'],
                    'AIC': params['performance']['aic'],
                    'Parameters': str(params['parameters'])
                }
                summary_data.append(row)
                
            except Exception as e:
                row = {
                    'Stock': stock,
                    'Selected_Model': 'ERROR',
                    'Strategy': strategy,
                    'R²': np.nan,
                    'RMSE': np.nan,
                    'AIC': np.nan,
                    'Parameters': str(e)
                }
                summary_data.append(row)
        
        return pd.DataFrame(summary_data)

def select_best_models(stocks: List[str], 
                      strategy: str = 'optimization_friendly',
                      parameter_manager: Optional[ParameterManager] = None) -> Dict[str, Tuple[str, BaseImpactModel]]:
    if parameter_manager is None:
        from .parameter_manager import get_parameter_manager
        parameter_manager = get_parameter_manager()
    
    selector = ModelSelector(parameter_manager)
    return selector.select_models_for_portfolio(stocks, strategy)
