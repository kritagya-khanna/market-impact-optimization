import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

class ParameterManager:
    def __init__(self, storage_path: str = "data/parameters/fitted_parameters.json"):
        self.storage_path = Path(storage_path)
        self.parameters = {}
        self.metadata = {}
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.storage_path.exists():
            self.load_parameters()
    
    def load_parameters(self) -> None:
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            self.parameters = data.get('stocks', {})
            self.metadata = data.get('metadata', {})
            
        except Exception as e:
            self.parameters = {}
            self.metadata = {}
    
    def save_parameters(self, parameters: Dict[str, Any]) -> None:
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(parameters, f, indent=2, default=str)
            
            self.parameters = parameters.get('stocks', {})
            self.metadata = parameters.get('metadata', {})
            
        except Exception as e:
            pass
    
    def extract_from_notebook_results(self, 
                                    all_model_results: Dict, 
                                    performance_summary: pd.DataFrame,
                                    stocks: List[str]) -> Dict[str, Any]:

        structured_params = {
            'stocks': {},
            'metadata': {
                'extraction_date': datetime.now().isoformat(),
                'source': 'notebook_analysis',
                'model_types': ['Linear', 'Square Root', 'Power Law']
            }
        }
        
        for stock in stocks:
            if stock in all_model_results:
                stock_data = self._extract_stock_parameters(
                    stock, 
                    all_model_results[stock], 
                    performance_summary
                )
                structured_params['stocks'][stock] = stock_data
            
        self.save_parameters(structured_params)
        return structured_params
    
    def _extract_stock_parameters(self, 
                                stock: str, 
                                model_results: Dict, 
                                performance_df: pd.DataFrame) -> Dict[str, Any]:
        
        stock_params = {
            'models': {},
            'best_model': None,
            'selection_criteria': {}
        }
        stock_performance = performance_df[performance_df['Stock'] == stock]

        for model_name, results in model_results.items():
            if results is not None:
                model_perf = stock_performance[stock_performance['Model'] == model_name]
                
                if not model_perf.empty:
                    perf_row = model_perf.iloc[0]
                    
                    model_info = {
                        'parameters': results['parameters'],
                        'performance': {
                            'r2': float(perf_row['R²']),
                            'rmse': float(perf_row['RMSE']),
                            'aic': float(perf_row['AIC'])
                        },
                        'fitted': True,
                        'valid': self._validate_parameters(model_name, results['parameters'])
                    }
                else:
                    model_info = {
                        'parameters': results['parameters'],
                        'performance': {
                            'r2': results.get('r2', 0),
                            'rmse': results.get('rmse', float('inf')),
                            'aic': results.get('aic', float('inf'))
                        },
                        'fitted': True,
                        'valid': self._validate_parameters(model_name, results['parameters'])
                    }
                
                stock_params['models'][model_name] = model_info

        best_model = self._select_best_model(stock_params['models'])
        stock_params['best_model'] = best_model
        
        return stock_params
    
    def _validate_parameters(self, model_name: str, parameters: Dict) -> bool:
        try:
            if model_name == 'Linear':
                beta = parameters.get('beta', 0)
                return 0 < beta < 0.01
            
            elif model_name == 'Square Root':
                alpha = parameters.get('alpha', 0)
                return 0 < alpha < 0.01
            
            elif model_name == 'Power Law':
                alpha = parameters.get('alpha', 0)
                beta = parameters.get('beta', 0)
                return (0 < alpha < 0.01) and (0 < beta < 1)
            
            return False
            
        except Exception:
            return False
    
    def _select_best_model(self, models: Dict) -> Optional[str]:
        valid_models = {name: info for name, info in models.items() 
                       if info['fitted'] and info['valid']}
        
        if not valid_models:
            return None

        best_model = max(valid_models.keys(), 
                        key=lambda x: valid_models[x]['performance']['r2'])
        
        return best_model
    
    def get_stock_parameters(self, stock: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        if stock not in self.parameters:
            raise ValueError(f"No parameters found for stock: {stock}")
        
        stock_data = self.parameters[stock]

        if model_name is None:
            model_name = stock_data.get('best_model')
            if model_name is None:
                raise ValueError(f"No best model identified for stock: {stock}")
        
        if model_name not in stock_data['models']:
            available_models = list(stock_data['models'].keys())
            raise ValueError(f"Model {model_name} not found for {stock}. Available: {available_models}")
        
        model_data = stock_data['models'][model_name]
        
        return {
            'stock': stock,
            'model_name': model_name,
            'parameters': model_data['parameters'],
            'performance': model_data['performance'],
            'is_best_model': model_name == stock_data['best_model'],
            'valid': model_data['valid']
        }
    
    def get_best_model_parameters(self, stock: str) -> Dict[str, Any]:
        return self.get_stock_parameters(stock, model_name=None)
    
    def get_all_stock_parameters(self, model_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        result = {}
        
        for stock in self.parameters.keys():
            try:
                result[stock] = self.get_stock_parameters(stock, model_name)
            except Exception as e:
                pass
        
        return result
    
    def get_available_stocks(self) -> List[str]:
        return list(self.parameters.keys())
    
    def get_available_models(self, stock: str) -> List[str]:
        if stock not in self.parameters:
            return []
        
        return list(self.parameters[stock]['models'].keys())
    
    def get_parameter_summary(self) -> pd.DataFrame:
        summary_data = []
        
        for stock, stock_data in self.parameters.items():
            for model_name, model_info in stock_data['models'].items():
                row = {
                    'Stock': stock,
                    'Model': model_name,
                    'Best_Model': stock_data['best_model'] == model_name,
                    'R²': model_info['performance']['r2'],
                    'RMSE': model_info['performance']['rmse'],
                    'AIC': model_info['performance']['aic'],
                    'Valid': model_info['valid'],
                    'Parameters': str(model_info['parameters'])
                }
                summary_data.append(row)
        
        return pd.DataFrame(summary_data)

_parameter_manager = None

def get_parameter_manager(storage_path: Optional[str] = None) -> ParameterManager:
    global _parameter_manager
    
    if _parameter_manager is None or storage_path is not None:
        path = storage_path or "data/parameters/fitted_parameters.json"
        _parameter_manager = ParameterManager(path)
    
    return _parameter_manager

def extract_and_save_parameters(all_model_results: Dict, 
                               performance_summary: pd.DataFrame,
                               stocks: List[str],
                               save_path: Optional[str] = None) -> Dict[str, Any]:
    manager = get_parameter_manager(save_path)
    return manager.extract_from_notebook_results(all_model_results, performance_summary, stocks)
