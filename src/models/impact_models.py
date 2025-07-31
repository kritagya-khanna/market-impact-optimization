import numpy as np
from typing import Dict, Union
import warnings

class BaseImpactModel:    
    def __init__(self, parameters: Dict[str, float]):
        self.parameters = parameters
        self.model_name = self.__class__.__name__
        
    def calculate_impact(self, order_size: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        raise NotImplementedError("Subclasses must implement calculate_impact method")
    
    def get_parameter_names(self) -> list:
        raise NotImplementedError("Subclasses must implement get_parameter_names method")
    
    def validate_parameters(self) -> bool:
        required_params = self.get_parameter_names()
        return all(param in self.parameters for param in required_params)

class LinearModel(BaseImpactModel):
    
    def calculate_impact(self, order_size: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        beta = self.parameters['beta']
        return beta * order_size
    
    def get_parameter_names(self) -> list:
        return ['beta']
    
    def validate_parameters(self) -> bool:
        if not super().validate_parameters():
            return False
        beta = self.parameters['beta']
        return 0 < beta < 0.01

class SquareRootModel(BaseImpactModel):
    
    def calculate_impact(self, order_size: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        alpha = self.parameters['alpha']
        order_size = np.maximum(order_size, 0.01)
        return alpha * np.sqrt(order_size)

    def get_parameter_names(self) -> list:
        return ['alpha']

    def validate_parameters(self) -> bool:
        if not super().validate_parameters():
            return False
        alpha = self.parameters['alpha']
        return 0 < alpha < 0.01

class PowerLawModel(BaseImpactModel):
    def calculate_impact(self, order_size: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        alpha = self.parameters['alpha']
        beta = self.parameters['beta']
        order_size = np.maximum(order_size, 0.01)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return alpha * np.power(order_size, beta)
    
    def get_parameter_names(self) -> list:
        return ['alpha', 'beta']
    
    def validate_parameters(self) -> bool:
        if not super().validate_parameters():
            return False
        
        alpha = self.parameters['alpha']
        beta = self.parameters['beta']
        return (0 < alpha < 0.1) and (0.1 < beta < 2.0)

MODEL_CLASSES = {
    'Linear': LinearModel,
    'Square Root': SquareRootModel,
    'Power Law': PowerLawModel
}

def create_model(model_name: str, parameters: Dict[str, float]) -> BaseImpactModel:
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CLASSES.keys())}")
    
    model_class = MODEL_CLASSES[model_name]
    return model_class(parameters)