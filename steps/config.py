from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """Model configuration parameters."""
    model_name: str = "LinearRegression"
    