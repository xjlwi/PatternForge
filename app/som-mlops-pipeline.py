# pipeline.py
import os
import sys
import argparse
import yaml
import logging
from datetime import datetime
import mlflow
from mlflow.models import infer_signature
import joblib
import numpy as np
import subprocess
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SOM_Pipeline")

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate the configuration file.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Boolean indicating if config is valid
    """
    required_sections = ['data', 'model', 'training', 'output']
    for section in required_sections:
        if section not in config:
            logger.error(f"Missing required section '{section}' in config")
            return False
    
    # Validate data section
    if 'path' not in config['data']:
        logger.error("Missing 'path' in data section")
        return False
    
    # Validate model section
    model_required = ['width', 'height']
    for param in model_required:
        if param not in config['model']:
            logger.error(f"Missing '{param}' in model section")
            return False
    
    # Validate training section
    if 'n_iterations' not in config['training']:
        logger.error("Missing 'n_iterations' in training section")
        return False
    
    return True


def prepare_data(config: Dict[str, Any]) -> str:
    """
    Prepare and validate the input data.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Path to the prepared data file
    """
    data_config = config.get('data', {})
    input_path = data_config.get('path')
    
    # Check if data file exists
    if not os.path.exists(input_path):
        logger.error(f"Data file not found: {input_path}")
        raise FileNotFoundError(f"Data file not found: {input_path}")
    
    try:
        # Load and validate data
        data = np.load(input_path)
        
        # Check data dimensions
        if len(data.shape) != 2:
            logger.error(f"Invalid data shape: {data.shape}. Expected 2D array.")
            raise ValueError(f"Invalid data shape: {data.shape}. Expected 2D array.")
        
        expected_dim = config['model'].get('input_dim', 3)
        if data.shape[1] != expected_dim:
            logger.error(f"Data dimension mismatch. Expected {expected_dim}, got {data.shape[1]}")
            raise ValueError(f"Data dimension mismatch. Expected {expected_dim}, got {data.shape[1]}")
        
        # Perform optional preprocessing
        if data_config.get('normalize', False):
            # Normalize data to [0, 1] range
            data_min = np.min(data, axis=0)
            data_max = np.max(data, axis=0)
            data = (data - data_min) / (data_max - data_min + 1e-10)
            
            # Save preprocessed data
            preproc_dir = os.path.join(config['output']['base_dir'], 'preprocessed')
            os.makedirs(preproc_dir, exist_ok=True)
            
            preprocessed_path = os.path.join(preproc_dir, 'normalized_data.npy')
            np.save(preprocessed_path, data)
            
            # Save normalization parameters for inference
            norm_params = {
                'min': data_min,
                'max': data_max
            }
            joblib.dump(norm_params, os.path.join(preproc_dir, 'norm_params.joblib'))
            
            logger.info(f"Data normalized and saved to {preprocessed_path}")
            return preprocessed_path
        
        return input_path
        
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise


def train_model(config_path: str) -> str:
    """
    Run the training script with the provided config.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Path to the trained model
    """
    try:
        # Call the training script as a module
        logger.info(f"Starting model training with config: {config_path}")
        
        # Option 1: Import and call the function directly
        from kohonen_production import train_model
        train_model(config_path)
        
        # Option 2: Run as a subprocess
        # cmd = [sys.executable, 'kohonen_production.py', '--config', config_path]
        # subprocess.run(cmd, check=True)
        
        # Load config to get output path
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(config['output']['base_dir'], timestamp)
        model_path = os.path.join(output_dir, 'models', 'som_model.joblib')
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at expected path: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Model training completed. Model saved to {model_path}")
        return model_path
    
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise


def evaluate_model(model_path: str, config_path: str) -> dict:
    """
    Evaluate the trained model.
    
    Args:
        model_path: Path to the trained model
        config_path: Path to the configuration file
        
    Returns:
        Dictionary of evaluation metrics
    """
    try:
        logger.info(f"Evaluating model: {model_path}")
        
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Prepare test data if validation split is specified
        data_config = config.get('data', {})
        if data_config.get('split_ratio', 0) > 0:
            # Implementation for validation data evaluation
            pass
        
        # For this example, we'll just load the model and calculate metrics on training data
        from .som import SelfOrganizingMap, load_data
        
        # Load model and data
        som = SelfOrganizingMap.load(model_path)
        
        data_path = data_config.get('path')
        if data_config.get('normalize', False):
            preproc_dir = os.path.join(config['output']['base_dir'], 'preprocessed')
            data_path = os.path.join(preproc_dir, 'normalized_data.npy')
        
        data = load_data(data_path)
        
        # Calculate quantization error
        qe = som._calculate_quantization_error(data)
        
        # Calculate additional metrics like topographic error
        # (Implementation would depend on specific requirements)
        
        metrics = {
            "quantization_error": qe,
            # Add more metrics as needed
        }
        
        logger.info(f"Model evaluation completed: {metrics}")
        return metrics
    
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        raise


def register_model(model_path: str, metrics: dict, config_path: str) -> str:
    """
    Register the model in MLflow.
    
    Args:
        model_path: Path to the trained model
        metrics: Evaluation metrics
        config_path: Path to configuration file
        
    Returns:
        MLflow model URI
    """
    try:
        logger.info(f"Registering model from {model_path}")
        
        # Load config and model
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        from kohonen_production import SelfOrganizingMap
        som = SelfOrganizingMap.load(model_path)
        
        # Create sample data for signature
        data_config = config.get('data', {})
        input_dim = config['model'].get('input_dim', 3)
        sample_input = np.random.random((5, input_dim))
        sample_output = som.map_input(sample_input)
        
        # Create model signature
        signature = infer_signature(sample_input, sample_output)
        
        # Prepare artifact path
        artifact_path = f"som_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create a MLflow custom model
        with mlflow.start_run() as run:
            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            # Log parameters
            mlflow.log_params({
                "width": som.width,
                "height": som.height,
                "input_dim": som.input_dim,
                "alpha0": som.alpha0,
                "sigma0": som.sigma0,
            })
            
            # Log the joblib model
            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=SelfOrganizingMapWrapper(som),
                artifacts={"som_model": model_path},
                signature=signature
            )
            
            # Register the model
            model_uri = f"runs:/{run.info.run_id}/{artifact_path}"
            registered_model_name = config.get('experiment_name', 'SOM_Model')
            
            mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name
            )
            
            logger.info(f"Model registered as {registered_model_name}")
            return model_uri
    
    except Exception as e:
        logger.error(f"Error registering model: {str(e)}")
        raise


class SelfOrganizingMapWrapper(mlflow.pyfunc.PythonModel):
    """
    MLflow PythonModel wrapper for SelfOrganizingMap.
    """
    
    def __init__(self, som=None):
        self.som = som
    
    def load_context(self, context):
        if self.som is None:
            # Load the model from the artifacts
            self.som = joblib.load(context.artifacts["som_model"])
    
    def predict(self, context, model_input):
        """
        Map input data to SOM grid coordinates.
        
        Args:
            context: MLflow context
            model_input: Input data as pandas DataFrame or numpy array
            
        Returns:
            Numpy array of mapped positions
        """
        # Convert to numpy if DataFrame
        if hasattr(model_input, 'values'):
            input_data = model_input.values
        else:
            input_data = model_input
            
        # Map inputs to SOM grid
        return self.som.map_input(input_data)


def main():
    parser = argparse.ArgumentParser(description="MLOps pipeline for SOM model")
    parser.add_argument('--config', type=str, required=True, help='Path to configuration YAML file')
    parser.add_argument('--stage', type=str, default='all', 
                        choices=['prepare', 'train', 'evaluate', 'register', 'all'],
                        help='Pipeline stage to execute')
    args = parser.parse_args()
    
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate configuration
        if not validate_config(config):
            logger.error("Invalid configuration")
            sys.exit(1)
        
        # Create base output directory
        output_base = config['output'].get('base_dir', 'outputs')
        os.makedirs(output_base, exist_ok=True)
        
        # Set up MLflow
        experiment_name = config.get('experiment_name', 'SOM_Pipeline')
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Execute requested pipeline stage(s)
            if args.stage in ['prepare', 'all']:
                prepared_data_path = prepare_data(config)
                logger.info(f"Data preparation completed: {prepared_data_path}")
            
            if args.stage in ['train', 'all']:
                model_path = train_model(args.config)
                logger.info(f"Model training completed: {model_path}")
            
            if args.stage in ['evaluate', 'all']:
                metrics = evaluate_model(model_path, args.config)
                logger.info(f"Model evaluation completed: {metrics}")
            
            if args.stage in ['register', 'all']:
                model_uri = register_model(model_path, metrics, args.config)
                logger.info(f"Model registration completed: {model_uri}")
            
        logger.info("Pipeline completed successfully")
    
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
