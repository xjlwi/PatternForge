# kohonen_production.py
import numpy as np
import logging
import os
import time
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any
import joblib
import argparse
import yaml
import mlflow
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SOM_Training")

class SelfOrganizingMap:
    """Self-Organizing Map implementation for production use."""
    
    def __init__(self, width: int, height: int, input_dim: int = 3, 
                 alpha0: float = 0.1, sigma0: Optional[float] = None):
        """
        Initialize SOM with given dimensions.
        
        Args:
            width: Number of neurons in x dimension
            height: Number of neurons in y dimension
            input_dim: Dimension of input data (default 3 for RGB)
            alpha0: Initial learning rate
            sigma0: Initial neighborhood radius (defaults to max(width,height)/2)
        """
        self.width = width
        self.height = height
        self.input_dim = input_dim
        self.alpha0 = alpha0
        self.sigma0 = sigma0 if sigma0 is not None else max(width, height) / 2
        self.weights = np.random.random((width, height, input_dim))
        self.metrics = {"quantization_error": []}
        
    def find_bmu(self, sample: np.ndarray) -> Tuple[int, int]:
        """
        Find the Best Matching Unit (BMU) for a given input sample.
        
        Args:
            sample: Input data point
            
        Returns:
            Tuple of (x, y) coordinates of the BMU
        """
        # Calculate squared distance to each neuron
        diff = self.weights - sample
        sq_distances = np.sum(diff**2, axis=2)
        
        # Find coordinates of the BMU
        bmu_idx = np.unravel_index(np.argmin(sq_distances), (self.width, self.height))
        return bmu_idx
    
    def _calculate_influence(self, bmu_pos: Tuple[int, int], sigma: float) -> np.ndarray:
        """
        Calculate neighborhood influence based on BMU position and current sigma.
        
        Args:
            bmu_pos: Position of the BMU
            sigma: Current neighborhood radius
            
        Returns:
            Array of influence values
        """
        # Create a grid of neuron positions
        x_grid, y_grid = np.meshgrid(np.arange(self.width), np.arange(self.height))
        x_grid = x_grid.transpose()
        y_grid = y_grid.transpose()
        
        # Calculate distances from all neurons to BMU
        bmu_x, bmu_y = bmu_pos
        sq_distance = (x_grid - bmu_x)**2 + (y_grid - bmu_y)**2
        
        # Calculate influence (neighborhood function)
        influence = np.exp(-sq_distance / (2 * sigma**2))
        return influence
    
    def train(self, input_data: np.ndarray, n_iterations: int, 
              batch_size: int = None, evaluation_interval: int = 100) -> Dict[str, list]:
        """
        Train the SOM on input data.
        
        Args:
            input_data: Training data of shape (n_samples, input_dim)
            n_iterations: Maximum number of training iterations
            batch_size: Size of mini-batches (None means full batch)
            evaluation_interval: Calculate and log metrics every N iterations
            
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Starting SOM training with {n_iterations} iterations")
        logger.info(f"Input data shape: {input_data.shape}")
        start_time = time.time()
        
        # Parameter for decay functions
        decay_factor = n_iterations / np.log(self.sigma0)
        batch_size = len(input_data) if batch_size is None else batch_size
        
        try:
            for t in range(n_iterations):
                # Calculate current learning rate and neighborhood radius
                iteration_progress = t / n_iterations
                sigma_t = self.sigma0 * np.exp(-t / decay_factor)
                alpha_t = self.alpha0 * np.exp(-t / decay_factor)
                
                # Process mini-batch
                indices = np.random.choice(len(input_data), size=batch_size, replace=False)
                batch = input_data[indices]
                
                for sample in batch:
                    # Find BMU for current sample
                    bmu_pos = self.find_bmu(sample)
                    
                    # Calculate neighborhood influence
                    influence = self._calculate_influence(bmu_pos, sigma_t)
                    
                    # Apply influence to reshape neighborhood (vectorized)
                    influence_reshaped = influence[:, :, np.newaxis]
                    update = alpha_t * influence_reshaped * (sample - self.weights)
                    self.weights += update
                
                # Log progress and calculate metrics periodically
                if t % evaluation_interval == 0 or t == n_iterations - 1:
                    # Calculate quantization error
                    q_error = self._calculate_quantization_error(input_data)
                    self.metrics["quantization_error"].append(q_error)
                    
                    # Log progress
                    elapsed = time.time() - start_time
                    logger.info(f"Iteration {t}/{n_iterations} - "
                                f"QE: {q_error:.6f} - "
                                f"Alpha: {alpha_t:.6f} - "
                                f"Sigma: {sigma_t:.6f} - "
                                f"Time: {elapsed:.2f}s")
                    
                    # Log to MLflow
                    mlflow.log_metric("quantization_error", q_error, step=t)
                    mlflow.log_metric("learning_rate", alpha_t, step=t)
                    mlflow.log_metric("neighborhood_radius", sigma_t, step=t)
            
            logger.info(f"SOM training completed in {time.time() - start_time:.2f}s")
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
            
    def _calculate_quantization_error(self, data: np.ndarray) -> float:
        """
        Calculate quantization error as average distance between samples and their BMUs.
        
        Args:
            data: Input data of shape (n_samples, input_dim)
            
        Returns:
            Float value of quantization error
        """
        total_error = 0.0
        for sample in data:
            bmu_x, bmu_y = self.find_bmu(sample)
            bmu_weight = self.weights[bmu_x, bmu_y]
            total_error += np.sqrt(np.sum((sample - bmu_weight)**2))
        return total_error / len(data)
    
    def get_feature_map(self) -> np.ndarray:
        """
        Get the trained feature map (weight matrix).
        
        Returns:
            Numpy array of shape (width, height, input_dim)
        """
        return self.weights
    
    def map_input(self, input_data: np.ndarray) -> np.ndarray:
        """
        Map input data to SOM grid coordinates.
        
        Args:
            input_data: Input data of shape (n_samples, input_dim)
        
        Returns:
            Array of mapped positions of shape (n_samples, 2)
        """
        mapped_positions = np.zeros((len(input_data), 2), dtype=int)
        for i, sample in enumerate(input_data):
            mapped_positions[i] = self.find_bmu(sample)
        return mapped_positions
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'SelfOrganizingMap':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded SelfOrganizingMap instance
        """
        logger.info(f"Loading model from {filepath}")
        return joblib.load(filepath)
    
    def visualize(self, title: str = "Self-Organizing Map", 
                  output_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize the SOM weight map.
        
        Args:
            title: Title for the plot
            output_path: If provided, save figure to this path
            
        Returns:
            Matplotlib figure object
        """
        # Ensure weights are within [0, 1] for RGB visualization
        normalized_weights = np.clip(self.weights, 0, 1)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(title)
        ax.imshow(normalized_weights)
        ax.set_xticks([])
        ax.set_yticks([])
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_path}")
            
        return fig


def load_data(filepath: str) -> np.ndarray:
    """
    Load training data from file.
    
    Args:
        filepath: Path to the data file
        
    Returns:
        Numpy array of shape (n_samples, input_dim)
    """
    try:
        return np.load(filepath)
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {str(e)}")
        raise


def train_model(config_path: str) -> None:
    """
    Main training function to be called from the pipeline.
    
    Args:
        config_path: Path to configuration YAML file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract configurations
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    output_config = config.get('output', {})
    
    # Set up output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_config.get('base_dir', 'outputs'), timestamp)
    model_dir = os.path.join(output_dir, 'models')
    viz_dir = os.path.join(output_dir, 'visualizations')
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    # Start MLflow run
    experiment_name = config.get('experiment_name', 'SOM_Training')
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"SOM_{timestamp}"):
        # Log parameters
        mlflow.log_params({
            "width": model_config.get('width', 10),
            "height": model_config.get('height', 10),
            "input_dim": model_config.get('input_dim', 3),
            "alpha0": model_config.get('alpha0', 0.1),
            "sigma0": model_config.get('sigma0', None),
            "n_iterations": training_config.get('n_iterations', 1000),
            "batch_size": training_config.get('batch_size', None),
        })
        
        try:
            # Load data
            logger.info(f"Loading data from {data_config.get('path')}")
            input_data = load_data(data_config.get('path'))
            
            # Create and train model
            som = SelfOrganizingMap(
                width=model_config.get('width', 10),
                height=model_config.get('height', 10),
                input_dim=model_config.get('input_dim', 3),
                alpha0=model_config.get('alpha0', 0.1),
                sigma0=model_config.get('sigma0', None)
            )
            
            training_metrics = som.train(
                input_data=input_data,
                n_iterations=training_config.get('n_iterations', 1000),
                batch_size=training_config.get('batch_size', None),
                evaluation_interval=training_config.get('evaluation_interval', 100)
            )
            
            # Save model
            model_path = os.path.join(model_dir, 'som_model.joblib')
            som.save(model_path)
            mlflow.log_artifact(model_path)
            
            # Create and save visualization
            viz_path = os.path.join(viz_dir, 'som_visualization.png')
            som.visualize(title=f"SOM ({som.width}x{som.height})", output_path=viz_path)
            mlflow.log_artifact(viz_path)
            
            # Log final metrics
            final_qe = training_metrics["quantization_error"][-1]
            mlflow.log_metric("final_quantization_error", final_qe)
            
            logger.info(f"Training completed successfully. Final QE: {final_qe:.6f}")
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}")
            mlflow.log_param("error", str(e))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Self-Organizing Map model')
    parser.add_argument('--config', type=str, required=True, help='Path to yaml file')
    args = parser.parse_args()
    
    train_model(args.config)
