

## Requirements
Install from `requirements.txt` Ensure that the following dependencies are installed:
- Python 3.11+
- Required python packages installed via `requirements.txt`
```
pip install -r requirements.txt
```
- MLFlow for experiment tracking and model registration.
- A Valid configuration file in YAML Format _refer `config.yaml`_
---


## Pipeline Stages
The pipeline is divided into the following stages:

- **Prepare**: Prepares and validates the input data. Optionally normalizes the data.
- **Train**: Trains the SOM model using the provided configuration.
- **Evaluate**: Evaluates the trained model using metrics like quantization error.
- **Register**: Registers the trained model in MLflow for deployment and tracking.
All: Executes all the above stages sequentially.

---

## Usage
Run the pipeline using the following command:

```
python app/som-mlops-pipeline.py --config <path_to_config> --stage <stage>
```

Arguments:
- `--config`: Path to the configuration YAML file.
- `--stage`: Pipeline stage to execute. Options: `prepare`, `train`, `evaluate`, `register`, `all`.

**Example**
To execute all stages:

```
python app/som-mlops-pipeline.py --config app/som-config-file.yaml --stage all
```

To execute only the training stage:

```
python app/som-mlops-pipeline.py --config app/som-config-file.yaml --stage train
```

## Output Structure
The pipeline then generate outpus in directory in `output.base_dir` configuration.
```
outputs/
    <timestamp>/
        models/
            som_model.joblib       # Trained SOM model
        preprocessed/
            normalized_data.npy    # Preprocessed data (if normalization is enabled)
            norm_params.joblib     # Normalization parameters

```

## Notes
- Ensure the input data file exists and matches the expected format (e.g., a 2D NumPy array).
- If normalization is enabled, the pipeline saves normalization parameters for use during inference.
- MLflow must be configured with a tracking server if you intend to register models.