# diss

This report proposes and applies an approach to formally characterizing uncertainty in medical segmentation tasks. The proposed method uses a Structured Uncertainty Prediction model to approximate the output of an ensemble model by training the model to predict the mean and covariance of a Gaussian distribution. Since this approach encodes a full covariance matrix, it allows for the sampling of spatially correlated predictions after training. The uncertainty is captured formally in the distribution and allows for model introspection. This approach is demonstrated on a brain tumor segmentation task where uncertainty estimation is crucial for making an informed decision on treatment options. However, results are limited and further research needs to be conducted.

# run supn_cholespy_image docker image

# installs
pip install wandb matplotlib opencv-python scikit-image pandas

# file structure

diss-main/
│
├── compute_stats.py              # Compute dataset statistics
├── devices.py                    # Device configuration (CPU/GPU)
├── ensemble_metrics.py          # Metrics for ensemble evaluation
├── metrics.py                   # Core metric functions (e.g., IoU, accuracy)
├── sampling.py                  # Custom data sampling methods
├── synth_data.py                # Synthetic data generation
├── train.py                     # Model training pipeline
├── u_net.py                     # U-Net model architecture
│
├── models/
│   ├── model.py                 # Primary model definition
│   ├── model2.py                # Alternate model variant
│   └── supn_blocks.py           # Custom building blocks for models
│
├── notebooks/
│   ├── data_loader.py           # Dataset loading utilities
│   ├── data_sanity_check.py     # Visual/data sanity checks
│   ├── data_split.py            # Data splitting for training/testing
│   ├── model_pred.py            # Model inference on new data
│   ├── other_data_loaders.py    # Extra data loader utilities
│   ├── rename_data.py           # Batch renaming of dataset files
│   ├── save_pred_logit_space.py # Save predictions in logit space
│   └── tester.py                # Script to test trained models
│
└── .gitignore                   # Files and directories to ignore in git
