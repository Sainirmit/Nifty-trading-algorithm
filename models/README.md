# Models Directory

This directory stores trained machine learning models for the Nifty 50 trading algorithm.

## Model Types

The project includes several types of models:

1. **Regression Models**

   - Predict the next day's price movement
   - Used for generating precise price targets

2. **Classification Models**

   - Predict directional movement (up/down)
   - Used for generating buy/sell signals

3. **Ensemble Models**
   - Combine multiple model predictions
   - Used to improve robustness and reduce overfitting

## File Format

Models are saved in the following formats:

- Scikit-learn models: `.joblib` or `.pkl` files
- TensorFlow/Keras models: `.h5` or SavedModel format

## Model Naming Convention

Models should be named according to the following convention:

```
model_type-features-timeframe-date_trained.extension
```

Example: `lstm-technical_features-daily-20230415.h5`

## Model Performance Metrics

Each model should have an accompanying JSON file with performance metrics:

```
model_name.json
```

This file should include:

- Training accuracy/loss
- Validation accuracy/loss
- Test set performance
- Parameter values
- Feature importance (if applicable)

## Model Versioning

Always increment the model version when retraining or updating a model. This ensures traceability of results.

## Model Deployment

For live trading, models should be copied to the deployment environment from this directory.

## Note

Large model files (>50MB) should not be committed to version control. Instead, consider using Git LFS or uploading to a separate storage service.
