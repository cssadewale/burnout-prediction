# Burnout Prediction Fix Notes

## Fixed

- Added Python 3.12 deployment pinning and compatible binary-wheel dependencies.
- Made the model path relative to `app.py`.
- Added clear model-loading error handling.
- Added an exact feature-schema check against the saved Gradient Boosting regressor.
- Exposed when a raw regression output falls outside the documented 0–1 range before display clipping.

## Remaining model-quality limitation

The original repository does not include a fitted scaler. The app therefore retains the documented training bounds for min-max normalization. The production-grade follow-up is to retrain/export a single pipeline containing preprocessing and the regressor, then remove the manual bounds.
