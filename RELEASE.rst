Release 0.5.0
=============

This release focuses on the new `metrics` module and implements several
Keras metrics.

Major Features and Improvements
-------------------------------

* `tfmr.metrics`:

  * Added new confusion metrics module with multiple binary, multiclass and
    multilabel metrics: `Accuracy`, `TruePositiveRate`, `TrueNegativeRate`,
    `PositivePredictiveValue`, `NegativePredictiveValue`, `Precision`, `Recall`,
    `Sensitivity`, `Specificity`, `Selectivity`, `TverskyIndex`, `FBetaScore`,
    `F1Score` and `IoU`. This module also exposes the abstract base class
    `ConfusionMetric`.
  * Added new image quality assessment metrics module with 2D/3D
    `PeakSignalToNoiseRatio`, `StructuralSimilarity` and
    `MultiscaleStructuralSimilarity`.
