# References

Li, T., Lawlor, A., Narasimhamurthy, A., Peng, X., & Hu, C. (2025). *UConn-MathWorks LFP/Gr Second-Life Battery Aging Dataset* [Data set]. University of Connecticut, REIL Datasets, 3. https://digitalcommons.lib.uconn.edu/reil_datasets/3/

**How it informs the task:** This is the source dataset used in the benchmark. The task uses Complete Batch 1 RPT measurements from this UConn-MathWorks LFP/graphite second-life ageing dataset. The dataset provides the repeated Reference Performance Tests, capacity measurements, pulse tests, and cell-replicate structure used for SOH target construction, diagnostic feature extraction, and unseen-cell evaluation.

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30*.

**How it informs the task:** This paper introduced SHAP, which is used in the reference solution to summarize global feature importance for the selected tree-based SOH model. SHAP is used only for interpretation of the final reference model, not for constructing the hidden target or grading predictions.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., VanderPlas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825-2830.

**How it informs the task:** Scikit-learn provides the grouped cross-validation, regression models, feature-selection interfaces, hyperparameter search, and evaluation metrics used in the reference implementation.
