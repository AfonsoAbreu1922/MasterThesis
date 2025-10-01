# MasterThesis

Lung cancer continues to be one of the most common causes of mortality across the globe. This
is primarily owing to the late-stage diagnosis of the illness, accompanied by low survival rates.
Currently, the gold standard for tumour malignancy characterization is tissue biopsy. However,
it comes with various limitations, such as being invasive, potentially having significant clinical
risks. Furthermore, Computer-Aided Diagnosis (CAD) systems based on deep learning that utilize
Computed Tomography scans to assist in clinical decision-making have emerged as noninvasive
options that offer a more promising alternative.
In most cases, deep learning models’ success relies on the very large annotated databases of
medical images. Gathering such an amount of information is time-consuming, expensive, and also
labour-intensive, which can be a critical barrier to the widespread adoption of CAD systems. The
challenge of unlabeled data in medical imaging repositories can be solved with semi-supervised
learning, as it allows the inclusion of unlabeled data into the training process.
This dissertation applies the FixMatch framework, a semi-supervised method that combines
pseudo-labeling and consistency regularization, to classify lung nodules’ malignancy. The framework evaluation on CT data, both labeled from the LIDC-IDRI dataset and unlabeled from the
Luna25 dataset, featured an explainability analysis with Gradient-weighted Class Activation Mapping (Grad-CAM) to compare the fully-supervised and semi-supervised models’ interpretability.
Results showed that incorporating unlabeled data improved performance over a purely supervised baseline, with an increase of 2% in AUROC when using a 5:1 ratio of unlabeled to labeled
data. Nevertheless, the explainability analysis didn’t promote sustainable conclusions in the clinical context, despite observations of more localized activation maps from the semi-supervised
model.
This work contributes to the effort to improve the early detection of lung cancer and the expansion of healthcare access. Integrating and experimenting with a semi-supervised framework helps
with the goal of continuous development of AI tools that could have a major role in the medical
area.
