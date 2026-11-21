# SEM-based Classification of Microbial Colonisation on MFC Electrodes

Deep transfer learning and explainable AI (Grad-CAM) for classifying microbial colonisation on microbial fuel cell (MFC) electrodes from scanning electron microscopy (SEM) images.

## Overview

- Five-class colonisation scheme for SEM image patches
- Benchmarking of pretrained CNN architectures with cross-validation and fixed-split validation
- Comparison with traditional image-analysis baselines
- Grad-CAM explanations of model predictions

## Getting started

```bash
git clone https://github.com/Nasim-Amani/sem-mfc-biofilm-classification.git
cd sem-mfc-biofilm-classification
pip install -r requirements.txt

# Example: train EfficientNetB0 with 4-fold cross-validation
python -m src.train_cv --model EfficientNetB0 --data_dir data
```

## Data availability

The SEM dataset is not included in this repository. It is available from the authors on reasonable request.

## Citation

If you use this code, please cite it using the **"Cite this repository"** button on the right. The citation of the accompanying paper will be added after publication.

## License

This project is released under the [MIT License](LICENSE).

## Contact

**Nasim Amani**, [GitHub](https://github.com/Nasim-Amani)
