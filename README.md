# ST2S: A Spatiotemporal Two-Stage Learning Framework for Pulse Estimation

## Overview
This repository contains the implementation of **ST2S**, a spatiotemporal two-stage learning framework for remote photoplethysmography (rPPG)-based heart rate estimation. The framework introduces a novel spatiotemporal representation of video data and a two-stage learning approach to improve pulse estimation accuracy by selecting the most informative input images for the regression model.

**Note:** The accompanying paper for this work, *"ST2S-rPPG: A Spatiotemporal Two-Stage Learning Approach for Pulse Estimation Using Video"*, is currently under review. Details and updates will be shared upon publication.

---

## Repository Contents
- **`CNN.py`**: Implementation of the Convolutional Neural Network (CNN) used for pulse regression.
- **`classifier.py`**: Multi-Layer Perceptron (MLP) classifier for second-stage learning to classify "good" and "bad" images.
- **`binary_class_generation.py`**: Script for generating the binary dataset based on MAE thresholds.
- **`st_image_generator.py`**: Generates spatiotemporal images from video segments.
- **`st_cropping.py`**: Crops stabilized facial regions from videos.
- **`train.py`**: Training script for the CNN.
- **`prediction_only.py`**: Script for evaluating the trained model on test datasets.
- **`result_analysis.py`**: Tools for analyzing and visualizing results.
- **`config.yaml`**: Configuration file for managing training parameters.

---

## Setup
### Prerequisites
- Python 3.8+
- Recommended: NVIDIA GPU with CUDA support for training.

## Datasets
The framework is evaluated on the following datasets:

- MMSE-HR: Zhang, Zheng, et al. "Multimodal spontaneous emotion corpus for human behavior analysis." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
- UBFC-rPPG: Sabour, Rita Meziati, et al. "Ubfc-phys: A multimodal database for psychophysiological studies of social stress." IEEE Transactions on Affective Computing 14.1 (2021): 622-636.

Note: Due to licensing restrictions, datasets are not included in this repository. Refer to the respective dataset authors for access.

## License

This code is released under the [MIT License](LICENSE).

## Citation

If using this code or methodology, please cite the paper:

```bibtex
@article{TBA,
  title={ST2S-rPPG: A Spatiotemporal Two-Stage Learning Approach for Pulse Estimation Using Video},
  author={TBA},
  journal={TBA},
  year={2024}
}
