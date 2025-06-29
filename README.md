# Monet Style Transfer with CycleGANs

This repository contains a CycleGAN-based deep learning project to transform real-world landscape photos into Monet-style paintings. The task was completed as part of a **peer-graded assignment** for the **Introduction to Deep Learning** course and submitted to the [Kaggle competition: GANs Getting Started](https://www.kaggle.com/competitions/gan-getting-started).

## Files

**Included in this repository:**
- `gan_monet_kaggle_assignment.ipynb`: Main Jupyter notebook containing data validation, model architecture, training code for multiple CycleGAN variants, image generation, and Kaggle submission preparation.
- `gan_monet_kaggle_assignment.pdf`: PDF version of the full notebook run including EDA, model training, evaluation, and final results — ideal for quick review without setting up the environment.
> This notebook is designed to be reproducible and includes all steps from training to evaluation. Some sections were run on Kaggle for GPU support, and corresponding model files were reused locally for reporting and submission.

## Model Variants

We trained and compared the following CycleGAN variants:

1. **Baseline Model**  
   - 2 ResNet blocks in the generator  
   - Standard training with `@tf.function` and Adam optimizer (learning rate = 0.0002)

2. **Variant 1**  
   - 6 ResNet blocks in the generator  
   - Trained same as baseline for better style transfer fidelity

3. **Variant 2**  
   - 1 ResNet block  
   - Trained using **eager-mode warm-up**, **instance normalization**, and **label smoothing** for improved training stability

## Results

Each model was evaluated by submitting generated images to Kaggle to receive a **MiFID score**, which measures similarity to real Monet paintings. Side-by-side visual results and scores are included in the notebook.

## Requirements

- Python 3.10+
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- PIL (Pillow)

Install the necessary packages with:

```bash
pip install tensorflow pandas numpy matplotlib pillow
