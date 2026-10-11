# Deep Learning AtoZ
A collection of various deep learning architectures, models, tips ,code snippets, and mini projects for TensorFlow and PyTorch. Includes topics like Computer Vision, Natural Language Processing (NLP), Recurrent Neural Networks (RNNs) and more.

## PyTorch Basics
| Title | Description | Tool/Library | Link |
|-------|-------------|--------------|------|
| Linear Regression from Scratch | Build linear regression with raw tensors and autograd, then compare with `torch.nn` | PyTorch | [Link](pytorch-basic/Linear_Regression_from_Scratch_using_PyTorch.ipynb) |
| Logistic Regression | Logistic regression classifier with PyTorch | PyTorch | [Link](pytorch-basic/Logistic_Regression_with_PyTorch.ipynb) |
| Feedforward Neural Network | A basic feedforward neural network | PyTorch | [Link](pytorch-basic/pytorch_feedforward_neural_network.ipynb) |
| Autograd | Automatic differentiation in PyTorch | PyTorch | [Link](pytorch-basic/Autograd.ipynb) |
| Autograd with Tensors | Working with autograd on tensors | PyTorch | [Link](pytorch-basic/Autograd_with_tensors.ipynb) |
| Optimizers | Overview of PyTorch optimizers | PyTorch | [Link](pytorch-basic/Optimizers.ipynb) |
| CPU to GPU | Moving tensors and models between CPU and GPU | PyTorch | [Link](pytorch-basic/CPU_to_GPU.ipynb) |
| Validation | Adding a validation loop to training | PyTorch | [Link](pytorch-basic/Validation.ipynb) |
| Visualising Network Loss | Plotting and inspecting training loss | PyTorch | [Link](pytorch-basic/Visualising_the_network_loss.ipynb) |
| Working with FMNIST | Training on the Fashion-MNIST dataset | PyTorch | [Link](pytorch-basic/Working_with_the_FMNIST_dataset.ipynb) |
| Troubleshooting | Common issues and fixes | PyTorch | [Link](pytorch-basic/Troubleshooting.ipynb) |
| PyTorch Playground | Scratch space for experiments | PyTorch | [Link](pytorch-basic/PyTorch_Playground.ipynb) |

## TensorFlow Basics
| Title | Description | Tool/Library | Link |
|-------|-------------|--------------|------|
| Approximating a Sine Function | Fit a neural network to approximate `sin(x)` | TensorFlow | [Link](tf-basic/sine_function.ipynb) |
| TensorFlow 2 Basic Operations | Introduction to TensorFlow 2 tensors and operations | TensorFlow | [Link](tf-basic/tf2_basic_operation.ipynb) |

## Projects
| Title | Description | Tool/Library | Link |
|-------|-------------|--------------|------|
| Binary Image Classifier (Cats vs Dogs) | Train a CNN to classify cat vs dog images, with train/validation split, metric plots, and single-image inference | PyTorch | [Link](projects/Binary_Image_Classifier_Cats_vs_Dogs.ipynb) |

## Functional API (code)
- [MLP with the Functional API](code/mlp_functional_api.py)
- [CNN with the Functional API](code/cnn_functional_api.py)
- [LSTM with the Functional API](code/lstm_functional_api.py)

## Other
- [Scale Image Pixel Data](notebook/Scale_Image_Pixel_Data_for_Deep_Learning.ipynb)
- [Guide to Deal with Text Data](notebook/a_gentle_guide_to_deal_with_text_data.ipynb)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This collection builds on the work and documentation of the open-source deep learning community:

- [PyTorch](https://pytorch.org/) — framework powering the notebooks in `pytorch-basic/`.
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) — used for the Functional API examples in `code/` and the `tf-basic/` notebooks.
- [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/), and [Matplotlib](https://matplotlib.org/) — for data handling and visualization.
- [Project Jupyter](https://jupyter.org/) — the notebook environment used throughout.
- The broader deep learning community for the tutorials, datasets, and reference material that inspired these examples.
