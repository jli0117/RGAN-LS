# RGAN-LS: Data-augmented landslide displacement forecasting using generative adversarial network

Accurate prediction of landslide displacement enables effective early warning and risk management. However, the scarcity of landslide on-site measurement data has hindered the development and application of data-driven models, such as novel machine learning (ML) models. To mitigate such challenges, this study proposes a novel framework using generative adversarial networks (GANs), a recent advance in generative artificial intelligence (AI), to augment limited data and improve the accuracy of landslide displacement prediction. A recurrent GAN model, RGAN-LS, is developed to generate realistic synthetic multivariate time series data mimicking the characteristics of real data. By harnessing the power of the generative AI approach, the data scarcity for training advanced ML models in predicting landslide displacement can be alleviated. More broadly, this opens new possibilities for generative AI in geohazard risk management for other research purposes.

## Getting Started

The instructions below will guide you through the process of setting up and running the RGAN-LS model on your local machine.

### Prerequisites

You will need Python 3.6+ and the following packages:

- Pandas  - 1.1.3
- Numpy - 1.21.2
- TensorFlow - 1.14.0
- Keras - 2.2.5
- Matplotlib - 3.3.3
- Scikit-learn - 0.23.2

### Configuration

You can modify the hyperparameters for training in the `params` dictionary within the main function. Expand `params_list` to include multiple hyperparameter configurations for grid search.

### Training

The training process involves:

1. Reading and normalizing the dataset.
2. Defining time steps for prediction.
3. Reshaping the data into sequences suitable for RNN input.
4. Training the RGAN-LS model with specified hyperparameters.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* https://github.com/jsyoon0823/TimeGAN
* https://github.com/olofmogren/c-rnn-gan
