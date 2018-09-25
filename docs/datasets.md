# Datasets

We use tfrecord format data for this project.

## Celeba

See [convert_celeba.py](/datasets/convert_celeba.py) for how to download and convert the dataset.

## Getchu

The preprocessed dataset can be found [here](https://drive.google.com/open?id=1hhL9KynRneFN6LY4rhqLxjvXtNt2iA1u). We crop the images using an [anime face detector](https://github.com/nagadomi/lbpcascade_animeface). 

See the ["Towards the Automatic Anime Characters Creation with Generative Adversarial Networks"](https://arxiv.org/abs/1708.05509) paper for details

## Cats

Training set contains 8402 images and validation set contains 100 images. The dataset is generated from a modified version of [Deep learning with cats](https://github.com/AlexiaJM/Deep-learning-with-cats)

The preprocessed dataset can be found [here](https://drive.google.com/file/d/1mWsIBq3mtU0KcVNjLc65fpQ4Cw2J8BdP/view?usp=sharing).