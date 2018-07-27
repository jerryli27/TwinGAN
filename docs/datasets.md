# Datasets

We use tfrecord format data for this project.

## Celeba

See [convert_celeba.py](/datasets/convert_celeba.py) for how to download and convert the dataset.

## Getchu

The preprocessed dataset can be found [here](https://drive.google.com/open?id=1hhL9KynRneFN6LY4rhqLxjvXtNt2iA1u). We crop the images using an [anime face detector](https://github.com/nagadomi/lbpcascade_animeface). 

See the ["Towards the Automatic Anime Characters Creation with Generative Adversarial Networks"](https://arxiv.org/abs/1708.05509) paper for details.