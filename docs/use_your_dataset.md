# How to use your own dataset

## Convert to tfrecord

Assume you have a directory containing the images you would like to use. You would need to convert them into tfrecord format first. Converting to tfrecord has some benefits, including easier preprocessing on cpu without blocking gpu computation.

```
# Note this script by default only converts jpg, jpeg, and png images.
python datasets/convert_image_only.py \
--train_directory="/REPLACE/THIS/WITH/TRAIN/DIR/" \
--validation_directory="/REPLACE/THIS/WITH/VAL/DIR/" \
--output_directory="/REPLACE/THIS/WITH/OUTPUT/DIR/" \
--train_shards=8 \
--validation_shards=2 \
--num_threads=2
```

## Use your converted tfrecord

Now try your dataset out! Point `dataset_dir` to your dataset. For example if you want to train a PGGAN on your dataset, use the following command. Note this code is just an example. See [Train your model from scratch](training.md) for details on how to train the model.

```
python image_generation.py
--dataset_name="image_only"
--dataset_dir="/REPLACE/THIS/WITH/DIR/TO/TFRECORDS/"
--dataset_use_target=True
--train_dir="./checkpoints/temp/"
--dataset_split_name=train
--learning_rate=0.0001
--learning_rate_decay_type=fixed
--is_training=True
--train_image_size=4
--preprocessing_name="danbooru"
--generator_network="pggan"
--ignore_missing_vars=True
--max_number_of_steps=50000
```

## Standard datasets

Slim provides a few standard datasets such as imagenet [here](https://github.com/tensorflow/models/tree/master/research/slim/datasets).

## More customized dataset

In general if your dataset only contains image data, you are all set by using flags as below:

```
--dataset_name="image_only"
--dataset_dir="./REPLACE/THIS/WITH/DIR/TO/TFRECORDS/"
```

However if you would like more, e.g. if your dataset contains labels for each image, take a look at [convert_celeba.py](/datasets/convert_celeba.py) or  [convert_anime_faces.py](/datasets/convert_anime_faces.py) for examples of how to convert more than an image.

You may also want to write your own dataset class. Take a look at any classes inside [dataset_factory.py](/datasets/dataset_factory.py), define your own class, and add it to the **'datasets_map'** inside dataset_factory. After that, you can just change the `dataset_name` flag to your dataset name to start using it.


