# Train your model from scratch

## TwinGAN for unsupervised image translation

An example TwinGAN training script is shown below:

```
# Note: replace 'pggan_runner.py' with 'image_generation.py'
# if you would like to run each training stage manually.
python pggan_runner.py
--program_name=twingan
--dataset_name="image_only"
# Assume you have data like 
# ./data/celeba/train-00000-of-00100.tfrecord,  
# ./data/celeba/train-00001-of-00100.tfrecord ...
--dataset_dir="./data/celeba/"
--unpaired_target_dataset_name="anime_faces"
--unpaired_target_dataset_dir="./data/anime_faces/"
--train_dir="./checkpoints/twingan_faces/"
--dataset_split_name=train
--preprocessing_name="danbooru"
--resize_mode=RESHAPE
--do_random_cropping=True
--learning_rate=0.0001
--learning_rate_decay_type=fixed
--is_training=True
--generator_network="pggan"
--use_unet=True
--num_images_per_resolution=300000
--loss_architecture=dragan
--gradient_penalty_lambda=0.25
--pggan_max_num_channels=256
--generator_norm_type=batch_renorm
--hw_to_batch_size="{4: 8, 8: 8, 16: 8, 32: 8, 64: 8, 128: 4, 256: 3, 512: 2}"
--do_pixel_norm=True
--l_content_weight=0.1
--l_cycle_weight=1.0
```

Training to resolution 32x32 takes approximately half a day depending on the hardware. Full training to 256x256 can take up to a week or two.

## PGGAN

Different from TwinGAN, PGGAN is a generative model. That is to say, there is no "source image". It generates real-looking images from scratch conditioned on a random vector. Please read the [PGGAN paper](https://arxiv.org/abs/1710.10196) for more details. Note that we do not provide a completely faithful reproduction of the PGGAN paper. We do borrow their network structure as-is.

An example PGGAN training script is shown below:

```
python pggan_runner.py
--program_name=image_generation
--dataset_name="image_only"
# Assume you have data like 
# ./data/celeba/train-00000-of-00100.tfrecord,  
# ./data/celeba/train-00001-of-00100.tfrecord ...
--dataset_dir="./data/celeba/"
--dataset_use_target=True
--dataset_split_name=train
--train_dir="./checkpoints/pggan_celeba/"
--preprocessing_name="danbooru"
--resize_mode=RESHAPE
--do_random_cropping=True
--learning_rate=0.0001
--learning_rate_decay_type=fixed
--is_training=True
--generator_network="pggan"
--max_number_of_steps=600000
--loss_architecture=dragan
--gradient_penalty_lambda=0.25
--pggan_max_num_channels=512  # 256 also works emperically.
--generator_norm_type=batch_renorm
--do_pixel_norm=True
```


## After Training

After training finishes, you can 
- Evaluate the training result
- Use the model to infer on other images. 

See [infer_and_eval.md](infer_and_eval.md)

## Common questions

#### Training is too slow

Training speed varies across platforms, hardwares, and settings. For your reference, on a TitanV with batch size 16 and using PGGAN with DRAGAN on image resolution 4x4, the step-per-second is usually around 40.  

Are you using a gpu? Training using cpu is strongly discouraged and should only be used for debugging.

Do you have large images? Your cpu may be too overloaded with converting images to a smaller size. Try resizing your images when converting them to tfrecords.

Try use more preprocessing threads if your cpu is powerful enough. change `--num_readers` and `--num_preprocessing_threads`.

Try decreasing the batch size a little. But keep in mind that it may affect the output quality.

If none of those help, you can either try mixed precision training or if you have multiple GPUs, try running it on multiple GPUs (Not tested, but in theory it should work out of the box :) ).

#### What does the parameters mean?

Please take a look at flag definitions in [image_generation.py](/image_generation.py) or other related files.

#### Can I use my dataset?

Please take a look at [use your dataset](use_your_dataset.md).

#### Can I use multiple GPUs?

Yes, but the code does not officially support multi-gpu training and you may encounter some bugs.

To train with two GPUs on the same machine, you can add the following flags:
```
--sync_replicas=False
--replicas_to_aggregate=1
--num_clones=2
--worker_replicas=1
```

Note that the batch size is **per GPU**, therefore a batch size of 8 on two GPUs will be in effect batch size 16.

To only use one of the GPUs during training, you can make only gpu0 visible to tensorflow by adding `CUDA_VISIBLE_DEVICES=0`
