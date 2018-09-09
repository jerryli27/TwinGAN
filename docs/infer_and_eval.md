# Inference

We provide two pre-trained models: [human to anime](https://drive.google.com/open?id=1dXfqAODQxB2uNhyQANtZICAjwhNMWnbl) and [human to cats](https://drive.google.com/open?id=1UJEqlH_1sfdmWs6MXKV4H69NGad0rdUB)

Run the following command to translate the demo inputs. Note that this command works the best for models trained using twingan. You'll have to change this a little to make it work with image_generation models.

```
# Make sure you're under

python inference/image_translation_infer.py \
--model_path="/PATH/TO/MODEL/"
--image_hw=256  # 256 for anime, 128 for cats.
--input_tensor_name="sources_ph"
--output_tensor_name="custom_generated_t_style_source:0"
--input_image_path="./demo/inference_input/cropped"
--output_image_path="./demo/inference_output/temp"
```

# Evaluation

You can modify your training script into eval script by changing the following flags. (If you are using pggan_runner, it automatically sets the `checkpoint_path` and the `eval_dir` for you.)

```
--checkpoint_path="/PATH/TO/CHECKPOINT"
--eval_dir="/PATH/TO/SAVE/EVAL/RESULT"
--dataset_split_name=validation
--is_training=False
--do_custom_eval=True
--calc_swd=True
--use_tf_swd=False
--swd_num_images=8192
--swd_save_images=False
```

Here's an example script that evaluates the PGGAN output using msss.

```
python image_generation.py
--train_dir="/PATH/TO/CHECKPOINT"
--checkpoint_path="/PATH/TO/CHECKPOINT"
--eval_dir="/PATH/TO/SAVE/EVAL/RESULT"
--batch_size=32
--dataset_name="image_only"
--dataset_dir="/PATH/TO/DATASET"
--dataset_split_name=validation
--is_training=False
--train_image_size=32
--preprocessing_name="danbooru"
--generator_network="pggan"
--loss_architecture=dragan
--gradient_penalty_lambda=0.25
--do_custom_eval=True
--calc_swd=True
--use_tf_swd=False
--swd_num_images=8192
--swd_save_images=False
```

# FAQ

## ModuleNotFoundError

If you see `ModuleNotFoundError: No module named 'util_io'`, you're probably not under the TwinGAN directory, or you have not set PYTHONPATH correctly. See [Issue 10](https://github.com/jerryli27/TwinGAN/issues/10) and [3](https://github.com/jerryli27/TwinGAN/issues/3) for details.