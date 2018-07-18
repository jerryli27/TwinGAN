# Refactoring checklist

## Dataset
- [x] CelebA
- [x] Getchu

## Trainer

- [X] Base class
- [X] image_generation
- [X] image_translation_sc
- [ ] classifier(optional)
- [X] pggan runner
- [X] rename image_translation_sc to twin_gan.
- [X] Add command to do eval.
- [X] Add script to do inference
- [X] Double check all scripts provided works.

## Utilities

- [X] util_io
- [X] util_misc
- [X] pggan
- [X] pggan_utils
- [X] nn libs
- [X] refine_sketch script. Decide what to do with it.
- [X] neural style script.
- [X] object detection script.


## Interface.

- [X] Tutorial. (Postponed.)

## Tutorial

- [X] Dataset
- [X] Inference
- [X] Train
- [X] Clean classifier code.
- [X] Eval
- [X] Interface
- [X] Inference actual example images with trained model and used command.
- [ ] Add human-to-cat model.

## Other

## Bugs

- [ ] Random cropping during eval does not work, therefore the real image observed at training and at eval will be different.

