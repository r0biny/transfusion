RUN_NAME=mnist-sheer-0203-2328
NUM_STEPS=50000

python ~/Code/transfusion/inference/test_mnist_augment.py \
    --checkpoint ~/Code/transfusion/checkpoints/$RUN_NAME/step-$NUM_STEPS.pt \
    --output-dir ~/Code/transfusion/inference_results/$RUN_NAME