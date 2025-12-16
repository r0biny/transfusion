RUN_NAME=run-mnist-1212-1609
NUM_STEPS=42000

# python ./test_mnist_with_unet.py \
#     --checkpoint ../checkpoints/$RUN_NAME/step-$NUM_STEPS.pt \
#     --num-random 10 \
#     --output-dir ../inference_results/$RUN_NAME

python ./test_mnist_with_unet.py \
    --checkpoint ../checkpoints/$RUN_NAME/step-$NUM_STEPS.pt \
    --digits 8 8 8 8 \
    --output-dir ../inference_results/$RUN_NAME