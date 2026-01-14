RUN_NAME=run-mnist-1216-1345
NUM_STEPS=50000

python ./test_mnist_with_unet.py \
    --checkpoint ../checkpoints/$RUN_NAME/step-$NUM_STEPS.pt \
    --num-random 10 \
    --digits 0 0 0 0 0 0 0 0\
    --output-dir ../inference_results/$RUN_NAME