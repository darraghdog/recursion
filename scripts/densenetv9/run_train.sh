N_GPU=1
WDIR='densenetv9'
FOLD=0
bsub  -q lowpriority -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=docker.optum.com/dhanley2/bert:pytorch_build \
            -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/recursion/scripts/$WDIR  && python3 train.py  \
            --nbags 5  --epochs 30 --fold $FOLD  --lr 0.001 --batchsize 16 --workpath scripts/$WDIR  --weightsname weights/pytorch_model_densenet$FOLD.bin"
