N_GPU=1
WDIR='densenetv51'
SIZE='512'


FOLD=3
for FILTER in '1U2OS' '2U2OS' '3U2OS' '4U2OS'
do
    bsub  -q lowpriority -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:apex_build \
            -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/recursion/scripts/$WDIR  && python3 trainexpgrp.py  \
            --logmsg Recursion-longfinetune-$SIZE-fp16 --nbags 20  --epochs 200 --fold $FOLD  --lr 0.00025  --lrmult 20  --batchsize 16  --workpath scripts/$WDIR  \
            --xtrasteps 500 --expfilter $FILTER --probsname probs_$SIZE  --cutmix_prob 1.0 --precision half  --beta 1.0  --imgpath data/mount/512X512X6/ \
            --weightsname weights/pytorch_cut_model_512_densenet$FOLD.bin"
done
