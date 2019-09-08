N_GPU=1
WDIR='densenetv47'
SIZE='512'
PRIO='lowpriority'


for FILTER in '1U2OS' '2U2OS' '3U2OS' '4U2OS'
do
    for FOLD in 5
    do
        bsub  -q $PRIO -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:apex_build \
            -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/recursion/scripts/$WDIR  && python3 phalanxexpgrp.py  \
            --logmsg Recursion-concat-$SIZE-fp16 --nbags 20  --epochs 800 --fold $FOLD  --lr 0.000025 --lrmult 20  --batchsize 16  --workpath scripts/$WDIR  \
            --probsname probs_$SIZE  --expfilter $FILTER  --cutmix_prob 1.0 --precision half  --beta 1.0  --imgpath data/mount/512X512X6/ \
            --weightsname weights/pytorch_cut_model_512_densenet$FOLD.bin"
    done
done

for FILTER in '1U2OS' '2U2OS' '3U2OS' '4U2OS'
do
    for FOLD in 1 2 3
    do
        bsub  -q $PRIO -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:apex_build \
            -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/recursion/scripts/$WDIR  && python3 phalanxexpgrp.py  \
            --logmsg Recursion-concat-$SIZE-fp16 --nbags 20  --epochs 400 --fold $FOLD  --lr 0.000025 --lrmult 20  --batchsize 16  --workpath scripts/$WDIR  \
            --probsname probs_$SIZE  --expfilter $FILTER  --cutmix_prob 1.0 --precision half  --beta 1.0  --imgpath data/mount/512X512X6/ \
            --weightsname weights/pytorch_cut_model_512_densenet$FOLD.bin"
    done
done