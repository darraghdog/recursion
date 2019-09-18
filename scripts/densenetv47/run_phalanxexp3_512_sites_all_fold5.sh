N_GPU=1
WDIR='densenetv47'
FOLD=5
SIZE='512'
PRIO='lowpriority'


for FILT in 'HUVEC' 'HEPG' 'RPE'
do
    for SITE in 1 2
    do
        bsub  -q $PRIO -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:apex_build \
            -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/recursion/scripts/$WDIR  && python3 phalanxexp3.py  \
            --logmsg Recursion-concat-$SIZE-fp16 --nbags 20  --epochs 300 --fold $FOLD  --lr 0.000025 --lrmult 20  --batchsize 16  --workpath scripts/$WDIR  \
            --probsname probs_$SIZE  --expfilter $FILT  --cutmix_prob 1.0 --precision half  --beta 1.0  --imgpath data/mount/512X512X6/ \
            --site $SITE --weightsname weights/pytorch_cut_model_512_densenet$FOLD.bin"
    done
done

FILT='U2OS'
for SITE in 1 2
    do
    bsub  -q $PRIO -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:apex_build \
            -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/recursion/scripts/$WDIR  && python3 phalanxexp3.py  \
            --logmsg Recursion-concat-$SIZE-fp16 --nbags 20  --epochs 600 --fold $FOLD  --lr 0.000025 --lrmult 20  --batchsize 16  --workpath scripts/$WDIR  \
            --probsname probs_$SIZE  --expfilter $FILT  --cutmix_prob 1.0 --precision half  --beta 1.0  --imgpath data/mount/512X512X6/ \
            --site $SITE --weightsname weights/pytorch_cut_model_512_densenet$FOLD.bin"
done
