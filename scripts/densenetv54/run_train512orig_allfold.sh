N_GPU=4
WDIR='densenetv54'
FOLD=5
SIZE='512'


bsub  -q low2 -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:apex_build \
            -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/recursion/scripts/$WDIR  && python3 trainorig.py  \
            --logmsg Recursion-concat-$SIZE-fp16 --nbags 10  --epochs 150 --fold $FOLD  --lr 0.00005 --lrmult 30 --accum 4  --batchsize 16  --workpath scripts/$WDIR  \
            --probsname probs_$SIZE  --cutmix_prob 1.0 --precision half  --beta 1.0  --imgpath data/mount/512X512X6/ --weightsname weights/pytorch_cut_model_512_densenet$FOLD.bin"
