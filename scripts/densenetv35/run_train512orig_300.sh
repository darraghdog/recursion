N_GPU=4
WDIR='densenetv35'
FOLD=0
SIZE='512'

bsub  -q lowpriority -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:apex_build \
            -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/recursion/scripts/$WDIR  && python3 trainorig.py  \
            --logmsg Recursion-exptconcat-$SIZE --nbags 20  --epochs 100 --fold $FOLD  --lr 0.000025 --lrmult 20  --batchsize 64  --workpath scripts/$WDIR  \
            --probsname probs_$SIZE  --cutmix_prob 1.0 --precision half  --beta 1.0  --imgpath data/mount/512X512X6/ --weightsname weights/pytorch_cut_model_512_densenet$FOLD.bin"

#bsub  -q lowpriority -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:apex_build \
#            -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/recursion/scripts/$WDIR  && python3 trainorig.py  \
#            --logmsg Recursion-addexpt-cut0-$SIZE --nbags 20  --epochs 100 --fold $FOLD  --lr 0.000025 --lrmult 20  --batchsize 64  --workpath scripts/$WDIR  \
#            --probsname probs_$SIZE  --cutmix_prob 0.0 --precision half  --beta 1.0  --imgpath data/mount/256X256X6/ --weightsname weights/pytorch_cut_model_256_densenet$FOLD.bin"
