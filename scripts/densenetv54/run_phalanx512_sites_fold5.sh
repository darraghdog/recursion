N_GPU=2
WDIR='densenetv54'
FOLD=5
SIZE='512'


#bsub  -q lowpriority -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:apex_build \
#            -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/recursion/scripts/$WDIR  && python3 phalanx.py  \
#            --logmsg Recursion-concat-$SIZE-fp16 --nbags 30  --epochs 100 --fold $FOLD  --lr 0.000025 --lrmult 20  --batchsize 16  --workpath scripts/$WDIR  \
#            --site 1 --emb 0 --probsname probs_$SIZE  --cutmix_prob 1.0 --precision half  --beta 1.0  --imgpath data/mount/512X512X6/ --weightsname weights/pytorch_cut_model_512_densenet$FOLD.bin"

for site in 0
do
    bsub  -q lowpriority -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:apex_build \
            -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/recursion/scripts/$WDIR  && python3 phalanx.py  \
            --logmsg Recursion-concat-$SIZE-fp16 --nbags 30  --epochs 150 --fold $FOLD  --lr 0.000025 --lrmult 20  --batchsize 16  --workpath scripts/$WDIR  \
            --site 2 --emb 0 --probsname probs_$SIZE  --cutmix_prob 1.0 --precision half  --beta 1.0  --imgpath data/mount/512X512X6/ --weightsname weights/pytorch_cut_model_512_densenet$FOLD.bin"
done
