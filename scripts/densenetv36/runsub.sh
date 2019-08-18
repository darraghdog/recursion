N_GPU=1
WDIR='densenetv36'

bsub  -q lowpriority -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:apex_build \
            -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/recursion/eda  && python3 post_processing_leak.py \
             ../scripts/$WDIR/probs_v36_fold5.csv ../scripts/$WDIR/sub_v36_fold5.csv"
