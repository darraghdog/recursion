N_GPU=1
WDIR='dieter'
bsub  -q lowpriority -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=docker.optum.com/dhanley2/bert:cgan_build \
            -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/recursion/$WDIR  && python3 train_HEPG2_6channel.py  \
            --epochs 12 --normfile experiment_normalizations256X256X6.p --dimsize 256 --datapath data/mount/256X256X6"

#bsub  -q lowpriority -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=docker.optum.com/dhanley2/bert:cgan_build \
#            -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/recursion/$WDIR  && python3 train_HEPG2.py  \
#            --dimsize 128"
