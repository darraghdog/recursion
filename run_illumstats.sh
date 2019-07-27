N_GPU=1
bsub  -q lowpriority -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=docker.optum.com/dhanley2/bert:cgan_build \
            -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/recursion/eda/  && python3 illumination_stats.py.py  \
            --imgpath data/mount/256X256X6/"
