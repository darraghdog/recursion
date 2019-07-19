N_GPU=1
WDIR='dieter'

bsub  -q lowpriority -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=docker.optum.com/dhanley2/bert:pytorch_build \
            -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/recursion/$WDIR  && python3 downsize_6channel.py  \
            --dimsize 128"

#bsub  -q lowpriority -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=docker.optum.com/dhanley2/bert:pytorch_build \
#            -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/recursion/$WDIR  && python3 downsize.py  \
#            "