N_GPU=1
WDIR='densenetv53'
FOLD=5
SIZE='512'

bsub  -q lowpriority -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:apex_build \
            -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/recursion/scripts/$WDIR  && python3 few_shot_proba.py  \
            --logmsg fewshot- --embpath scripts/$WDIR    --workpath scripts/$WDIR  --dfname _df_site0_{}_probs_512_fold5.pk \
            --probsname emb_probs_$WDIR.csv.gz  --embname _emb_site0_{}_probs_512_fold5.pk --imgpath data/mount/512X512X6/"


bsub  -q lowpriority -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:apex_build \
            -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/recursion/scripts/$WDIR  && python3 few_shot_proba.py  \
            --logmsg fewshot- --embpath scripts/$WDIR    --workpath scripts/$WDIR  --dfname _df_site0_{}_probs_512_fold5.pk \
            --probsname cls_probs_$WDIR.csv.gz  --embname _cls_site0_{}_probs_512_fold5.pk --imgpath data/mount/512X512X6/"
