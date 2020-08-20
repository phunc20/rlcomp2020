#workon rlcomp2020
python train.py | tee train-log-$(date +%D-%Hh%Mm%Ss | tr / -).txt
