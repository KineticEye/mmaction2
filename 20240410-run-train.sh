NUM_GPUS=4
CONFIG="20240410-activity-dataset-encord-roboflow-merged-vit-small-p16_videomae-v2-clearml.py"
bash ./tools/dist_train.sh ../config/spatio-temporal-det/$CONFIG $NUM_GPUS
