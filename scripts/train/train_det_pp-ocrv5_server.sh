python main.py -c paddlex/configs/modules/text_detection/PP-OCRv5_server_det.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=/media/dataset1/jinlovespho/ocr_plantynet/data/det_training_data \
    -o Global.output=./train_results/det/v5_server_train_gpu23_ep200_bs16_lr1e-3 \
    -o Global.device=gpu:2,3


# python main.py -c /PATH/TO/RECOGNIZER/CONFIG/YAML/FILE \
#     -o Global.mode=train \
#     -o Global.dataset_dir=/PATH/TO/RECOGNIZER/TRAINING/DATA \
#     -o Global.output=/SAVING/PATH/DIRECTORY \
#     -o Global.device=gpu:2,3

