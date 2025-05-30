python main.py -c paddlex/configs/modules/text_recognition/korean_PP-OCRv3_mobile_rec.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=/media/dataset1/jinlovespho/ocr_plantynet/data/rec_training_data \
    -o Global.output=./train_results/rec/v5_server_train_gpu0123_ep100_bs256_lr1e-3 \
    -o Global.device=gpu:0,1,2,3


# python main.py -c /PATH/TO/RECOGNIZER/CONFIG/YAML/FILE \
#     -o Global.mode=train \
#     -o Global.dataset_dir=/PATH/TO/RECOGNIZER/TRAINING/DATA \
#     -o Global.output=/SAVING/PATH/DIRECTORY \
#     -o Global.device=gpu:0,1,2,3

