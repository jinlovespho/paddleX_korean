# python test.py \
#     --pipeline OCR \
#     --korean_font_path fonts/NanumGothic.ttf \
#     --test_imgs_path /SET/PATH/TO/TESTING/IMAGES \
#     --test_anns_path /SET/PATH/TO/TESTING/ANNOTATIONS \
#     --save_root_path /SET/PATH/TO/SAVING/RESULTS \
#     --gpu 1 \


# example script
python test.py \
    --pipeline OCR \
    --korean_font_path fonts/NanumGothic.ttf \
    --test_imgs_path /media/dataset1/jinlovespho/ocr_plantynet/data/filtered_train_test/test_images \
    --test_anns_path /media/dataset1/jinlovespho/ocr_plantynet/data/filtered_train_test/test_anns \
    --save_root_path ./test_results/trained_det_rec \
    --gpu 1 \
