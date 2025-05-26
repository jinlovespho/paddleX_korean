# 🪴 PaddleX for Korean OCR (with PlantyNet)

Fine-tuned PaddleX models for Korean OCR in collaboration with PlantyNet.

---

## ⚙️ Dependencies and Installation
```
# clone repo 
git clone https://github.com/jinlovespho/paddleX_korean.git
cd paddleX_korean

# create conda env
conda create -n paddlex python=3.9 -y 
conda activate paddlex 

# install paddle libraries
pip install paddlepaddle 
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
pip install -e .
paddlex --install PaddleOCR
pip install paddlex[ocr]==3.0


# install additional libraries
pip install albucore==0.0.16
pip install konlpy 
pip install git+https://github.com/haven-jeon/PyKoSpacing.git
pip install numpy==1.26.4 
pip install nltk

# additionally download (if error appears)
nltk.download('punkt')
nltk.download('punkt_tab')
```

## 📂 Dataset Structure
```
ocr_dataset/
    └── test_images/
        └── 000000030633-01_007.png 
        └── ...
    └── test_anns/
        └── 000000030633-01_007.json
        └── ...
```

## 🚀 Evaluation


#### Step1. Open OCR yaml file 
```
paddlex/configs/pipelines/OCR.yaml
```

#### Step2. Modify detection/recognition model_name and model_dir in the yaml file
```
.
.
SubModules:
  TextDetection:
    .
    .
    model_name: PP-OCRv5_server_det
    model_dir: ckpt/det_ckpt/best_accuracy/inference
    .
    .
  TextRecognition:
    .
    .
    model_name: korean_PP-OCRv3_mobile_rec
    model_dir: ckpt/rec_ckpt/best_accuracy/inference
    .
    .
```

#### Step3. Run eval bash script
```
bash scripts/test/run_test.sh
```

---

#### Bash script explanation
````
python test.py \
    --pipeline OCR \
    --korean_font_path fonts/NanumGothic.ttf \
    --test_imgs_path /SET/PATH/TO/TESTING/IMAGES \
    --test_anns_path /SET/PATH/TO/TESTING/ANNOTATIONS \
    --save_root_path /SET/PATH/TO/SAVING/RESULTS \
    --gpu 1 \

# example usage
python test.py \
    --pipeline OCR \
    --korean_font_path fonts/NanumGothic.ttf \
    --test_imgs_path ocr_dataset/test_images \
    --test_anns_path ocr_dataset/test_anns \
    --save_root_path ./test_results/trained_det_rec \
    --gpu 1 \
````






## Acknowledgments
This project is based on [PaddleX](https://github.com/PaddlePaddle/PaddleX). Thanks for their awesome works 

## Contact
If you have any questions, please feel free to contact: `msjchr@korea.ac.kr`
