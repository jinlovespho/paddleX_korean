import re
import os
import cv2
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paddlex import create_pipeline
from konlpy.tag import Okt, Mecab
import nltk 
from nltk.tokenize import word_tokenize, sent_tokenize
from pykospacing import Spacing
import argparse

# nltk.download('punkt')
# nltk.download('punkt_tab')

spacing = Spacing()
okt=Okt()

def restore_korean_spacing(text):
    return spacing(text)

def contains_korean(text):
    return any('\uac00' <= c <= '\ud7a3' for c in text)

def restore_korean_spacing(text):
    pos_tokens = okt.pos(text, stem=True)
    result = []
    for i, (word, tag) in enumerate(pos_tokens):
        result.append(word)
        if i < len(pos_tokens) - 1:
            next_tag = pos_tokens[i + 1][1]
            if next_tag in ['Noun', 'Verb', 'Adjective', 'Determiner', 'Adverb']:
                if tag not in ['Josa', 'Punctuation', 'Suffix']:
                    result.append(' ')
            elif next_tag == 'Punctuation':
                continue
            elif tag == 'Punctuation':
                result.append(' ')
    return ''.join(result).strip()

def get_fitting_font(text, box_width, box_height, font_path):
    max_font_size = min(int(box_height), 20)
    for size in range(max_font_size, 4, -1):
        font = ImageFont.truetype(font_path, size)
        bbox = font.getbbox(text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        if text_w <= box_width and text_h <= box_height:
            return font
    return ImageFont.truetype(font_path, 5)


def merge_words_by_symbol_break(words, img_w, img_h):
    merged_texts = []
    merged_boxes = []

    current_text = ""
    current_vertices = []

    def norm2abs(vertex):
        return int(vertex['x'] * img_w), int(vertex['y'] * img_h)

    def merge_bbox(vertices_list):
        xs = [v['x'] for box in vertices_list for v in box if 'x' in v]
        ys = [v['y'] for box in vertices_list for v in box if 'y' in v]
        if not xs or not ys:
            return 0, 0, 0, 0  # fallback safe box if no valid coords
        return (
            int(min(xs) * img_w), int(min(ys) * img_h),
            int(max(xs) * img_w), int(max(ys) * img_h)
        )

    for word in words:
        symbols = word.get("symbols", [])
        bbox = word.get("boundingBox", {}).get("normalizedVertices", [])
        if len(bbox) < 4 or any(('x' not in v or 'y' not in v) for v in bbox):
            continue

        for i, symbol in enumerate(symbols):
            current_text += symbol['text']
        current_vertices.append(bbox)

        # Check if current symbol has a break
        last_symbol = symbols[-1]
        break_type = last_symbol.get("property", {}).get("detectedBreak", {}).get("type", None)
        if break_type in ("SPACE", "LINE_BREAK", "EOL_SURE_SPACE") or word == words[-1]:
            # Finalize current merged word
            x0, y0, x1, y1 = merge_bbox(current_vertices)
            merged_texts.append(current_text)
            merged_boxes.append(((x0, y0), (x1, y1)))
            current_text = ""
            current_vertices = []

    return merged_texts, merged_boxes


def iou(boxA, boxB):
    xA = max(boxA[0][0], boxB[0][0])
    yA = max(boxA[0][1], boxB[0][1])
    xB = min(boxA[1][0], boxB[1][0])
    yB = min(boxA[1][1], boxB[1][1])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[1][0] - boxA[0][0] + 1) * (boxA[1][1] - boxA[0][1] + 1)
    boxBArea = (boxB[1][0] - boxB[0][0] + 1) * (boxB[1][1] - boxB[0][1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


def evaluate(gt_boxes, gt_texts, pred_boxes, pred_texts, iou_thresh=0.5):
    matched_gt = set()
    matched_pred = set()
    correct_text_count = 0

    for i, (p_box, p_text) in enumerate(zip(pred_boxes, pred_texts)):
        best_iou = 0
        best_j = -1
        for j, g_box in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou_val = iou(p_box, g_box)
            if iou_val > best_iou:
                best_iou = iou_val
                best_j = j

        if best_iou >= iou_thresh:
            matched_gt.add(best_j)
            matched_pred.add(i)
            if p_text.strip() == gt_texts[best_j].strip():
                correct_text_count += 1

    tp = len(matched_pred)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)

    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'Correct Text Matches': correct_text_count,
        'Detection Precision': tp / (tp + fp + 1e-6),
        'Detection Recall': tp / (tp + fn + 1e-6),
        'Recognition Accuracy': correct_text_count / (tp + 1e-6)
    }



# --- MAIN PIPELINE ---

parser = argparse.ArgumentParser()
parser.add_argument('--pipeline', type=str, help='pipeline type')
parser.add_argument('--korean_font_path', type=str, default='fonts/NanumGothic.ttf')
parser.add_argument('--test_imgs_path', type=str, help='path to testing images')
parser.add_argument('--test_anns_path', type=str, help='path to testing annotations')
parser.add_argument('--save_root_path', type=str)
parser.add_argument('--gpu', type=str)
args = parser.parse_args()


# set korean font
font_path = args.korean_font_path
font = ImageFont.truetype(font_path, size=15)


# load test data
test_imgs_path = args.test_imgs_path
test_anns_path = args.test_anns_path

test_imgs = sorted(os.listdir(test_imgs_path))
test_anns = sorted(os.listdir(test_anns_path))

assert len(test_imgs) == len(test_anns), 'check number of test imgs and anns'
num_test_imgs = len(test_imgs)

# load paddle
pipeline = create_pipeline(pipeline=args.pipeline, device=f'gpu:{args.gpu}')

det_model_name = pipeline._pipeline.text_det_model.model_name 
rec_model_name = pipeline._pipeline.text_rec_model.model_name

# set save path
SAVE_ROOT_PATH = f'{args.save_root_path}/{det_model_name}_{rec_model_name}'
save_gt_path = f'{SAVE_ROOT_PATH}/gt'
save_paddle_img_path = f'{SAVE_ROOT_PATH}/pred_imgs'
save_paddle_ann_path = f'{SAVE_ROOT_PATH}/pred_anns'

os.makedirs(save_gt_path, exist_ok=True)
os.makedirs(save_paddle_img_path, exist_ok=True)
os.makedirs(save_paddle_ann_path, exist_ok=True)


all_tp, all_fp, all_fn, all_correct = 0, 0, 0, 0

for img, ann in zip(test_imgs, test_anns):
    img_path = os.path.join(test_imgs_path, img)
    ann_path = os.path.join(test_anns_path, ann)
    img_id = os.path.splitext(img)[0]

    gt_texts, gt_boxes = [], []

    # --- Google OCR Ground Truth ---
    with open(ann_path, 'r') as f:
        ann_data = json.load(f)
    if 'fullTextAnnotation' in ann_data['responses'][0]:
        blocks = ann_data['responses'][0]['fullTextAnnotation']['pages'][0]['blocks']
        img_pil = Image.open(img_path)
        img_w, img_h = img_pil.size
        canvas = Image.new("RGB", (img_w, img_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        gt_texts, gt_boxes = [], []

        for blk in blocks:
            for para in blk.get("paragraphs", []):
                merged_texts, merged_boxes = merge_words_by_symbol_break(para.get("words", []), img_w, img_h)
                gt_texts.extend(merged_texts)
                gt_boxes.extend(merged_boxes)

        for text, ((x0, y0), (x1, y1)) in zip(gt_texts, gt_boxes):
            draw.rectangle([x0, y0, x1, y1], outline='black', width=2)
            draw.text((x0, y0), text, fill='black', font=get_fitting_font(text, x1 - x0, y1 - y0, font_path=font_path))

        # Combine original image and the canvas side-by-side
        combined = Image.new("RGB", (img_w * 2, img_h), (255, 255, 255))
        combined.paste(img_pil, (0, 0))
        combined.paste(canvas, (img_w, 0))
        combined.save(f'{save_gt_path}/googleocr_{img_id}.jpg')

    # --- PaddleOCR Predictions ---
    output = pipeline.predict(
        input=img_path,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    pred_texts, pred_boxes = [], []
    # Side-by-side visualization
    img_pil = Image.open(img_path)
    img_w, img_h = img_pil.size
    canvas = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for res in output:
        for text, poly in zip(res['rec_texts'], res['rec_polys']):
            words = text.split()
            if not words:
                continue

            box = poly.tolist()
            x1, y1 = box[0]
            x2, y2 = box[2]

            width = x2 - x1
            height = y2 - y1

            n = len(words)
            if abs(width) >= abs(height):
                # Horizontal text → split horizontally
                step = width / n
                for i in range(n):
                    new_x1 = x1 + i * step
                    new_x2 = x1 + (i + 1) * step
                    pred_texts.append(words[i])
                    pred_boxes.append(((new_x1, y1), (new_x2, y2)))
            else:
                # Vertical text → split vertically
                step = height / n
                for i in range(n):
                    new_y1 = y1 + i * step
                    new_y2 = y1 + (i + 1) * step
                    pred_texts.append(words[i])
                    pred_boxes.append(((x1, new_y1), (x2, new_y2)))

        res.save_to_json(save_path=f"{save_paddle_ann_path}/paddleocr_{img_id}.json")

    pred_texts2 = []
    pred_boxes2 = []

    for text, box in zip(pred_texts, pred_boxes):
        if contains_korean(text):
            # Use PyKoSpacing for better spacing restoration
            spaced_text = spacing(text)
            tokens = spaced_text.split()
            if not tokens:
                continue

            (x1, y1), (x2, y2) = box
            width = x2 - x1
            height = y2 - y1
            n = len(tokens)

            if abs(width) >= abs(height):
                # Horizontal splitting
                step = width / n
                for i, token in enumerate(tokens):
                    new_x1 = x1 + i * step
                    new_x2 = x1 + (i + 1) * step
                    pred_texts2.append(token)
                    pred_boxes2.append(((new_x1, y1), (new_x2, y2)))
            else:
                # Vertical splitting
                step = height / n
                for i, token in enumerate(tokens):
                    new_y1 = y1 + i * step
                    new_y2 = y1 + (i + 1) * step
                    pred_texts2.append(token)
                    pred_boxes2.append(((x1, new_y1), (x2, new_y2)))
        else:
            pred_texts2.append(text)
            pred_boxes2.append(box)
    
    
    pred_texts = pred_texts2
    pred_boxes = pred_boxes2
        
    # --- Evaluate ---
    result = evaluate(gt_boxes, gt_texts, pred_boxes, pred_texts)
    all_tp += result['TP']
    all_fp += result['FP']
    all_fn += result['FN']
    all_correct += result['Correct Text Matches']
    
    matched_gt = set()
    color_map = []

    for i, (p_box, p_text) in enumerate(zip(pred_boxes, pred_texts)):
        best_iou = 0
        best_j = -1
        for j, g_box in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou_val = iou(p_box, g_box)
            if iou_val > best_iou:
                best_iou = iou_val
                best_j = j

        if best_iou >= 0.1:
            matched_gt.add(best_j)
            if p_text.strip() == gt_texts[best_j].strip():
                color = 'green'  # TP + correct recognition
            else:
                color = 'orange'  # TP + wrong recognition
        else:
            color = 'red'  # FP

        color_map.append(color)

    # Draw each box with its assigned color
    for (p0, p1), text, color in zip(pred_boxes, pred_texts, color_map):
        # Normalize coordinates
        x0, y0 = min(p0[0], p1[0]), min(p0[1], p1[1])
        x1, y1 = max(p0[0], p1[0]), max(p0[1], p1[1])
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        # Adjust font fitting for normalized box
        font = get_fitting_font(text, x1 - x0, y1 - y0, font_path=font_path)
        draw.text((x0, y0), text, fill=color, font=font)


    # Combine side-by-side
    combined = Image.new("RGB", (img_w * 2, img_h), (255, 255, 255))
    combined.paste(img_pil, (0, 0))
    combined.paste(canvas, (img_w, 0))
    combined.save(f"{save_paddle_img_path}/paddleocr_{img_id}.jpg")
    
    
    # --- Detailed Logging ---
    log_path = f"{SAVE_ROOT_PATH}/pred_logs"
    os.makedirs(log_path, exist_ok=True)
    with open(f"{log_path}/log_{img_id}.txt", "w", encoding="utf-8") as log_file:
        log_file.write(f"=== Detailed Recognition Results for {img_id} ===\n\n")
        log_file.write(f"--- Predictions ---\n")
        for i, (p_box, p_text) in enumerate(zip(pred_boxes, pred_texts)):
            match_status = color_map[i]
            if match_status == 'green':
                note = "⭕️"
            elif match_status == 'orange':
                note = "❌"
            else:
                note = "❓"
            log_file.write(f"[{i:02d}] | {note} Text: '{p_text}' | Box: {p_box}\n")

        log_file.write(f"\n--- Ground Truths ---\n")
        for j, (gt_text, gt_box) in enumerate(zip(gt_texts, gt_boxes)):
            log_file.write(f"[{j:02d}] GT Text: '{gt_text}' | Box: {gt_box}\n")

    print(f"[{img_id}] Precision: {result['Detection Precision']:.3f}, "
          f"Recall: {result['Detection Recall']:.3f}, "
          f"RecAcc: {result['Recognition Accuracy']:.3f}")
    print(f"→ Detailed log saved to {log_path}/log_{img_id}.txt")

# --- Final Summary ---
precision = all_tp / (all_tp + all_fp + 1e-6)
recall = all_tp / (all_tp + all_fn + 1e-6)
rec_acc = all_correct / (all_tp + 1e-6)
f1_det = 2 * precision * recall / (precision + recall + 1e-6)

print("\n=== Overall Evaluation ===")
print(f"Total True Positives: {all_tp}")
print(f"Total False Positives: {all_fp}")
print(f"Total False Negatives: {all_fn}")
print(f"Total Correct Text Matches: {all_correct}")
print(f"Detection Precision: {precision:.3f}")
print(f"Detection Recall: {recall:.3f}")
print(f'Detection F1-score: {f1_det:.3f}')
print(f"Recognition Accuracy: {rec_acc:.3f}")


# --- Save Overall Summary ---
summary_path = f"{SAVE_ROOT_PATH}"
os.makedirs(summary_path, exist_ok=True)
summary_file = os.path.join(summary_path, f"pred_summary_for_{num_test_imgs}.txt")

with open(summary_file, "w", encoding="utf-8") as f:
    f.write("=== Overall Evaluation ===\n")
    f.write(f"Total True Positives: {all_tp}\n")
    f.write(f"Total False Positives: {all_fp}\n")
    f.write(f"Total False Negatives: {all_fn}\n")
    f.write(f"Total Correct Text Matches: {all_correct}\n")
    f.write(f"Detection Precision: {precision:.3f}\n")
    f.write(f"Detection Recall: {recall:.3f}\n")
    f.write(f"Detection F1-score: {f1_det:.3f}\n")
    f.write(f"Recognition Accuracy: {rec_acc:.3f}\n")

print(f"→ Overall summary saved to {summary_file}")