import io
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms, models
from ultralytics import YOLO
import numpy as np

# -------------------------
# Helper functions
# -------------------------
def xyxy_to_int(xyxy):
    return [int(x) for x in xyxy]

def iou(boxA, boxB):
    # boxes in [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou_val = interArea / float(boxAArea + boxBArea - interArea)
    return iou_val

def draw_label(draw, box, text, font, rect_color=(0, 255, 0)):
    x1, y1, x2, y2 = box

    # Use textbbox to compute text width/height (compatible with Pillow 10+)
    try:
        text_bbox = draw.textbbox((x1, y1), text, font=font)
    except Exception:
        # fallback - measure at origin
        text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    padding = 4
    # Background rectangle for text - prefer above the box if space
    bg_x1 = x1
    bg_x2 = x1 + text_width + 2 * padding
    bg_y2 = y1
    bg_y1 = y1 - (text_height + 2 * padding)
    if bg_y1 < 0:
        # not enough space above, place inside box top
        bg_y1 = y1
        bg_y2 = y1 + text_height + 2 * padding

    draw.rectangle((bg_x1, bg_y1, bg_x2, bg_y2), fill=rect_color)
    text_x = bg_x1 + padding
    text_y = bg_y1 + padding
    draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)

# -------------------------
# Model & device setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# COCO pretrained YOLO to detect people and generic objects
coco_model = YOLO("yolov8n.pt")  # ensure this file is available in working dir or path

# Your custom YOLO trained for Bird/Drone
custom_yolo = YOLO("yolov8/yolov8_best.pt")  # path to your custom weights

# ResNet50 classifier (Bird / Drone)
model = models.resnet50(pretrained=False)
num_classes = 2
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

state_dict = torch.load("models/resnet50_best.pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Preprocess for ResNet
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
classes = ['Bird', 'Drone']

# Streamlit UI
st.title("AeroVision: Bird & Drone Detection")

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.write("**Original image:**")
    st.image(image, use_container_width=True)

    # Run COCO model first to detect people (class 0 in COCO = person)
    coco_results = coco_model(image)
    coco_boxes = []
    try:
        coco_box_objs = coco_results[0].boxes
        # robust access: work whether boxes.xyx y or boxes.xyxy present
        for i, b in enumerate(coco_box_objs):
            # class id and conf
            cls_idx = int(b.cls[0].item()) if hasattr(b, "cls") else int(b.cls)
            conf = float(b.conf[0].item()) if hasattr(b, "conf") else float(b.conf)
            # xy coordinates
            if hasattr(coco_box_objs, "xyxy"):
                xy = coco_box_objs.xyxy[i].tolist()
            elif hasattr(b, "xyxy"):
                xy = b.xyxy[0].tolist()
            else:
                # fallback, try b.xyxy
                xy = [float(x) for x in b.xyxy]
            # capture person boxes only
            if cls_idx == 0:
                coco_boxes.append([int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])])
    except Exception:
        # if something odd happens, just continue with empty person list
        coco_boxes = []

    # Run custom YOLO for bird/drone detection
    custom_results = custom_yolo(image)
    detections = []  # will hold dicts: {xy, yolo_conf, yolo_cls, cls_label, cls_conf, overlap_person}
    try:
        custom_box_objs = custom_results[0].boxes
        for i, b in enumerate(custom_box_objs):
            # xy
            if hasattr(custom_box_objs, "xyxy"):
                xy = custom_box_objs.xyxy[i].tolist()
            elif hasattr(b, "xyxy"):
                xy = b.xyxy[0].tolist()
            else:
                xy = [float(x) for x in b.xyxy]
            x1, y1, x2, y2 = map(int, xy)
            # conf and cls
            yolo_conf = float(custom_box_objs.conf[i].item()) if hasattr(custom_box_objs, "conf") else (float(b.conf[0].item()) if hasattr(b, "conf") else float(b.conf))
            yolo_cls = int(custom_box_objs.cls[i].item()) if hasattr(custom_box_objs, "cls") else (int(b.cls[0].item()) if hasattr(b, "cls") else int(b.cls))

            # Check overlap with any detected person box (IoU > threshold => skip)
            overlap_person = False
            for pbox in coco_boxes:
                if iou([x1, y1, x2, y2], pbox) > 0.3:
                    overlap_person = True
                    break

            cls_label = None
            cls_conf = None

            if overlap_person:
                cls_label = "Person (skipped)"
                cls_conf = 0.0
            else:
                # Crop and classify with ResNet
                cropped = image.crop((x1, y1, x2, y2)).convert("RGB")
                inp = preprocess(cropped).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(inp)
                    probs = torch.softmax(out, dim=1)[0]
                    top_conf, top_idx = torch.max(probs, dim=0)
                    cls_label = classes[int(top_idx)]
                    cls_conf = float(top_conf.item())

            detections.append({
                "xy": [x1, y1, x2, y2],
                "yolo_conf": yolo_conf,
                "yolo_cls": yolo_cls,
                "cls_label": cls_label,
                "cls_conf": cls_conf,
                "overlap_person": overlap_person
            })
    except Exception:
        # no detections or unexpected structure
        detections = []

    # Build annotated image
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=16)
    except Exception:
        font = ImageFont.load_default()

    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det["xy"]
        if det["overlap_person"]:
            label_text = "Person (skipped)"
            box_color = (255, 165, 0)  # orange for skipped
        else:
            cls_conf_pct = det["cls_conf"] * 100 if det["cls_conf"] is not None else 0.0
            label_text = f"{det['cls_label']} ({cls_conf_pct:.0f}%)"
            box_color = (0, 255, 0)  # green for positive detection

        # Draw bounding box and label
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
        draw_label(draw, (x1, y1, x2, y2), label_text, font, rect_color=box_color)

        # Small YOLO confidence below the box - using textbbox instead of textsize
        yolo_conf_pct = det["yolo_conf"] * 100
        small_text = f"yolo:{yolo_conf_pct:.0f}%"
        small_text_x = x1
        small_text_y = y2 + 4
        # compute bbox for small text
        try:
            small_bbox = draw.textbbox((small_text_x, small_text_y), small_text, font=font)
        except Exception:
            small_bbox = draw.textbbox((0, 0), small_text, font=font)
        small_w = small_bbox[2] - small_bbox[0]
        small_h = small_bbox[3] - small_bbox[1]
        draw.rectangle([small_text_x, small_text_y, small_text_x + small_w + 6, small_text_y + small_h + 6], fill=(0, 0, 0))
        draw.text((small_text_x + 3, small_text_y + 3), small_text, fill=(255, 255, 255), font=font)

    # Show final annotated image
    st.write("### Final annotated image (detection + classification):")
    st.image(annotated, use_container_width=True)

    # Also show a simple detections table / summary
    st.write("### Detections summary:")
    if len(detections) == 0:
        st.write("No bird/drone detections found.")
    else:
        for i, det in enumerate(detections):
            if det["overlap_person"]:
                st.write(f"• Object {i+1}: Person overlap detected — skipped classification (YOLO conf: {det['yolo_conf']:.2f})")
            else:
                st.write(f"• Object {i+1}: {det['cls_label']} — classifier conf: {det['cls_conf']:.2f}, YOLO conf: {det['yolo_conf']:.2f}")

    # Optional: provide download link for annotated image
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    buf.seek(0)
    st.download_button("Download annotated image", data=buf, file_name="annotated.png", mime="image/png")
