import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import tempfile
import torch
from torchvision import transforms, models
from ultralytics import YOLO
import pandas as pd

# -------------------------
# PAGE LAYOUT & HEADER
# -------------------------
st.set_page_config(page_title="AeroVision – Bird & Drone Detection", layout="wide")

st.markdown("""
<style>
.main-title { font-size:50px; font-weight:900; text-align:center; margin-bottom:-10px;}
.subtitle { text-align:center; font-size:18px; color:#555; margin-bottom:30px;}
.section-title { font-size:26px; font-weight:700; margin-top:20px; color:#333;}
.feature-box { background-color:#f3f6ff; padding:20px; border-radius:12px; margin-bottom:25px; border-left:4px solid #5a8dee;}
.top-buttons { display:flex; justify-content:center; gap:20px; margin-bottom:20px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🦅 AeroVision – Bird & Drone Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-powered real-time system for detecting Birds, Drones & Humans</p>", unsafe_allow_html=True)

# -------------------------
# PROJECT OVERVIEW & FEATURES
# -------------------------
st.markdown("<h2 class='section-title'>📘 Project Overview</h2>", unsafe_allow_html=True)
st.markdown("""
<div class='feature-box'>
AeroVision detects Birds, Drones, and Persons using:<br><br>
✔ <b>YOLOv8 Object Detection</b><br>
✔ <b>ResNet-50 Classifier (optional)</b><br>
✔ Image & Video Upload Detection<br>
✔ Bounding Boxes with Confidence<br>
✔ Summary Table & Download Options<br>
Ideal for <b>security, wildlife monitoring, hazard management, and airspace safety</b>.
</div>
""", unsafe_allow_html=True)

st.markdown("<h2 class='section-title'>🚀 Key Features</h2>", unsafe_allow_html=True)
st.markdown("""
<div class='feature-box'>
🔹 Image & Video Upload Detection<br>
🔹 Advanced YOLOv8 Model<br>
🔹 ResNet-50 Classifier (optional)<br>
🔹 Smart Person Filtering<br>
🔹 Clean UI & Download Options<br>
🔹 Summary Table with Confidences<br>
🔹 No Coding Required
</div>
""", unsafe_allow_html=True)

# -------------------------
# INPUT CONTROLS
# -------------------------
st.markdown('<div class="top-buttons">', unsafe_allow_html=True)

input_type = st.radio("Input Type:", ["Upload Image", "Upload Video"], index=0, horizontal=True)
use_classifier = st.checkbox("Use ResNet Classifier", value=True)
yolo_thresh = st.slider("YOLO Confidence Threshold", 0.0, 1.0, 0.3, 0.05)

uploaded_file = None
if input_type == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])
elif input_type == "Upload Video":
    uploaded_file = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def xyxy_to_int(xyxy): return [int(x) for x in xyxy]

def iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB-xA) * max(0, yB-yA)
    if interArea == 0: return 0.0
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def draw_label(draw, box, text, font, rect_color=(0,255,0)):
    x1,y1,x2,y2 = box
    try: text_bbox = draw.textbbox((x1,y1), text, font=font)
    except: text_bbox = draw.textbbox((0,0), text, font=font)
    tw,th = text_bbox[2]-text_bbox[0], text_bbox[3]-text_bbox[1]
    padding=4
    bx1,by1 = x1,y1-th-padding*2
    bx2,by2 = x1+tw+padding*2,y1
    draw.rectangle([bx1,by1,bx2,by2], fill=rect_color)
    draw.text((bx1+padding,by1+padding), text, fill=(255,255,255), font=font)

def create_summary_table(detections):
    data=[]
    for i, det in enumerate(detections):
        if det["overlap_person"]:
            data.append([i+1,"Person (skipped)",0.0,det["yolo_conf"]])
        else:
            data.append([i+1,det["cls_label"],det["cls_conf"],det["yolo_conf"]])
    return pd.DataFrame(data, columns=["Object #","Class","Classifier Conf","YOLO Conf"])

def detect_and_annotate(img_pil):
    # YOLO detection
    coco_results = coco_model(img_pil)
    person_boxes=[]
    try:
        for b in coco_results[0].boxes:
            cls_idx=int(b.cls[0])
            conf=float(b.conf[0])
            if cls_idx==0 and conf>=yolo_thresh:
                person_boxes.append(xyxy_to_int(b.xyxy[0]))
    except: pass

    results=custom_yolo(img_pil)
    detections=[]
    for b in results[0].boxes:
        xy=xyxy_to_int(b.xyxy[0])
        x1,y1,x2,y2=xy
        yolo_conf=float(b.conf[0])
        if yolo_conf<yolo_thresh: continue
        overlap_person=any(iou(xy,p)>0.3 for p in person_boxes)
        cls_label="Person (skipped)" if overlap_person else results[0].names[int(b.cls[0])]
        cls_conf=0.0 if overlap_person else float(b.conf[0])
        if use_classifier and not overlap_person:
            cropped=img_pil.crop((x1,y1,x2,y2))
            inp=preprocess(cropped).unsqueeze(0).to(device)
            with torch.no_grad():
                out=classifier_model(inp)
                probs=torch.softmax(out,dim=1)[0]
                cls_conf_val, idx=torch.max(probs,dim=0)
                cls_label = classes[idx]
                cls_conf = float(cls_conf_val.item())
        detections.append({"xy":xy,"yolo_conf":yolo_conf,"cls_label":cls_label,"cls_conf":cls_conf,"overlap_person":overlap_person})

    annotated=img_pil.copy()
    draw=ImageDraw.Draw(annotated)
    font=ImageFont.load_default()
    for det in detections:
        x1,y1,x2,y2=det["xy"]
        label=det["cls_label"]
        conf=det["cls_conf"]
        color=(255,165,0) if det["overlap_person"] else (0,255,0)
        draw.rectangle([x1,y1,x2,y2], outline=color, width=3)
        display=f"{label} ({conf*100:.0f}%)" if conf is not None else label
        draw_label(draw,[x1,y1,x2,y2],display,font,color)
    return annotated, create_summary_table(detections)

# -------------------------
# LOAD MODELS
# -------------------------
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coco = YOLO("yolov8n.pt")
    custom = YOLO("yolov8/yolov8_best.pt")
    classifier = models.resnet50(pretrained=False)
    classifier.fc = torch.nn.Linear(classifier.fc.in_features, 2)
    state = torch.load("models/resnet50_best.pth", map_location=device)
    classifier.load_state_dict(state)
    classifier.to(device).eval()
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    classes = ["Bird","Drone"]
    return device, coco, custom, classifier, preprocess, classes

device, coco_model, custom_yolo, classifier_model, preprocess, classes = load_models()

# -------------------------
# IMAGE DETECTION
# -------------------------
if input_type == "Upload Image" and uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    annotated_img, summary_df = detect_and_annotate(img)
    st.image(annotated_img, caption="Annotated Image", use_container_width=True)
    buf=io.BytesIO()
    annotated_img.save(buf, format="PNG"); buf.seek(0)
    st.download_button("Download Annotated Image", buf, "annotated.png")
    st.write("### 📊 Detection Summary")
    st.dataframe(summary_df)

# -------------------------
# VIDEO DETECTION
# -------------------------
if input_type == "Upload Video" and uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    frames = []
    all_detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        annotated_frame, summary_df = detect_and_annotate(img_pil)
        frames.append(np.array(annotated_frame))
        all_detections.extend(summary_df.to_dict('records'))
    cap.release()

    # Save annotated video
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(temp_video.name, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()

    st.success("🎉 Video processed successfully!")
    st.video(temp_video.name)
    st.write("### 📊 Detection Summary")
    st.dataframe(pd.DataFrame(all_detections))
