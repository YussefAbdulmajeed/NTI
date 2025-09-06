import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# -------------------------------
# Load Model (replace with your path)
# -------------------------------
@st.cache_resource
def load_model():
    model = torch.load("mask_classifier.pth", map_location="cpu")
    model.eval()
    return model

model = load_model()

# -------------------------------
# Image Preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # standard ImageNet normalization
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("😷 Mask Classification App")
st.write("Upload an image to check if the person is **Wearing a Mask** or **Not Wearing a Mask**.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0)  # add batch dimension

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    # Class mapping (0 -> No Mask, 1 -> Mask)
    classes = ["No Mask 🚫", "Mask ✅"]
    st.subheader(f"Prediction: {classes[predicted.item()]}")
