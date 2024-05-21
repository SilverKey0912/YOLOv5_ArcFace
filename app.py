import streamlit as st
import argparse
import torch
from detect_module import detect  # Assuming your detect function is in detect_module.py
from PIL import Image
import os

def main():
    # Define default values for arguments
    default_yoloV5_weights = 'weights/yolo5_weights/best.pt'
    default_Areface_weights = 'weights/arcface_weights/resnet18_110.pth'
    default_silentFace_weights = "weights/anti_spoof_models/"
    default_source = '0'
    default_output = 'output'
    default_img_size = 640
    default_fourcc = 'mp4v'
    default_save_folder = r"C:\Users\lymin\Documents\Semester 8\yoloV5-arcface_forlearn-master\pic"

    st.title("Real-time Face Detection and Recognition")

    # Define the input options for Streamlit
    source = st.radio("Select Source", ("Webcam", "Video File"))

    if source == "Webcam":
        source = "0"    
    else:
        source = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    # button to upload image
    st.sidebar.title("Parameter setting")
    uploaded_file = st.sidebar.file_uploader("Choose image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # input name of people in image
        file_name = st.sidebar.text_input("Input name")

        if st.sidebar.button("Save"):
            # check if name is none
            if file_name.strip() == "":
                st.sidebar.warning("Please input name.")
            else:
                # get file extension
                file_extension = os.path.splitext(uploaded_file.name)[-1]
                # if user do not input file extension, default add file extension
                if not file_name.endswith(file_extension):
                    file_name += file_extension

                # Check for duplicate names
                save_path = os.path.join(default_save_folder, file_name)
                if os.path.exists(save_path):
                    st.sidebar.error("This person is already exist. Please input a different name.")
                else:
                    # Save image
                    image = Image.open(uploaded_file)
                    save_path = os.path.join(default_save_folder, file_name)
                    image.save(save_path)
                    st.sidebar.success("Save Successfully!")

    # Set up other configuration options
    yoloV5_weights = default_yoloV5_weights
    Areface_weights = default_Areface_weights
    silentFace_weights = default_silentFace_weights
    output = default_output
    img_size = default_img_size
    conf_thres = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4)
    iou_thres = st.sidebar.slider("IOU Threshold", 0.1, 1.0, 0.5)
    device = st.sidebar.selectbox("Device", ("cpu", "cuda"))
    # fourcc = st.sidebar.text_input("Output Video Codec", "mp4v")

    if st.button("Start Detection"):
        opt = argparse.Namespace(
            yoloV5_weights=yoloV5_weights,
            Areface_weights=Areface_weights,
            silentFace_weights=silentFace_weights,
            source=source,
            output=output,
            img_size=img_size,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            fourcc=default_fourcc,
            device=device,
            view_img=True,
            save_txt=False,
            classes=None,
            agnostic_nms=False,
            augment=False,
            open_rf=1,
        )

        with torch.no_grad():
            detect(opt)

if __name__ == "__main__":
    main()
