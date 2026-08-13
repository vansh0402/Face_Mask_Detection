import streamlit as st
import numpy as np
import cv2
import tempfile

from backend import detect_mask

st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide"
)

st.title("😷 Face Mask Detection")

st.write("AI-powered face mask detection using Computer Vision and Deep Learning.")

st.divider()

mode = st.radio(
    "Select Detection Mode",
    ["📷 Image", "🎥 Video", "📹 Webcam"],
    horizontal=True
)
if mode == "📷 Image":
    uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )
        frame = cv2.imdecode(file_bytes,cv2.IMREAD_COLOR)
        if frame is None:
            st.error("Unable to read the image.")
        else:
            with st.spinner("Detecting faces..."):
                result_frame, results = detect_mask(frame)

            st.subheader("Detection Result")

            result_image = cv2.cvtColor(result_frame,cv2.COLOR_BGR2RGB)
            st.image(result_image,use_container_width=True)
            st.divider()

            st.subheader("Detection Summary")

            total_faces = len(results)

            mask_count = sum(
                1
                for result in results
                if result["label"] == "MASK"
            )

            no_mask_count = sum(
                1
                for result in results
                if result["label"] == "NO MASK"
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric( "👤 Faces",total_faces)
            with col2:
                st.metric("😷 Mask",mask_count)
            with col3:
                st.metric("🚫 No Mask",no_mask_count)
            st.divider()

            st.subheader("Predictions")
            if total_faces == 0:
                st.warning("No face detected in the image.")
            else:
                for i, result in enumerate(results):
                    label = result["label"]
                    confidence = result["confidence"]
                    if label == "MASK":
                        st.success(f"😷 Face {i + 1}: MASK — "f"{confidence:.2f}%")
                    else:
                        st.error(f"🚫 Face {i + 1}: NO MASK — "f"{confidence:.2f}%")
elif mode == "🎥 Video":
    uploaded_video = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        temp_input = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp_input.write(
            uploaded_video.read()
        )

        temp_input.close()
        video = cv2.VideoCapture(temp_input.name)
        st.subheader("Video Detection")
        video_placeholder = st.empty()
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            result_frame, results = detect_mask(frame)
            result_image = cv2.cvtColor(result_frame,cv2.COLOR_BGR2RGB)
            video_placeholder.image(result_image,channels="RGB")
        video.release()
        st.success("Video processing completed.")
        
elif mode == "📹 Webcam":
    st.subheader("Webcam Detection")
    camera_image = st.camera_input(
        "Take a picture"
    )
    if camera_image is not None:
        file_bytes = np.asarray(
            bytearray(camera_image.read()),
            dtype=np.uint8
        )
        frame = cv2.imdecode(file_bytes,cv2.IMREAD_COLOR)
        if frame is None:
            st.error("Unable to read the camera image.")
        else:
            with st.spinner("Detecting faces..."):
                result_frame, results = detect_mask(frame)
            result_image = cv2.cvtColor(result_frame,cv2.COLOR_BGR2RGB)
            st.subheader("Detection Result")
            st.image(result_image,use_container_width=True)
            st.divider()
            st.subheader("Detection Summary")
            total_faces = len(results)
            mask_count = sum(
                1
                for result in results
                if result["label"] == "MASK"
            )
            no_mask_count = sum(
                1
                for result in results
                if result["label"] == "NO MASK"
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "👤 Faces",
                    total_faces
                )
            with col2:
                st.metric(
                    "😷 Mask",
                    mask_count
                )
            with col3:
                st.metric(
                    "🚫 No Mask",
                    no_mask_count
                )
            st.divider()
            st.subheader("Predictions")
            if total_faces == 0:
                st.warning(
                    "No face detected."
                )
            else:
                for i, result in enumerate(results):
                    label = result["label"]
                    confidence = result["confidence"]
                    if label == "MASK":
                        st.success(
                            f"😷 Face {i + 1}: MASK — "
                            f"{confidence:.2f}%"
                        )
                    else:
                        st.error(
                            f"🚫 Face {i + 1}: NO MASK — "
                            f"{confidence:.2f}%"
                        )