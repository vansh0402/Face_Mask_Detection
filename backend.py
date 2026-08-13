import cv2
from keras.models import load_model
from keras.utils import load_img,img_to_array
import numpy as np

# Load face detection model
facemodel = cv2.CascadeClassifier("face.xml")

# Load face detection model
maskmodel = load_model('Mask.h5')

#Face Mask detection Function
def detect_mask(frame): 
    # Detect Faces
    faces = facemodel.detectMultiScale(frame)
    results = []
    for (x, y, l, w) in faces:
        #Crop Detected Faces
        crop_face = frame[y:y+w, x:x+l]
        #Save temporary face image
        cv2.imwrite('temp.jpg', crop_face)
        # Resize image for model
        crop_face = load_img('temp.jpg',target_size=(150,150))
        # Convert image to array
        crop_face = img_to_array(crop_face)
        # Add batch dimension
        crop_face = np.expand_dims(crop_face,axis=0)
        #Prediction
        pred = maskmodel.predict(crop_face,verbose = 0)[0][0]
            # Draw Rectangle
        if pred >=0.5:
            # NO MASK
            label = "NO MASK"
            confidence = pred * 100
            color = (0,0,255)           
        else:
            # MASK
            label = "MASK"
            confidence = (1-pred) * 100
            color = (0,255,0) 

        cv2.rectangle(frame,(x, y),(x+l, y+w),color,4)
        cv2.putText(frame,f"{label} {confidence:.1f}%",(x, y - 10),cv2.FONT_HERSHEY_SIMPLEX,0.7, color,2)
        results.append({
            "label": label,
            "confidence": confidence
        })

    return frame, results




