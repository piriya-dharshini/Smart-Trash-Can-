import cv2
from roboflow import Roboflow
import numpy as np
rf = Roboflow(api_key="MdQYF8GWPymGlvHOD1MB")
project = rf.workspace().project("garbage-classification-3")
model = project.version(2).model
cap=cv2.VideoCapture(0)

while True:
            _, img=cap.read()
            predns=model.predict(img, confidence=40, overlap=30).json()
            predict=predns['predictions']

            #biodegradable,cloth,glass,metal,plastic,cardboard,paper
            boxes=[]
            confidences=[]
            try:
                predictions=predict[0]
                center_x=predictions['x']
                center_y=predictions['y']
                w=predictions['width']
                h=predictions['height']
                confdn=predictions['confidence']
                cls=predictions['class']

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)



                if confdn>0.5:
                    boxes.append([x,y,w,h])
                    confidences.append(float(confdn))

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                font = cv2.FONT_HERSHEY_PLAIN
                colors = np.random.uniform(0, 255, size=(len(boxes), 3))

                if len(boxes) > 0:
                    for i in indexes.flatten():
                        x, y, w, h = boxes[i]
                        confidence = str(round(confidences[i], 2))
                        color = colors[i]
                        cv2.rectangle(img, (int(x),int(y)),(int(x + w),int(y + h)), color, 2)
                        cv2.putText(img,cls + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)
                cv2.imshow('Image', img)
            except IndexError:
                pass
                cv2.imshow('Image',img)
            key=cv2.waitKey(1)
            if key==27:
                break

cap.release()
cv2.destroyAllWindows()

# visualize your prediction
# model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())