import cv2
from ultralytics import YOLO

def main():
    model = YOLO(r"C:\Users\Lucas Dev\Downloads\neuralens_generalized\runs\detect\train4\weights\last.pt")
    model.to("cuda:0")  # Move model to GPU for faster inference
    url = "https://192.168.18.4:8080/video"  # URL of the IP camera stream
    cap = cv2.VideoCapture(url)  # Open the video stream
    
    if not cap.isOpened():
        print ("camera error")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("cant grab frame")
            break
        
        results = model(frame)
        annotated_frame = results[0].plot()

        cv2.imshow("Object Detection Live", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()