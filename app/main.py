from PIL import Image
from io import BytesIO
from ultralytics import YOLO
from ultralytics.engine.results import Results
import uvicorn
from typing import Annotated
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

app = FastAPI()
# base_model = YOLO('yolov8s-oiv7.pt')  # Load model at startup

# base_model.export(format="edgetpu", imgsz=320, )  # creates 'yolov8n_full_integer_quant_edgetpu.tflite'

# Load the exported TFLite Edge TPU model
model = YOLO("/models/yolov8s-oiv7_full_integer_quant_edgetpu.tflite", task="detect")

@app.get("/")
async def root():
    return {"message": "Hello World"}

async def do_detection(file: UploadFile) -> Results:
    # Process the uploaded image for object detection
    image_bytes = await file.read()
    image_stream = BytesIO(image_bytes)
    image = Image.open(image_stream)
    # image = np.frombuffer(image_bytes, dtype=np.uint8)
    # image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Perform object detection with YOLOv8
    detections = model(image, imgsz=320)
    #detections = model.predict(image)
    return detections[0] #TODO Check to make sure a detection exists

@app.post("/detect")
async def detect(file: UploadFile):
    detection = do_detection(file)

    # Extract bounding box data
    boxes = detection.boxes.xyxy.cpu().numpy()
    scores = detection.boxes.conf.cpu().numpy()
    classes = detection.boxes.cls.cpu().numpy()

    # Format the results as a list of dictionaries
    results = []
    for box, score, cls in zip(boxes, scores, classes):
        results.append({
            'x1': float(box[0]),
            'y1': float(box[1]),
            'x2': float(box[2]),
            'y2': float(box[3]),
            'confidence': float(score),
            'class': int(cls)
        })

    return {'detections': results}

@app.post("/detect_img")
async def detect_img(file: UploadFile):
    detection = await do_detection(file)

    detection.show()
    detection.save(filename="result.jpg")
    
    return FileResponse("result.jpg")

class CPAIResponse:
    def __init__(self):
        self.success= True
        self.message = ""
        self.error = ""
        self.predictions = []
        self.count = 0
        self.inferenceMs = 0
        self.processMs = 0
        self.moduleId = ""
        self.moduleName = ""
        self.command = ""
        self.executionProvider = ""
        self.canUseGPU = True
        self.analysisRoundTripMs = 0
    

@app.post("/detect_cpai")
async def detect_cpai(image: UploadFile):
    detection = await do_detection(image)

    response = CPAIResponse()

    # Extract bounding box data
    boxes = detection.boxes.xyxy.cpu().numpy()
    scores = detection.boxes.conf.cpu().numpy()
    classes = detection.boxes.cls.cpu().numpy()

    # Format the results as a list of dictionaries
    for box, score, cls in zip(boxes, scores, classes):
        response.predictions.append({
            'x_min': float(box[0]),
            'y_min': float(box[1]),
            'x_max': float(box[2]),
            'y_max': float(box[3]),
            'confidence': float(score),
            'label': model.names[int(cls)].lower()
        })
    
    #TODO Fill out other fields

    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)