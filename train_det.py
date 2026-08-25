from ultralytics import YOLO
from ultralytics import RTDETR
import time


def train_estad():
    model = YOLO("yolov8s.yaml", task='detect') 
    # model = YOLO("./EvoFusionRuns/exp1/M3FD/iter0/weights/best.pt", task='detect')  
    model.train(data="M3FD.yaml", batch=32, epochs=300, imgsz=512, close_mosaic=0, patience=50, workers=4, device=1,
                project="./EvoFusionRuns/M3FD/", name="MetaFusion")  # plfuse4
    # model.train(data="MSOD.yaml", batch=32, epochs=300, imgsz=512, close_mosaic=0, patience=50, workers=8, device=1,
    #             project="./EvoFusionRuns/MSOD/", name="train0b64")  # plfuse4
    # model.train(data="LLVIP.yaml", batch=32, epochs=300, imgsz=512, close_mosaic=0, patience=50, workers=8, device=1,
    #             project="./EvoFusionRuns/LLVIP/", name="train0")  # plfuse4
    # model.train(resume=True, patience=300)
    metrics = model.val(batch=1, split="val", plots=False, save=False, device=1, mode="val", name="val0")
    print(metrics.box.map)
    metrics = model.val(batch=1, split="train", plots=False, save=False, device=1, mode="val", name="val00")
    print(metrics.box.map)




if __name__ == '__main__':
    #time.sleep(60*180)
    train_estad()

