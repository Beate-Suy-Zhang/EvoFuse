from ultralytics import YOLO

if __name__ == '__main__':    

    model = YOLO("./EvoFusionRuns/vvtevo/M3FD/iter0/weights/best.pt")
    
    metrics = model.val(batch=1, split="val", plots=False, save=False, device=1, mode="val")
    print(metrics.box.map)
    metrics = model.val(batch=1, split="test", plots=False, save=False, device=1, mode="val")
    print(metrics.box.map)
    metrics = model.val(batch=1, split="train", plots=False, save=False, device=1, mode="val")
    print(metrics.box.map)
    # metrics = model.val(batch=1, split="train", plots=False, save=False, device=1)
    # print(metrics.box.map)