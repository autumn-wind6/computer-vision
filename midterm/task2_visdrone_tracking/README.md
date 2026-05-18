# Task 2: VisDrone Detection and Multi-Object Tracking

This folder is the project directory for task 2. Dataset files live under the root `data/` folder. Training, tracking, and analysis outputs are saved under `runs/task2/`.

Recommended structure:

```text
期中作业/
  data/
    VisDrone/
      VisDrone2019-DET-train/
        images/
        annotations/
      VisDrone2019-DET-val/
        images/
        annotations/
    video.mp4
  task2_visdrone_tracking/
    data.py
    train.py
    track_video.py
    analyze_occlusion.py
```

Generated files will be written here:

```text
data/VisDrone/.../labels/
task2_visdrone_tracking/visdrone_det.yaml
runs/task2/detect/...
runs/task2/outputs/...
```

## Environment

```powershell
pip install ultralytics supervision opencv-python pillow pandas matplotlib tqdm
```

## 1. Prepare VisDrone Labels

Run from the assignment root:

```powershell
python .\task2_visdrone_tracking\data.py
```

Default input:

```text
data/VisDrone
```

Default output:

```text
task2_visdrone_tracking/visdrone_det.yaml
data/VisDrone/VisDrone2019-DET-train/labels
data/VisDrone/VisDrone2019-DET-val/labels
```

If your dataset is elsewhere, pass it explicitly:

```powershell
python .\task2_visdrone_tracking\data.py `
  --visdrone-root "D:\桌面\HW2.2\VisDrone"
```

## 2. Train YOLOv8s

```powershell
python .\task2_visdrone_tracking\train.py
```

Default output:

```text
runs/task2/detect/visdrone_yolov8s_lowvram/weights/best.pt
```

The notebook recorded this result:

- mAP50: 0.334
- mAP50-95: 0.186
- best class: car, mAP50 about 0.756

## 3. Track Video and Count Line Crossings

Put a 10-30 second test video at:

```text
data/video.mp4
```

Then run:

```powershell
python .\task2_visdrone_tracking\track_video.py
```

Default outputs:

```text
runs/task2/outputs/result_tracking_custom.mp4
runs/task2/outputs/tracking_log.csv
```

If your video has another name, pass it explicitly:

```powershell
python .\task2_visdrone_tracking\track_video.py `
  --video .\data\f83f733151781d172e7ecb34aec0f132.mp4
```

## 4. Occlusion and ID-Switch Analysis

```powershell
python .\task2_visdrone_tracking\analyze_occlusion.py --right-half-only
```

Default outputs:

```text
runs/task2/outputs/silver_suv_analysis/frame_*.jpg
runs/task2/outputs/silver_suv_analysis/target_history.csv
runs/task2/outputs/silver_suv_analysis/id_switches.txt
```

The notebook observation was the 6-8 second segment where a silver SUV is partially blocked by a tree trunk. With the continuity-based target selection rule, the recorded ID switches include:

- Frame 232: ID 7 -> 16
- Frame 261: ID 16 -> 11

The likely reason is that the tree occlusion causes temporary target loss and weak appearance cues, so ByteTrack reconnects the detection as a new identity.
