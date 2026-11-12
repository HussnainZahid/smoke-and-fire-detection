# Smoke and Fire Detection

## Real-Time Fire & Smoke Detection Using Computer Vision and Deep Learning

A software-based desktop surveillance application that analyzes video footage and looks for visible **fire** and **smoke**. The application uses a custom fine-tuned **YOLO26n** object-detection model, OpenCV video processing, and an optional computer-vision flame-verification step.

When the application starts, it loads the trained model from `models/best.pt`, scans `data/input/` for supported video files, and opens a desktop dashboard. The user can select a video, start or pause analysis, adjust separate Fire and Smoke confidence thresholds, enable or mute the alert sound, and review the annotated output saved in `data/output/`.

> **Safety notice:** This is an academic/research prototype for visual monitoring and early-warning support. It is not a certified fire alarm or a replacement for approved smoke detectors, fire alarms, sprinklers, emergency procedures, or industrial safety controls. Do not rely on this software as the only means of fire protection.

---

## Table of Contents

1. [Project at a Glance](#1-project-at-a-glance)
2. [Main Features](#2-main-features)
3. [How the System Works](#3-how-the-system-works)
4. [Detection and Hybrid Flame Verification](#4-detection-and-hybrid-flame-verification)
5. [Technologies Used](#5-technologies-used)
6. [Dataset Information](#6-dataset-information)
7. [Model Training: Transfer Learning](#7-model-training-transfer-learning)
8. [Reported Training and Evaluation Results](#8-reported-training-and-evaluation-results)
9. [Project Folder Structure](#9-project-folder-structure)
10. [System Requirements](#10-system-requirements)
11. [Installation on Windows](#11-installation-on-windows)
12. [How to Run the Application](#12-how-to-run-the-application)
13. [How to Use the Dashboard](#13-how-to-use-the-dashboard)
14. [Detection Thresholds and Alerts](#14-detection-thresholds-and-alerts)
15. [Output Videos](#15-output-videos)
16. [Troubleshooting](#16-troubleshooting)
17. [Limitations and Future Improvements](#17-limitations-and-future-improvements)
18. [Dataset License and Acknowledgement](#18-dataset-license-and-acknowledgement)
19. [Useful References](#19-useful-references)

---

## 1. Project at a Glance

| Item | Details |
|---|---|
| Project name | Smoke and Fire Detection |
| Application type | Software-based desktop application |
| Main task | Fire and smoke object detection in video |
| Deep-learning model | YOLO26n |
| Training method | Transfer Learning / Fine-Tuning |
| Main computer-vision library | OpenCV |
| Desktop interface | Tkinter |
| Model framework | PyTorch and Ultralytics |
| Detection classes | `fire` and `smoke` |
| Model file | `models/best.pt` |
| Input video folder | `data/input/` |
| Annotated output folder | `data/output/` |

### Project Objectives

- Detect visible fire and smoke in recorded video footage.
- Read video frames and analyze them using a trained deep-learning model.
- Draw a bounding box around each accepted detection.
- Display the class name and confidence percentage on each box.
- Use separate confidence controls for Fire and Smoke.
- Display an alert banner and play a warning beep when a qualifying detection occurs.
- Save the analyzed video with the detection boxes drawn on it.
- Provide a simple desktop interface that does not require the user to run separate Python commands for each video.

The project is designed for experimentation and demonstration. Its performance must be checked on representative, unseen footage before any operational use is considered.

---

## 2. Main Features

### 2.1 Automatic Video Loading

When the application starts, it checks the `data/input/` directory for supported video files. A video can be selected from the dashboard. The user can also use **Browse Video** to choose a video from another location.

Supported extensions in the project specification:

- `.mp4`
- `.avi`
- `.mov`
- `.mkv`
- `.wmv`

The actual formats that can be read depend on the OpenCV video backend and the codecs installed on the computer.

### 2.2 Fire Detection

The YOLO26n model predicts visible flame regions. Accepted Fire detections are drawn with a **red bounding box** and a label similar to:

```text
FIRE 95.5%
```

### 2.3 Smoke Detection

The YOLO26n model predicts visible smoke regions. Accepted Smoke detections are drawn with a **cyan/blue bounding box** and a label similar to:

```text
SMOKE 92.4%
```

The percentages above are examples of how a label may look. Actual confidence scores depend on the input video and model prediction.

### 2.4 Playback Controls

The dashboard provides controls to:

- Start detection.
- Pause and resume processing.
- Stop the current analysis.
- Select another video.
- Browse for a video file.

### 2.5 Visual and Sound Alerts

When a Fire or Smoke prediction passes its configured class threshold, the dashboard can display a warning and play an audible beep. The user can enable or disable the beep from the interface.

### 2.6 Annotated Video Saving

The application writes processed frames to an output video. The intended output naming pattern is:

```text
data/output/annotated_<video-name>.mp4
```

The output video contains the original footage with the Fire and Smoke labels and bounding boxes overlaid.

---

## 3. How the System Works

The application follows this processing pipeline:

```text
Input video from data/input/ or Browse Video
                    |
                    v
             OpenCV Video Reader
                    |
                    v
          Read video frame by frame
                    |
                    v
          YOLO26n Fire/Smoke Model
                    |
                    v
       Fire and Smoke predictions
       (class, confidence, coordinates)
                    |
                    v
      Optional flame-colour verification
       (RGB + HSV + YCrCb + hot-core check)
                    |
                    v
         Apply class confidence limits
                    |
                    v
       Draw Fire and Smoke bounding boxes
                    |
             +------+------+
             |             |
             v             v
       Alert banner     Beep sound
             |             |
             +------+------+
                    |
                    v
       Write annotated output video
                    |
                    v
             data/output/
```

### Step-by-step explanation

1. **Load the model:** The application looks for `models/best.pt` and loads the custom-trained YOLO26n model.
2. **Check the classes:** The model should expose the two expected class names: `fire` and `smoke`.
3. **Open the video:** OpenCV opens the selected video and reads its frames.
4. **Run detection:** Each processed frame is passed to the YOLO model.
5. **Filter predictions:** Fire and Smoke predictions are checked against their separate confidence thresholds.
6. **Run flame verification:** If enabled, the hybrid computer-vision step checks for strong flame-like colour and intensity patterns that may be difficult for the detector to localize clearly.
7. **Annotate the frame:** The application draws the accepted bounding boxes, class labels, and confidence percentages.
8. **Show alerts:** The application displays a warning and may play the beep sound, depending on the detection and sound settings.
9. **Save the output:** The annotated frames are written to a video in `data/output/`.

---

## 4. Detection and Hybrid Flame Verification

### 4.1 YOLO26n Object Detection

YOLO26n is the main detection model. It predicts bounding boxes, class IDs, and confidence scores for Fire and Smoke.

The class mapping used by this project is:

| Class ID | Class name | Bounding-box colour |
|---:|---|---|
| `0` | `fire` | Red |
| `1` | `smoke` | Cyan / Blue |

The model's bounding-box coordinates are used to draw rectangles around predicted regions. The application should not create fixed or decorative boxes that are unrelated to model predictions.

### 4.2 Hybrid Flame Verification

The project description includes a supplementary colour-based flame-verification step. It is intended to help identify a possible visible flame region when a flame is partly obscured by a dense smoke plume or when the neural network's Fire prediction is weak.

The described verifier examines several colour representations:

- **RGB:** Looks at red, green, and blue pixel values.
- **HSV:** Helps describe colour and brightness separately.
- **YCrCb:** Separates luminance from colour information.
- **Hot-core intensity check:** Looks for a sufficiently bright central region that may be consistent with a visible flame.

This is a heuristic computer-vision aid, not a second trained fire detector. Smoke, orange lighting, reflections, welding, and other bright objects can resemble flame colours. Therefore, colour verification can create false positives and must be tested against negative examples. It cannot guarantee that a fire hidden in smoke will always be found.

The application should keep neural-network predictions and colour-verifier candidates distinguishable in its internal results and, where possible, in the user interface. A colour-verifier candidate should not be presented as a YOLO model prediction unless the model itself made that prediction.

### 4.3 Confidence Scores

A confidence score is the model's estimated confidence for a particular prediction. It is not a guaranteed probability that the prediction is correct in the real world.

The project uses separate Fire and Smoke thresholds so the user can tune them independently. The documented default settings are:

| Class | Default threshold |
|---|---:|
| Fire | 35% |
| Smoke | 45% |

These values are application operating settings, not model accuracy figures. They may need adjustment after evaluation on representative footage.

---

## 5. Technologies Used

| Technology | Purpose in this project |
|---|---|
| **Python 3.10+** | Main programming language. |
| **PyTorch** | Runs the neural network and performs model inference. |
| **Ultralytics YOLO26n** | Detects Fire and Smoke and returns bounding boxes and confidence scores. |
| **OpenCV (`opencv-python`)** | Reads video frames, processes images, converts colour spaces, draws boxes, and writes annotated video. |
| **Tkinter** | Builds the desktop dashboard and playback controls. Tkinter is included with many standard Python installations; on some systems it may need to be installed separately. |
| **Pillow (PIL)** | Converts image frames for display in the Tkinter interface. |
| **NumPy** | Supports efficient image arrays and colour/intensity calculations. |

### Why these technologies?

- Python has a large ecosystem for AI and computer vision.
- YOLO provides object-detection predictions in a format that can be drawn on video frames.
- OpenCV provides the tools needed to read, process, and save videos.
- Tkinter provides a desktop interface without requiring a web browser or a separate web server.
- PyTorch runs the trained model, while NumPy supports image calculations.

---

## 6. Dataset Information

### 6.1 Dataset Source

**Dataset name:** CCTV Smoke & Fire Emergency Detection Dataset  
**Source:** Hugging Face  
**Repository:** <https://huggingface.co/datasets/Simuletic/CCTV-Smoke-Fire-Emergency-Detection-Dataset>

The dataset is a small collection of synthetic CCTV-style images intended for Fire and Smoke detection experiments.

### 6.2 Dataset Summary

| Item | Details |
|---|---:|
| Images in the raw collection | 240 |
| Images with label files | 239 |
| Images without a label file | 1 |
| Image resolution stated for the collection | 1024 × 1024 |
| Detection classes | 2: Fire and Smoke |
| Fire bounding boxes reported | 121 |
| Smoke bounding boxes reported | 141 |
| Annotation format | YOLO text format |

The raw collection includes one image without a matching annotation file:

```text
fire_detected_var16_img3.png
```

That image was quarantined/excluded during the described preparation process. It was not treated as a negative/background image because no verified annotation was available for it.

### 6.3 Dataset Split

The described experiment split the 239 valid image-label pairs into three sets:

| Split | Images | Approximate share | Purpose |
|---|---:|---:|---|
| Training | 179 | 75% | Used to update model weights. |
| Validation | 35 | 15% | Used to monitor training and select a checkpoint. |
| Test | 25 | 10% | Held out for final evaluation. |
| **Total** | **239** | **100%** | |

The split was designed to reduce data leakage. Where multiple images came from the same scene or variation group, those related images should be kept in the same split. This helps prevent the model from being tested on images that are almost identical to its training examples.

### 6.4 YOLO Annotation Format

Each YOLO annotation text file contains one row per object:

```text
class_id x_center y_center width height
```

The four box values are normalized between `0` and `1` relative to the image width and height.

For this project:

- `0` means Fire.
- `1` means Smoke.

An image with no target object may have an empty annotation file. An image with no label file is not automatically a negative image; it must be checked before use.

### 6.5 Dataset Limitations

The dataset is small and synthetic. It may not fully represent real industrial CCTV conditions, including different camera angles, lighting, distance, weather, compression, smoke density, steam, dust, exhaust, reflections, and other difficult scenes.

For a stronger model, add properly labelled real-world images and hard-negative examples such as steam, dust, fog, clouds, welding, and reflections. Keep entire videos, camera views, or related scenes separate between training, validation, and test sets to reduce data leakage.

---

## 7. Model Training: Transfer Learning

### 7.1 Training Approach

The model was developed using **Transfer Learning**, also called **Fine-Tuning**. The network was not trained from random initialization.

The process was:

1. Load the pretrained YOLO26n checkpoint (`yolo26n.pt`).
2. Adapt the detection head to the two project classes: `fire` and `smoke`.
3. Prepare the custom dataset in YOLO format.
4. Fine-tune the pretrained model on the training images.
5. Monitor performance on the validation images.
6. Save the best-performing checkpoint as `best.pt`.
7. Evaluate the selected model on the separate held-out test images.

In simple terms, the pretrained model already contains useful general visual features. Fine-tuning adjusts the model to recognize the Fire and Smoke classes in the project's dataset.

### 7.2 Training Environment

| Setting | Reported value |
|---|---|
| Platform | Google Colab |
| GPU | NVIDIA Tesla T4 |
| Framework | PyTorch |
| Model | YOLO26n pretrained checkpoint |
| Model parameters in the reported model summary | 2,504,580 |
| Number of classes | 2 |

### 7.3 Training Hyperparameters

| Parameter | Reported value |
|---|---:|
| Epochs | 30 |
| Image size | 640 × 640 |
| Batch size | 8 |
| Optimizer | AdamW |
| Learning rate (`lr`) | 0.001667 |
| Momentum | 0.9 |
| Early-stopping patience | 10 epochs |
| Pretrained weights | Enabled |

The best checkpoint was exported as:

```text
models/best.pt
```

The reported file size was approximately 5.20 MB.

### 7.4 Example Fine-Tuning Code

The following is an illustrative example of the described training approach. The exact dataset YAML path and split must match the training notebook.

```python
from ultralytics import YOLO

# Load pretrained YOLO26n weights.
model = YOLO("yolo26n.pt")

# Fine-tune for Fire and Smoke detection.
results = model.train(
    data="fire_smoke.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    pretrained=True,
    patience=10,
    device=0
)

# Ultralytics saves the best checkpoint in the run folder:
# runs/detect/train/weights/best.pt
```

The checkpoint used by the desktop application should be copied to `models/best.pt`.

---

## 8. Reported Training and Evaluation Results

The following figures are the results supplied for this project. They should be treated as reported experimental results and not as a guarantee of performance on real industrial CCTV footage.

### 8.1 Validation Results

The validation set contained 35 images.

| Metric | Overall | Fire | Smoke |
|---|---:|---:|---:|
| Precision | Not supplied overall in this run summary | 89.0% | 70.0% |
| Recall | Not supplied overall in this run summary | 93.5% | 70.0% |
| mAP50 | 87.4% reported final validation result | 97.9% | 76.9% |
| mAP50–95 | 46.4% | Not supplied | Not supplied |

The training history was also reported to reach an mAP50 peak of 95.0% at Epoch 29. This is a training-history observation and is not the same as the separately reported final validation result of 87.4%.

### 8.2 Held-Out Test Results

The held-out test set contained 25 unseen images.

| Metric | Overall | Fire | Smoke |
|---|---:|---:|---:|
| Precision | 81.3% | 72.0% | 90.6% |
| Recall | 87.0% | 86.7% | 87.4% |
| mAP50 | 87.7% | 85.4% | 89.9% |

### 8.3 Understanding the Metrics

- **Precision:** Of the detections made by the model, how many were correct according to the evaluation labels?
- **Recall:** Of the labelled Fire or Smoke objects, how many did the model find?
- **mAP50:** Mean Average Precision at an Intersection over Union (IoU) threshold of 0.50.
- **mAP50–95:** Mean Average Precision averaged across IoU thresholds from 0.50 to 0.95. It is a stricter measure of detection and box localization quality.

Precision, recall, and mAP are not the same as simple classification accuracy. For object detection, report the metric name and evaluation split rather than describing all scores simply as “accuracy.”

The validation and test sets are small, so the reported metrics can change noticeably when a few images are added or removed. Test results should be recalculated from the exact final `best.pt` checkpoint and the documented held-out test split when preparing a formal report.

---

## 9. Project Folder Structure

The intended project layout is:

```text
Smoke_and_fire_detection/
│
├── app.py                  # Main Tkinter desktop application
├── requirements.txt        # Python package dependencies
├── README.md               # Project documentation
│
├── Notebook/
│   └── ...                 # Google Colab/Jupyter training notebook(s)
│
├── models/
│   └── best.pt             # Custom fine-tuned YOLO26n weights
│
└── data/
    ├── input/              # Input videos
    └── output/             # Annotated output videos
```

### Folder descriptions

| Path | What it contains |
|---|---|
| `app.py` | Starts the desktop dashboard and coordinates video processing, detection, alerts, and output writing. |
| `requirements.txt` | Lists the Python packages needed by the application. |
| `Notebook/` | Training and evaluation notebooks. |
| `models/best.pt` | The trained Fire and Smoke detection model. |
| `data/input/` | Videos that the application can load automatically. |
| `data/output/` | Processed videos written by the application. |

The application may create additional temporary files or logs depending on its implementation. Do not store the original dataset or large temporary files in the output directory unless required.

---

## 10. System Requirements

Recommended starting requirements:

- Windows 10 or Windows 11.
- Python 3.10 or a compatible Python version supported by the installed PyTorch and Ultralytics releases.
- At least 8 GB RAM recommended for basic video processing; larger or high-resolution videos may need more memory.
- A CPU capable of running PyTorch inference.
- An NVIDIA GPU is recommended for faster processing but is not required for basic CPU inference.
- Enough disk space for the model, input video, and annotated output.

The actual processing speed depends on the computer, video resolution, codec, model settings, and whether a compatible GPU is available.

---

## 11. Installation on Windows

### 11.1 Open PowerShell

Move into the project folder. Change the path to match your computer:

```powershell
cd "E:\Smoke_and_fire_detection"
```

### 11.2 Create a Virtual Environment

```powershell
python -m venv .venv
```

### 11.3 Activate the Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation, you can use the following command for the current terminal session, subject to your organization's security policy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 11.4 Install Dependencies

Make sure `requirements.txt` includes the packages used by the actual `app.py`. A typical list for this project is:

```text
ultralytics
opencv-python
numpy
Pillow
```

If the application imports additional third-party packages, add them to `requirements.txt` as well.

Install the packages:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**OpenCV note:** Use one OpenCV distribution in the environment. For a desktop application that uses `cv2.imshow` or other GUI functions, `opencv-python` is generally preferable to `opencv-python-headless`. Do not install both packages together in the same environment unless you have a specific reason and have checked compatibility.

**PyTorch note:** Install a PyTorch build compatible with your operating system and hardware. For NVIDIA GPU acceleration, follow the official PyTorch installation selector for the appropriate CUDA build. CPU inference can be used if no compatible GPU is available.

### 11.5 Add the Trained Model

Copy your trained model file into:

```text
models/best.pt
```

The model must be the custom Fire and Smoke checkpoint, not the original generic `yolo26n.pt` file.

### 11.6 Add an Input Video

Copy one or more supported videos into:

```text
data/input/
```

---

## 12. How to Run the Application

With the virtual environment active, run:

```powershell
python app.py
```

The Tkinter desktop dashboard should open. The application should load `models/best.pt` and scan the `data/input/` folder for supported videos.

### Quick Run on Later Days

```powershell
cd "E:\Smoke_and_fire_detection"
.\.venv\Scripts\Activate.ps1
python app.py
```

If the dashboard does not open, check that Python includes Tkinter and that all required packages were installed in the active virtual environment.

---

## 13. How to Use the Dashboard

### 13.1 Start with an Input Video

1. Put a supported video in `data/input/` or use **Browse Video** to select one.
2. Start the application with `python app.py`.
3. Select the desired video from the video list or dropdown.
4. Click **Start Detection**.
5. Watch the video frames and the Fire/Smoke bounding boxes in the dashboard.

### 13.2 Playback Controls

- **Start Detection:** Begins frame-by-frame analysis.
- **Pause / Resume:** Temporarily pauses or resumes frame processing.
- **Stop:** Stops the current analysis.
- **Browse Video:** Opens a file picker to choose a video from another folder.
- **Video selector:** Switches between videos found in `data/input/`.

### 13.3 Adjust Confidence Thresholds

The documented default thresholds are:

| Control | Default |
|---|---:|
| Fire threshold | 35% |
| Smoke threshold | 45% |

Move the sliders to change the minimum confidence accepted for each class. A higher threshold normally removes more low-confidence predictions, but it can also remove genuine Fire or Smoke detections. Choose thresholds using a labelled validation set and independent test footage rather than by appearance alone.

### 13.4 Enable or Disable the Sound

Use the **Enable Audible Beep Alert** control to turn the alert sound on or off. The visual alert should remain available even when sound is muted.

### 13.5 Review the Output

After processing finishes, open `data/output/` and look for the annotated video:

```text
data/output/annotated_<video-name>.mp4
```

Review the video to check whether boxes follow the actual fire or smoke region and whether the class labels are correct.

---

## 14. Detection Thresholds and Alerts

### 14.1 Separate Thresholds

Fire and Smoke are visually different classes and may require different confidence thresholds. The dashboard therefore provides separate controls.

A lower threshold can increase recall but may also increase false positives. A higher threshold can increase precision for accepted detections but may miss faint smoke, distant flames, or partially obscured objects.

### 14.2 Alert Behaviour

The application can show a visual alert and play a beep when qualifying detections are found. The exact alert timing depends on the implementation of the detection loop and any alert cooldown logic.

A sound alert is only an additional notification. It must not replace the dashboard warning or approved building alarm equipment.

### 14.3 Hybrid Verification Caution

The colour-based flame verifier is a rule-based helper. It may help highlight a bright flame-like region, but it can also react to non-fire sources. Verify all such detections and measure false positives on scenes containing welding, orange lights, reflections, sunlight, and similar sources.

---

## 15. Output Videos

The application saves annotated videos in:

```text
data/output/
```

Example:

```text
data/output/annotated_factory_video.mp4
```

The annotated video is useful for:

- Reviewing detections after processing.
- Checking whether bounding boxes follow the object.
- Comparing model output with the original footage.
- Demonstrating the project during a presentation.

Do not use an annotated output video as proof of model accuracy by itself. For formal evaluation, compare predictions against ground-truth annotations using a separate labelled test set.

---

## 16. Troubleshooting

### Problem: `best.pt` not found

Check that the model is located at:

```text
models/best.pt
```

Check the exact filename and spelling. Make sure the application is being run from the project directory or that `app.py` builds paths relative to its own location.

### Problem: The model does not recognize the classes

Print the model's class mapping:

```python
from ultralytics import YOLO

model = YOLO("models/best.pt")
print(model.names)
```

The expected mapping is equivalent to:

```python
{0: "fire", 1: "smoke"}
```

If the mapping is different, check that the correct custom checkpoint is being loaded.

### Problem: OpenCV cannot open the video

- Check that the video file is not corrupted.
- Try another MP4 video.
- Check whether the required video codec is available.
- Confirm that the extension is supported by the application's file selector.

### Problem: Output video is not created or cannot play

- Check that `data/output/` exists and is writable.
- Check the OpenCV `VideoWriter` codec and whether it opened successfully.
- Confirm that the output frame width and height are valid.
- Try a different codec or convert the output to a browser-/Windows-compatible MP4 format.

### Problem: Detection is too weak or too noisy

- Review the Fire and Smoke confidence thresholds separately.
- Check the original video quality, lighting, distance, and compression.
- Verify that the training labels are accurate.
- Evaluate on independent videos.
- Add representative real-world Fire, Smoke, and negative examples before fine-tuning again.

### Problem: Alert sound does not play

- Check the application's sound toggle.
- Check the Windows volume and output device.
- Confirm that the computer is not muted.
- If the application uses a sound library, verify that the relevant dependency is installed.

### Problem: Processing is slow

- Try a smaller inference image size if appropriate.
- Process fewer frames per second for offline review, while clearly recording the sampling interval.
- Use a compatible NVIDIA GPU if available.
- Avoid loading every video frame into memory at once.

---

## 17. Limitations and Future Improvements

### Current Limitations

1. **Small dataset:** The model was trained using a relatively small dataset of synthetic CCTV-style images.
2. **Domain gap:** Synthetic images may differ from real industrial camera footage.
3. **Smoke difficulty:** Smoke can be transparent, diffuse, distant, or visually similar to steam, dust, fog, and exhaust.
4. **Lighting and camera variation:** Night scenes, glare, compression, camera movement, and occlusion can affect predictions.
5. **Heuristic flame verification:** Colour-based rules can produce false positives and cannot guarantee detection of a flame hidden by smoke.
6. **Threshold trade-off:** Increasing confidence thresholds may reduce false alarms but can also miss real events.
7. **Hardware and speed:** Processing speed depends on the computer, GPU, input resolution, and video codec.
8. **Not safety-certified:** The application has not been established as a certified life-safety system.

### Future Improvements

- Collect and label more real industrial fire and smoke images.
- Add negative examples such as steam, fog, dust, welding sparks, exhaust, and reflections.
- Increase the size and diversity of the held-out test set.
- Evaluate performance on complete unseen camera views and videos.
- Track detections across frames to reduce repeated alerts and temporary false positives.
- Tune Fire and Smoke thresholds independently using validation data.
- Measure false alarms per camera-hour and detection latency.
- Add multi-camera monitoring and configurable camera names.
- Add incident search, filters, export, and reporting.
- Optimize model inference for the target computer.

---

## 18. Dataset License and Acknowledgement

The dataset page identifies the sample as **CC BY-NC 4.0**. Users should check the dataset's current license details on the original Hugging Face page before redistributing data or using it in any commercial context.

Dataset source:

<https://huggingface.co/datasets/Simuletic/CCTV-Smoke-Fire-Emergency-Detection-Dataset>

When using the dataset in reports, demonstrations, or research, acknowledge the dataset creator and follow the license terms. The software developer should also verify the licenses and usage terms for all software dependencies and pretrained model weights used in a distributed project.

---

## 19. Useful References

- **Dataset:** [CCTV Smoke & Fire Emergency Detection Dataset – Hugging Face](https://huggingface.co/datasets/Simuletic/CCTV-Smoke-Fire-Emergency-Detection-Dataset)
- **Ultralytics documentation:** [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
- **YOLO26 model documentation:** [https://docs.ultralytics.com/models/yolo26/](https://docs.ultralytics.com/models/yolo26/)
- **Ultralytics training mode:** [https://docs.ultralytics.com/modes/train/](https://docs.ultralytics.com/modes/train/)
- **Ultralytics validation mode:** [https://docs.ultralytics.com/modes/val/](https://docs.ultralytics.com/modes/val/)
- **OpenCV documentation:** [https://docs.opencv.org/](https://docs.opencv.org/)
- **PyTorch documentation:** [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
- **Python Tkinter documentation:** [https://docs.python.org/3/library/tkinter.html](https://docs.python.org/3/library/tkinter.html)

---

## Project Summary

Smoke and Fire Detection is a desktop-based AI monitoring prototype. It uses a pretrained YOLO26n model fine-tuned on a small CCTV-style Fire and Smoke dataset through Transfer Learning. OpenCV reads the video frames, the model predicts Fire and Smoke regions, and the application draws red and cyan/blue bounding boxes with confidence percentages. A supplementary RGB, HSV, YCrCb, and hot-core heuristic can help identify flame-like regions, while visual and audible alerts notify the user of possible detections. The analyzed video is saved to `data/output/` for later review.

The project demonstrates how Computer Vision and Deep Learning can support video-based fire and smoke monitoring. Its results must be validated on larger, independent, real industrial datasets before any safety-related deployment.
# smoke-and-fire-detection
