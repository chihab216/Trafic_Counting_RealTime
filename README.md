
- Run `pip install -r requirements.txt` to install dependencies.



### Configuration
```
usage: Vehicle_Counting.py [-pathvideo] 
                           [--mctf MCTF] [--di DI] [--detector DETECTOR]
                           [--tracker TRACKER] [--record]
                           [--clposition CLPOSITION]
                           video

positional arguments:
  pathvideo                 relative/absolute path to video 
optional arguments:
 
 
  --mctf MCTF           maximum consecutive tracking failures i.e number of
                        tracking failures before the tracker concludes the
                        tracked object has left the frame
  --di DI               detection interval i.e number of frames before
                        detection is carried out again (in order to find new
                        vehicles and update the trackers of old ones)
  --detector DETECTOR   select a model/algorithm to use for vehicle detection
                        (options: yolo, haarc, ssd,bgsub | default: yolo)##bgsub not already work
  --tracker TRACKER     select a model/algorithm to use for vehicle tracking
                        (options: csrt, kcf | default: kcf)
  --record              record video and vehicle count logs
  --clposition CLPOSITION
                        position of counting line (options: top, bottom, left,
                        right | default: bottom)
```

### Notes
- To use the `yolo` detector, download the [YOLO v3 weights](https://pjreddie.com/media/files/yolov3.weights) and place it in the [detectors/yolo folder](/detectors/yolo).
- To use the `ssd` detector, download this [pre-trained model](https://drive.google.com/file/d/0BzKzrI_SkD1_WVVTSmQxU0dVRzA/view) and place it in the [detectors/ssd folder](/detectors/ssd).

### Examples

defaults:
```
python Vehicle_Counting.py "C:\yolo\pytorch-yolo-v3\videohanoi.mp4"
```

Custom configuration:

```
python Vehicle_Counting.py "C:\yolo\pytorch-yolo-v3\videohanoi.mp4"  --detector "haarc" --tracker "kcf" --di 5 --mctf 15
```





```
