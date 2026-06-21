import urllib.request
import os

def download_file(url, filename):
    print(f"Downloading {filename}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"Successfully downloaded {filename}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

download_file("https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg", "yolov3-tiny.cfg")
download_file("https://pjreddie.com/media/files/yolov3-tiny.weights", "yolov3-tiny.weights")
download_file("https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names", "coco.names")

print("All YOLO files downloaded!")
