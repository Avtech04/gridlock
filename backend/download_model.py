import urllib.request
import os

print("Downloading MobileNet SSD Prototxt...")
prototxt_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/voc/MobileNetSSD_deploy.prototxt"
urllib.request.urlretrieve(prototxt_url, "MobileNetSSD_deploy.prototxt")

print("Downloading MobileNet SSD Caffemodel...")
caffemodel_url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/voc/MobileNetSSD_deploy.caffemodel"
urllib.request.urlretrieve(caffemodel_url, "MobileNetSSD_deploy.caffemodel")

print("Download complete!")
