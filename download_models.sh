#!/bin/bash

# Создаем директории, если они не существуют
mkdir -p backend/models
mkdir -p gfpgan/weights

echo "--- Downloading GFPGAN v1.4 ---"
wget -O backend/models/GFPGANv1.4.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth

echo "--- Downloading CodeFormer ---"
wget -O backend/models/codeformer.pth https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth

echo "--- Downloading Detection and Parsing models ---"
wget -O gfpgan/weights/detection_Resnet50_Final.pth https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth
wget -O gfpgan/weights/parsing_parsenet.pth https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_parsenet.pth

echo "Done! All models have been downloaded to their respective folders."
