# ImageTT
ImageTT (Image Translator Typesetter) is a personal project created to help me translate scanned images to support my recreational activities and to support translations at my work place.

My ultimate goal is to create a program which can translate, typeset all kinds of scanned images to all kinds of preferred languages.

ImageTT is separated into 6 parts, respectively:
1) detection
2) ocr
3) translation
4) inpainting
5) merging texts
6) upscaling image

## Data
The data used at the moment will be focused on manga images (my recreational activity) as manga texts are one of the most commonly scanned images on the internet.
After a successful model has been created, other variations of scanned images such as textbooks or simple pdfs will be used.

Most ideally, this program should be able to translate and typeset all kinds of scanned images including textbooks, manga, comics, webpages, pdfs, word, ppt, etc..


## 1) Detection
To successfully inpaint text regions and to help ocr detect texts easily, detecting 'text objects' is necessary.
At the moment the following models have been tested:
- yolov5 (Local) => training in progress
- roboflow (API) => 90% accuracy
- crnn (Local) => training in progress

Ideally, if crnn is able to achieve a higher accuracy than roboflow then crnn will be used as the base model.

## 2) OCR
- tensorflow (works quite well)
- pytorch
- keras

## 3) translation
- gpt3 (api key)
- gpt3.5 (api key)
- gpt4 (api key)
- Google (online)
- youdao (api key)
- baidu (api key)
- deepl (api key)
- caiyun (api key)
- sugoiv4 (offline)
- m2m100 (offline)

## 4) Inpainting
- To research
## 5) Merging texts
- To research
## 6) Upscaling images
- could increase resolution
- could consider colorisation
