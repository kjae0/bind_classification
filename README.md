# bind_classification
image classification project for BIND

## 사용법
train.py 실행 후, inference.py 실행하여 prediction csv 파일 생성

## Example
### train.py
```
python train.py --img_dir [dataset directory] \
                --n_epochs [number of epochs] \
                --name [directory name for save files] \
                --model [architecture of neural net. efficientnetb3 / resnet50] \
```
batch size, optimizer, seed, n_gpu, gpu_number 등 다양한 arguments들 지정 가능.

### inference.py
```
python inference.py --img_dir [dataset directory] \
                    --save_dir [directory directory for save file] \
                    --model_dir [directory of saved model] \
```


