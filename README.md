# StudioSR
StudioSR is a PyTorch library providing implementations of training and evaluation of super-resolution models. StudioSR aims to offer an identical playground for modern super-resolution models so that researchers can readily compare and analyze a new idea. (inspired by [PyTorch-StudioGan](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN))


## Installation

### From [PyPI](https://pypi.org/project/studiosr/)
```bash
pip install studiosr
```

### From source (Editable)
```bash
git clone https://github.com/veritross/studiosr.git
cd studiosr
python3 -m pip install -e .
```


## Documentation
Documentation along with a quick start guide can be found in the [docs/](./docs/) directory.

### Quick Example

```bash
$ python -m studiosr --image image.png --scale 4 --model swinir
```

```python
from studiosr.models import SwinIR
from studiosr.utils import imread, imwrite

model = SwinIR.from_pretrained(scale=4).eval()
image = imread("image.png")
upscaled = model.inference(image)
imwrite("upscaled.png", upscaled)
```

### Train
```python
from studiosr import Evaluator, Trainer
from studiosr.data import DIV2K
from studiosr.models import SwinIR

dataset_dir="path/to/dataset_dir",
scale = 4
size = 64
dataset = DIV2K(
    dataset_dir=dataset_dir,
    scale=scale,
    size=size,
    transform=True, # data augmentations
    to_tensor=True,
    download=True, # if you don't have the dataset
)
evaluator = Evaluator(scale=scale)

model = SwinIR(scale=scale)
trainer = Trainer(model, dataset, evaluator)
trainer.run()
```

### Evaluate
```python
from studiosr import Evaluator
from studiosr.models import SwinIR
from studiosr.utils import get_device

scale = 2  # 2, 3, 4
dataset = "Set5"  # Set5, Set14, BSD100, Urban100, Manga109
device = get_device()
model = SwinIR.from_pretrained(scale=scale).eval().to(device)
evaluator = Evaluator(dataset, scale=scale)
psnr, ssim = evaluator(model.inference)
```


## Benchmark
- The evaluation metric is [PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio).
- You can check the full benchmark [here](./docs/benchmark.md).

| Method | Scale | Training Dataset | Set5   | Set14  | BSD100 | Urban100 |
| ------ | ----- | ---------------- | ------ | ------ | ------ | -------- |
| EDSR   | x 4   | DIV2K            | 32.485 | 28.814 | 27.721 | 26.646   |
| RCAN   | x 4   | DIV2K            | 32.639 | 28.851 | 27.744 | 26.745   |
| SwinIR | x 4   | DF2K             | 32.916 | 29.087 | 27.919 | 27.453   |
| HAT    | x 4   | DF2K             | 33.055 | 29.235 | 27.988 | 27.945   |

| Method | Scale | Training Dataset | Set5   | Set14  | BSD100 | Urban100 |
| ------ | ----- | ---------------- | ------ | ------ | ------ | -------- |
| EDSR   | x 3   | DIV2K            | 34.680 | 30.533 | 29.263 | 28.812   |
| RCAN   | x 3   | DIV2K            | 34.758 | 30.627 | 29.302 | 29.009   |
| SwinIR | x 3   | DF2K             | 34.974 | 30.929 | 29.456 | 29.752   |
| HAT    | x 3   | DF2K             | 35.097 | 31.074 | 29.525 | 30.206   |

| Method | Scale | Training Dataset | Set5   | Set14  | BSD100 | Urban100 |
| ------ | ----- | ---------------- | ------ | ------ | ------ | -------- |
| EDSR   | x 2   | DIV2K            | 38.193 | 33.948 | 32.352 | 32.967   |
| RCAN   | x 2   | DIV2K            | 38.271 | 34.126 | 32.390 | 33.176   |
| SwinIR | x 2   | DF2K             | 38.415 | 34.458 | 32.526 | 33.812   |
| HAT    | x 2   | DF2K             | 38.605 | 34.845 | 32.590 | 34.418   |

## License
StudioSR is an open-source library under the **MIT license**.



## For the Galaxy Image Restoration
### Environment setting
1. Conda 환경 생성
    ```sh
    conda create -n studiosr python=3.8
    ```

2. 생성한 환경 활성화
    ```sh
    conda activate studiosr
    ```

3. 필요한 패키지 설치
    ```sh
    pip install -r requirements.txt
    ```

### HAT model image size (32x32->64x64)
    - Training - 
        python trian_hat32.py
    
    - Inference -
        python test_hat32.py

### Note
기본적인 코드는 위의 명령어를 통하여 동작하며, min-max scaling을 수행한 데이터 기준으로 코드를 구성하였습니다. 
세부적인 코드 단위의 수정이 아래와 같이 필요합니다.

<b>Training</b>  
train_hat32.py 파일에 아래와 같은 내용의 수정이 필요합니다.
- dataset_dir  
Dataset dir가 아래와 같이 구성되어 있을 때의 root 경로를 지정해주시면 됩니다.  
```plaintext
├── root/
    ├── train/
    ├── valid/
    └── test/
```
- ckpt_path  
 ckpt파일의 저장을 희망하는 폴더명을 입력해주시면 됩니다.(폴더가 존재하지 않으면 폴더를 자동으로 생성합니다)

<b>Inference</b>
test_hat32.py 파일에 아래와 같은 내용의 수정이 필요합니다.  
- model_dir  
.pth 파일이 저장되어있는 폴더의 경로를 입력해주시면 됩니다

- test_dir  
test파일이 내포된 폴더의 경로를 입력해주시면 됩니다.

- save_path  
inference 결과물울 저장할 폴더의 경로를 입력해주면 됩니다.
