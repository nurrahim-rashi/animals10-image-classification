# Animals-10 Image Classification

A convolutional neural network that classifies animal photos into 10 classes: `butterfly`, `cat`,
`chicken`, `cow`, `dog`, `elephant`, `horse`, `sheep`, `spider`, and `squirrel`. The model uses
transfer learning (MobileNetV2) with custom `Conv2D` and pooling layers, and is exported to
SavedModel, TensorFlow Lite, and TensorFlow.js.

This is my final project for Dicoding's machine learning course.

## Dataset

- Source: [Animals-10 on Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10)
- Original size: 26,179 images in 10 classes
- Resolution: not uniform (1,002 different resolutions, from 60x57 to 6720x6000)
- After cleaning: 25,596 images
- Split (stratified): 80% train (20,476), 10% validation (2,560), 10% test (2,560)

The dataset is not included in this repository. The notebook downloads it with `kagglehub`.

### Data cleaning

Animals-10 was collected from Google Images, so it contains images that are not real animals, such as
tattoos, clip art, toys, and food. I removed 583 of them; they are listed in `excluded_images.txt`.

- Butterfly class: images that a pre-trained ImageNet model scored as unlikely to be an animal were removed automatically.
- Other classes: the lowest-scoring images were reviewed by eye and only the clearly wrong ones were removed.

The cleaning is not exhaustive. Some non-animal images may remain, and some real butterfly photos were removed by the automatic rule.

## Results

| Set        | Accuracy | Loss   |
|------------|----------|--------|
| Training   | 97.63%   | 0.0835 |
| Validation | 96.76%   | 0.1186 |
| Testing    | 95.98%   | 0.1532 |

Training stopped after 7 of 10 epochs, when both training and validation accuracy passed 96%.
The exported TF-Lite model scored 97.20% on a random sample of 500 test images.

Per-class F1 on the test set ranges from 0.89 (cow) to 0.99 (spider); cow, sheep, and horse are the
classes most often confused with each other.

## Model

A Keras `Sequential` model:

1. `Rescaling`: normalizes pixels to [-1, 1]
2. `MobileNetV2`: pre-trained on ImageNet, frozen, used as the feature extractor
3. `Conv2D` (128 filters, 3x3) and `MaxPooling2D`
4. `GlobalAveragePooling2D`, `Dropout`, `Dense(128)`, `Dropout`
5. `Dense(10, softmax)`: output layer

Training uses the Adam optimizer, augmentation on the training set only (flip, rotation, zoom, contrast),
and four callbacks: a custom target-accuracy callback, `EarlyStopping`, `ModelCheckpoint`, and `ReduceLROnPlateau`.

## Project structure

```
submission
├── tfjs_model
│   ├── group1-shard1of4.bin ... group1-shard4of4.bin
│   └── model.json
├── tflite
│   ├── model.tflite
│   └── label.txt
├── saved_model
│   ├── saved_model.pb
│   ├── fingerprint.pb
│   ├── assets
│   └── variables
├── notebook.ipynb
├── notebook.py
├── excluded_images.txt
├── README.md
└── requirements.txt
```

| File / folder         | Purpose |
|-----------------------|---------|
| `notebook.ipynb`      | Main notebook, already executed: data exploration, cleaning, training, evaluation, model conversion, and inference. |
| `notebook.py`         | The same code as a Python script. |
| `excluded_images.txt` | Images removed during data cleaning, one `<class folder>/<file name>` per line. |
| `saved_model/`        | Model in SavedModel format, for servers and cloud deployment. |
| `tflite/`             | Model in TF-Lite format for mobile and embedded devices. `label.txt` lists the class names in output order. |
| `tfjs_model/`         | Model in TensorFlow.js format, for running in the browser. |
| `requirements.txt`    | Python libraries and versions used. |

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

Training takes about 75 minutes on a laptop CPU (Apple M2); a GPU is much faster.

Note: `tensorflowjs` depends on `tensorflow_decision_forests` and `jax`, which this project does not use
and which are not available for every Python version. The notebook handles this automatically. If `pip`
fails on those dependencies, install it with `pip install --no-deps tensorflowjs`.

## Using the model

All exported models take a 224x224 RGB image with pixel values in 0-255 (normalization is built into
the model) and return 10 probabilities in the order given in `tflite/label.txt`.
