# Cats vs Dogs Image Classifier Using Transfer Learning
**97% Accuracy | with InceptionV3**  


## 📌 Key Features
- **97% accuracy** on both training and validation sets
- **Transfer learning** with `InceptionV3` pre-trained on ImageNet
- **Data augmentation** to prevent overfitting
- **End-to-end pipeline** in a single Jupyter notebook
- **Optimized for Google Colab** (GPU-ready)

## 🚀 Performance Metrics
| Metric       | Training | Validation |
|-------------|----------|------------|
| Accuracy    | 96%      | 97%        |
| Loss        | 0.08     | 0.09       |

*Achieved in just 10 epochs with 90/10 train-val split*

## 🛠️ Implementation Highlights
```python
# Core components
base_model = InceptionV3(weights='imagenet', include_top=False)
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1, activation='sigmoid')(x)
model = Model(base_model.input, x)
