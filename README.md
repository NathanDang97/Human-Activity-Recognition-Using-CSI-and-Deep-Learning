# Human Activity Recognition Using CSI and Deep Learning

## 🧠 Overview 
### Project Goal
This project builds a deep learning model to recognize human activities based on Wi-Fi Channel State Information (CSI) data. Using CSI enables contactless activity recognition without relying on wearable sensors. We implement a full pipeline, from loading and visualizing CSI signals, to building and training a CNN-LSTM model, to fine-tuning and evaluating performance. The project is structured for educational purposes, aiming to help learners understand how to process time-series signal data for classification tasks.

### Project Structure
- 📈 CSI Data Visualization: Explore signal patterns using heatmaps and subcarrier plots.
- 🧹 Data Preprocessing: Normalize and reshape CSI data into suitable input for neural networks.
- 🧠 Model Architecture: CNN-LSTM hybrid model capturing spatial and temporal signal features.
- 🏋️ Training and Fine-Tuning: Training loop, evaluation metrics, and simple fine-tuning for demonstration.
- 📊 Evaluation: Test set accuracy and confusion matrix to assess classification performance.

## 📁 Dataset

### 🌐 Source
The dataset used in this project is from Figshare:​

> **Title:** Dataset for Human Activity Recognition using Wi-Fi Channel State Information (CSI) data
> 
> **Authors:** Andrii Zhuravchak, Oleh Kapshii
> 
> **Source:** https://figshare.com/articles/dataset/Dataset_for_Human_Activity_Recognition_using_Wi-Fi_Channel_State_Information_CSI_data/14386892
> 
> **License:** CC BY 4.0

### 🗃️ Dataset Scope
The full dataset (Zhuravchak & Kapshii, 2021) consists of multiple recording sessions performed in different indoor environments and across both 2.4 GHz and 5 GHz Wi-Fi channels. Each session includes labeled CSI data for a range of human activities, with variations in room layout, frequency band, and device configuration.

For this project, I used Session 1 in Room 1, which was recorded using a 5 GHz Wi-Fi channel. This subset was selected to simplify experimentation and gain practical experience in handling CSI-based activity recognition, without introducing cross-session or cross-frequency variability. More details can be found in the section below.

### 📐 Dataset Structure Note
The CSI data in this project consists of 1026 features per sample, which we reshape into matrices of **114 subcarriers × 9 time steps**. This structure reflects the wireless channel’s frequency and temporal characteristics, enabling deep learning models to learn both spectral and sequential patterns. The number of subcarriers (114) is consistent with CSI collected on 5 GHz Wi-Fi channels using wider bandwidths (e.g., 80 MHz).

This dataset includes:
- `data.csv`: Raw CSI data per packet.
- `label.csv`: Activity labels per sample.
- `label_boxes.csv`: Bounding box information (optional extension).

## ▶️ Run the Training Pipeline
The source code can be found in the _src_ folder. Before running the code, make sure the _data_ folder contains _data.csv_ and _label.csv_.
Then run:
```
python main.py
```
Or, run with custom hyperparameters, e.g.:
```
python main.py --epochs 30 --batch_size 128 --learning_rate 0.0005 --hidden_dim 256
```

## 🔍 Results
- **Accuracy:** ~98%
- **Note:** Although fine-tuning yielded only a modest accuracy improvement, the structure of the code allows for easy future experimentation and optimization.

<details> <summary>📚 <strong>BibTeX Citation</strong></summary>
@dataset{zhuravchak2021csi,
  author       = {Andrii Zhuravchak and Oleh Kapshii},
  title        = {{Dataset for Human Activity Recognition using Wi-Fi Channel State Information (CSI) data}},
  year         = 2021,
  publisher    = {figshare},
  doi          = {10.6084/m9.figshare.14386892},
  url          = {https://figshare.com/articles/dataset/Dataset_for_Human_Activity_Recognition_using_Wi-Fi_Channel_State_Information_CSI_data/14386892}
}
</details>
