# Human Activity Recognition Using CSI and Deep Learning

## 🔍 Goal
This project implements a deep learning-based system to classify human activities using Wi-Fi Channel State Information (CSI) data. The dataset used is from Figshare and includes CSI values along with activity labels.

## 🧠 Project Goals
- Preprocess CSI data and align it with activity labels.
- Train a CNN-LSTM model for multi-class activity classification.
- Evaluate performance and explore enhancements using bounding box data.

## 📁 Dataset
The dataset used in this project is from Figshare:​

> **Title:** Dataset for Human Activity Recognition using Wi-Fi Channel State Information (CSI) data
> 
> **Authors:** Andrii Zhuravchak, Oleh Kapshii
> 
> **Source:** https://figshare.com/articles/dataset/Dataset_for_Human_Activity_Recognition_using_Wi-Fi_Channel_State_Information_CSI_data/14386892
> 
> **License:** CC BY 4.0

This dataset includes:
- `data.csv`: Raw CSI data per packet.
- `label.csv`: Activity labels per sample.
- `label_boxes.csv`: Bounding box information (optional extension).

## 🚀 Getting Started
1. Download the dataset and place it in the `data/` folder.
2. Run `scripts/preprocess.py` to generate `X.npy` and `y.npy`.
3. Run `scripts/train_model.py` to train the deep learning model.

## 🛠️ Requirements
- Python 3.8+
- PyTorch
- NumPy
- pandas
- scikit-learn

## 📌 Notes
- Adjust number of subcarriers or reshape logic in `preprocess.py` based on actual dataset shape.
- Optional: Use `label_boxes.csv` for incorporating spatial features.

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
