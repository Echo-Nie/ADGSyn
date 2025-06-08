# ADGSyn

This repository contains the source code for the paper **"ADGSyn: Dual-Stream Learning for Efficient Anticancer Drug Synergy Prediction"**, This paper is available at [here](https://arxiv.org/abs/2505.19144).



## 🌟 Abstract

Drug combinations play a critical role in cancer therapy by significantly enhancing treatment efficacy and overcoming drug resistance. However, the combinatorial space of possible drug pairs grows exponentially, making experimental screening highly impractical. Therefore, developing efficient computational methods to predict promising drug combinations and guide experimental validation is of paramount importance. In this work, we propose ADGSyn, an innovative method for predicting drug synergy. The key components of our approach include: (1) shared projection matrices combined with attention mechanisms to enable cross-drug feature alignment; (2) automatic mixed precision (AMP)-optimized graph operations that reduce memory consumption by 40% while accelerating training speed threefold; and (3) residual pathways stabilized by LayerNorm to ensure stable gradient propagation during training.Evaluated on the O’Neil dataset containing 13,243 drug–cell line combinations, ADGSyn demonstrates superior performance over eight baseline methods. Moreover, the framework supports full-batch processing of up to 256 molecular graphs on a single GPU, setting a new standard for efficiency in drug synergy prediction within the field of computational oncology.



## ✨ DualStream-Atten

![DualStreamAttention Module](https://github.com/user-attachments/assets/4092ea67-6252-4156-9a62-d77ee886603d)

## 🔧 Installation

**Prerequisites** 

Make sure you have Python 3.8+ and either `pip` or `conda` installed.

**Option 1: Using pip** 

Install dependencies via pip:

```bash
pip install -r requirements.txt
```

**Option 2: Using Conda (Recommended)** 

Create a virtual environment using Conda:

```bash
conda env create -f environment.yml
conda activate synpred-env
```

To verify installation:

```bash
conda list
```

## 🔨 Getting Started

Before running the model, please ensure that the data is correctly set up.

**Step 1: Prepare the Data** 

Refer to the `Data.md` file located in the `data/` directory for detailed instructions on how to download and preprocess the dataset.

> 💡 The default data path used by the script is `data/data.pt`. Make sure your processed data is placed accordingly.

**Step 2: Run the Training Script** 

Navigate to the `src/` directory and run the main script:

```bash
cd src
python main.py
```

By default, the script will train the model using the configuration specified in the code. You can modify hyperparameters directly in `main.py` or extend it to support command-line arguments if needed. 

## 📤 Contact

If you have any questions or need further assistance, feel free to reach out via email:  📧 nyxchaoji123@163.com. 

## 🙌 Acknowledgements

We gratefully acknowledge the support from the Yunnan Provincial Basic Research Youth Project. We also thank our supervisor for guidance and [sweetyoungthing](https://github.com/sweetyoungthing) for assistance.

## Citation
If you find ADGSyn useful in your research or applications, please kindly cite:
```
@article{nie2025adgsyn,
  title={ADGSyn: Dual-Stream Learning for Efficient Anticancer Drug Synergy Prediction},
  author={Nie, Yuxuan and Song, Yutong and Peng, Hong},
  journal={arXiv preprint arXiv:2505.19144},
  year={2025}
}
```
