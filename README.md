# ADGSyn: Dual-Stream Learning for Efficient Anticancer Drug Synergy Prediction

You can get the paper from : [ADGSyn](https://arxiv.org/abs/2505.19144)

## 🌟 Abstract
Drug combinations play a critical role in cancer therapy by significantly enhancing treatment efficacy and overcoming drug resistance. However, the combinatorial space of possible drug pairs grows exponentially, making experimental screening highly impractical. Therefore, developing efficient computational methods to predict promising drug combinations and guide experimental validation is of paramount importance. In this work, we propose ADGSyn, an innovative method for predicting drug synergy. The key components of our approach include: (1) shared projection matrices combined with attention mechanisms to enable cross-drug feature alignment; (2) automatic mixed precision (AMP)-optimized graph operations that reduce memory consumption by 40\% while accelerating training speed threefold; and (3) residual pathways stabilized by LayerNorm to ensure stable gradient propagation during training.Evaluated on the O’Neil dataset containing 13,243 drug–cell line combinations, ADGSyn demonstrates superior performance over eight baseline methods. Moreover, the framework supports full-batch processing of up to 256 molecular graphs on a single GPU, setting a new standard for efficiency in drug synergy prediction within the field of computational oncology.

## ✨DualStreamAttention Module

![image](https://github.com/user-attachments/assets/4092ea67-6252-4156-9a62-d77ee886603d)


## 🔧Requirements

```
python        3.11.5
pytorch       2.3.1
numpy         1.25.2
scikit-learn  1.3.0
```



## 🔨How to Start

Locate the `Data.md` file within the `data` directory to retrieve the data. Then, execute the `main.py` script in the `src` directory to proceed.

```bash
python main.py 
```
## 📧 Contact
If you have any question, please email `nyxchaoji123@163.com`.

## 🤗 Acknowledgements

Thanks to the Yunnan Provincial Basic Research Youth Project 🌟 for the support, to my supervisor for the guidance, and to [sweetyoungthing](https://github.com/sweetyoungthing) for the assistance 🙌.
