# Investigating the Impact of Rational Dilated Wavelet Transform on Motor Imagery EEG Decoding with Deep Learning Models


The impact of a Rational Dilated Wavelet Transform (RDWT) used as a plug-in preprocessing module before standard EEG classifiers. RDWT employs non-integer (rational) dilation factors (e.g., 3/2, 5/3), yielding a more flexible tiling of the time–frequency plane and potentially improving denoising and rhythm-specific enhance-ment in motor imagery settings. Rather than proposing a new network, our objective is to quantify when and how much RDWT helps across eterogeneous backbones and datasets.

Authors : Giuseppe Bonomo, Marco Siino, Rosario Sorbello, Ilenia Tinnirello

University of Palermo, Italia 

---
The repository includes implementations of several other well-known EEG classification architectures in the `models.py` file, which can be used as baselines for comparison. These include:

- **EEGNet**:[paper](https://arxiv.org/abs/1611.08024), [original code](https://github.com/vlawhern/arl-eegmodels)
- **EEG-TCNet**:[paper](https://arxiv.org/abs/2006.00622), [original code](https://github.com/iis-eth-zurich/eeg-tcnet)
- **MBEEG_SENet**:[paper](https://www.mdpi.com/2075-4418/12/4/995)
- **ShallowConvNet**:[paper](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730), [original code](https://github.com/braindecode/braindecode)

----
# Comparative preprocessing  

The following table presents a comparative analysis of different deep learning models with and without the application of the RDWT (Rational Dilated Wavelet Transform) preprocessing technique. The evaluation covers three benchmark EEG motor imagery datasets: BCI Competition IV-2a, BCI Competition IV-2b, and the High-Gamma Dataset (HGD). The aim is to assess the impact of RDWT on classification performance (accuracy and Cohen’s kappa score).


| Model           | Preprocessing | BCI 2a Acc. | BCI 2a κ | BCI 2b Acc. | BCI 2b κ | HGD Acc. | HGD κ |
|----------------|---------------|-------------|----------|-------------|----------|----------|--------|
| EEGTCNet       | None          | 64.35       | 52.50    | 95.81       | 58.90    | 86.60    | 82.10  |
|                | RDWT          | 68.79       | 58.40    | 96.09       | 66.60    | 87.14    | 82.90  |
| MBEEG_SENet    | None          | 70.49       | 60.60    | 96.39       | 67.10    | 88.61    | 84.80  |
|                | RDWT          | 72.72       | 63.60    | 96.53       | 64.60    | 90.26    | 87.00  |         
| ShallowConvNet | None          | 65.74       | 54.30    | 95.85       | 61.80    | 87.05    | 82.70  |
|                | RDWT          | 66.32       | 55.10    | 95.94       | 62.30    | 87.27    | 87.27  |
| EEGNet         | None          | 68.02       | 57.40    | 95.85       | 59.60    | 87.32    | 83.10  |
|                | RDWT          | 70.10       | 60.10    | 96.06       | 64.00    | 88.08    | 84.10  |

### **Note**   
> - The recomputed results for these datasets (including accuracy/kappa scores) are available in their respective dataset folders.  
> - *Unlike the previous table, the results reported here for the HGD and BCI IV-2b datasets include an enhanced preprocessing pipeline, which incorporates data augmentation and class balancing techniques. These strategies were employed to address class imbalance and improve the generalization capabilities of the models.*
> - These values were obtained using our implementation and preprocessing pipeline. Minor deviations from the original papers are expected.

### **Disclaimer**   
> - The results reported for **BCI 4-2a**, **BCI 4-2b** and **HGD** datasets were not recomputed by us and are directly extracted from the original papers.  
> - **HGD (High Gamma Dataset)**: Refers to physically executed movements (executed movements), not motor imagery (motor imagery).

----

# Dataset

This project uses three publicly available EEG motor imagery datasets for training and evaluation:

### 1. BCI Competition IV – Dataset 2a

- **Description**: EEG data from 9 subjects performing four different motor imagery tasks: left hand, right hand, feet, and tongue movements. Each subject completed two sessions (training and evaluation), with 288 trials per session.
- **Format**: `.mat` files
- **Download**: [BCI Competition IV – Dataset 2a](https://bnci-horizon-2020.eu/database/data-sets/001-2014/)

### 2. BCI Competition IV – Dataset 2b

- **Description**: EEG recordings from 9 subjects performing left and right hand motor imagery tasks. The dataset contains five sessions per subject, with three sessions including feedback.
- **Format**: `.gdf` files
- **Download**: [BCI Competition IV – Dataset 2b](https://www.bbci.de/competition/iv/download/)

### 3. High-Gamma Dataset (HGD)

- **Description**: EEG recordings from 14 subjects performing motor execution tasks, recorded using 128 channels. This dataset is well-suited for high-frequency EEG analysis.
- **Format**: `.mat` files
- **Download**: Available through the [GIN Repository](https://web.gin.g-node.org/robintibor/high-gamma-dataset)

> **Note**: Each dataset has a dedicated notebook in this repository, which includes download links and preprocessing instructions.


