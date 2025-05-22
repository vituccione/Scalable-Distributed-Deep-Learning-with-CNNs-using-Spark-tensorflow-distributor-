# Scalable Distributed Deep Learning with CNNs using Spark-tensorflow-distributor

This project demonstrates how to train Convolutional Neural Networks (CNNs) in a distributed fashion using **TensorFlow** and **Apache Spark**, specifically leveraging the `spark-tensorflow-distributor` library to scale training across Spark executors.

---

## 📌 Overview

As datasets grow in volume and complexity, training deep learning models like CNNs on single machines becomes inefficient. This project integrates **Spark** and **TensorFlow** to enable distributed training, optimizing both training time and scalability, even in CPU-bound environments.

- **Frameworks used**: Apache Spark 3.4.3, TensorFlow 2.16.1
- **Model**: CNN trained on CIFAR-10
- **Environment**: Local Hadoop Spark cluster (Standalone mode, CPU-based)
- **Goal**: Reduce training time without significantly impacting model accuracy

---

## 🔍 Key Features

- Distributed EDA and preprocessing using Apache Spark
- Training CNN models on a Spark cluster via `spark-tensorflow-distributor`
- Evaluation of training time, convergence, and accuracy across:
  - **1-slot (single executor)**
  - **2-slot (multi-executor)** configurations
- Implementation of data sharding, synchronization, and checkpointing
- Analysis of distributed training challenges and performance bottlenecks

---

## 📊 Results

| Metric           | 1-Slot Model | 2-Slot Model | Difference      |
|------------------|--------------|--------------|-----------------|
| Test Accuracy    | 70.9%        | 67.6%        | -3.3%           |
| Test Loss        | 0.869        | 0.961        | +0.092          |
| Training Time    | 56m 39s      | 48m 35s      | -14% faster     |

- Distributed training achieved **14% faster runtime**
- Accuracy drop of ~3% due to gradient synchronization overhead

---

## 🧪 Dataset

- **CIFAR-10**: 60,000 32x32 color images in 10 classes
- Preprocessed into **TFRecord format** and stored in HDFS
- EDA performed with Spark to validate:
  - Class balance
  - Brightness and contrast
  - Pixel range distribution

---

## 🧱 Model Architecture

- 2 convolutional blocks with **BatchNorm** and **ReLU**
- Max pooling and **Dropout (0.25-0.5)** for regularization
- Final **Dense** layer with 10-class softmax output
- Optimizer: **Adam** with fixed learning rate (0.001)

---

## ⚙️ Technologies Used

- Apache Spark (Standalone mode)
- TensorFlow
- Jupyter Notebook (for orchestration)
- HDFS for data and model storage
- Spark History Server for job monitoring

---

## 🧩 Limitations & Future Work

- System constrained to CPU-based single-node cluster
- Accuracy degradation due to synchronization delays
- Opportunities to improve via:
  - GPU acceleration
  - Data augmentation
  - Monitoring with TensorBoard
  - Model versioning with MLflow
 
## 👨‍💻 Author

**Marco Vitucci**  
MSc Data Analytics – CCT College Dublin  
📧 marco.vitucci@outlook.it

---

## 📜 Citation

If you use this project or build upon it, please cite:  
**Vitucci, M. (2025)**. *Scalable Distributed Deep Learning with CNNs using Spark-tensorflow-distributor*. CCT College Dublin.
