# Poison Cluster

## Overview of Different Attacks

### Mimicry-based Data Poisoning Attack: A Behaviour Borrowed from Nature

In Statistics and Machine Learning, many estimators and algorithms are sensitive to outliers. For example, calculating the mean of a distribution or performing clustering with KMeans are affected by the presence of extreme values. Oftentimes, it is advised to apply an outlier detection algorithm on the dataset and remove the extreme values or areas with low-density compared to the rest of the data. Applying outlier detection and issuing their blind removal may lead to unexpected and even fatal results in terms of the Machine Learning model that is to be trained by the cleaned data.

To exploit this vulnerability of the general Data Science pipeline, an interesting behaviour is borrowed from nature: *mimicry*, more specifically, the *cuckoo's parasitism*. In nature, many creatures in various environments follow mimicry. A behaviour that allows them to despite being harmless appear as harmful (to avoid being eaten by a predator) or to despite being harmful appear as a prey (to lure others into a trap). These creatures adapted to their environment, its dangers and opportunities to survive. A cuckoo selects another bird's nest to lay its eggs which mimic the characteristics of the eggs of the host as much as possible. The cuckoo's egg is an outlier yet by blending in the host environment, it avoids detection and fatal outcome of rejection by the host.

Similarly, considering outliers in a dataset, a data poisoning attack can ensure that these abnormal data points blend in the context by mimicking properties similar to the normal data, resulting in the misleading of the outlier detection mechanism. This is analogous to a creature to appear harmless or that is trying to avoid a predator (detection and purging mechanism).

Beyond the concealing effect that mimicry provides for the cuckoo’s egg to survive, there is a dark twist: once the cuckoo chick hatches, it may evict the host’s eggs from the nest, resulting in a contextual reversal—the cuckoo is no longer a minority. Similarly, if the mimicking process in outlier-prone regions not only strengthens but becomes dominant over the rest of the data, the autonomous outlier removal process may begin to recognize originally normal data as abnormal and purge them from the dataset.

In the following experiments only the mimicking part is covered, the contextual reversal stands as an idea to make the data poisoning attack more dangerous.

**Advantages:** Despite shifting the centroids along with the possibility to change cluster boundaries, it does not significantly change performance metrics, at least not for the worse. Some of the scores may even generate false confidence.

**Limitations:** Relies on the outliers in the dataset. If there are no outliers in the dataset then there is nothing to be concealed. In this case, manual injection of outliers may take place during the attack to ensure its success.

**Poisoned clustering results:**

![image](https://github.com/user-attachments/assets/592fb001-6867-423e-b5c5-d5a1b0034c03)

![image](https://github.com/user-attachments/assets/2d27bf04-5f0b-4ea1-bde7-c4b4cfc6fc8c)


### Hyperparameter Poisoning of Semi-supervised KMeans

A limited sample is used to determine centroids for each target class. The KMeans is initialized with these centroids. As the centroids traverse during the fitting process, label information follows them. The poisoning attack targets the sample data and flips its labels. This poisoning then may not result in the change of clustering but definietly will end up in misclassification.

**Advantages:** Only the sample data is needed to be poisoned. It makes entire cluster-wide mistakes. Centroids and cluster boundaries may remain unchanged.

**Limitations:** The poisoned model seems to be immune against label flipping of the sample data if the flip ratio is not large enough. According to the experiments, a reliable ratio could be $0.7$ for the _Iris_ and $0.6$ for the _Breast cancer_ dataset. Centroids and cluster boundaries may remain unchanged.


### Randomized Injection Attacks

This notebook explores how various noise injection techniques impact the behavior and reliability of clustering algorithms, particularly K-means, using the Iris dataset as a case study.

#### Attack Types Implemented

- **Gaussian Noise:** Adds small, random perturbations to every feature in the dataset, making clusters less distinct and harder to separate.
- **Uniform Outliers:** Introduces data points with values sampled far outside the original feature ranges. These extreme outliers are easy to identify but not always realistic.
- **Subtle Outliers:** Creates deceptive data points using the covariance structure of the dataset. These outliers are visually similar to valid samples and are more difficult to detect.
- **Label shuffling:** Shuffles the labels of the data, potentially confusing a semi supervised model when it compares the result of clustering and classification to ground truth.

#### Key Findings

- **K-means Vulnerabilities:** K-means is not designed to handle noise or outliers. It treats all points as equally valid, which can skew results.
- **Metric Misinterpretation:** Clustering evaluation metrics may show artificially high performance when distant noise forms its own cluster, misleading analysts.
- **Importance of Visualization:** Subtle attacks often go undetected by metrics alone. Visual inspection reveals cluster deformation more clearly.
- **Robustness Requirements:** Real-world applications require clustering methods that can tolerate both naive and adversarial noise injections.

The notebook demonstrates the fragility of standard clustering approaches in noisy environments and underscores the need for robust analytics pipelines.

## Usage Guide


### Mimicry-based Data Poisoning Attack and Hyperparameter Poisoning

[Mimicry Notebook (Iris)](synthetic_injection/mimicry_based_attack_(Iris).ipynb)

[Mimicry Notebook (Breast cancer)](synthetic_injection/mimicry_based_attack_(Breast_cancer).ipynb)

[Hyperparameter Poisoning Notebook (Iris)](synthetic_injection/exploit_semi_supervised_KMeans_(Iris).ipynb)

[Hyperparameter Poisoning Notebook (Breast cancer)](synthetic_injection/exploit_semi_supervised_KMeans_(Breast_cancer).ipynb)

#### 1. **Data Setup**
The notebooks download the `Iris` and `Breast cancer` datasets, respectively, using `sklearn.datasets` and store the dataset in Pandas DataFrame for ease of manipulation. The appropriate dataset in each notebook is explored and preprocessed.
#### 2. **Helpers**
In this section, various helper functions are placed that are used during modeling.
#### 3. **Modeling**
Main class and method definitions are located here. Poisoning the KMeans clustering model also takes place here.
#### 4. **Evaluation & Comparison**
Different evaluation scores are calculated for each model. These scores along with visualizations in the two-dimensional Principal Component space are used to compare the effects of the attacks. Advantages and limitations of the attacks are also included in this section.



### Randomized Injection Attacks

Open the notebook in a jupyter notebook environment and execute the cells in order.  
The cells are organized as follows:

#### 1. **Data Setup**

The notebook downloads the Iris dataset using `sklearn.datasets` and stores it in Pandas DataFrames for ease of manipulation.

#### 2. **Clustering Algorithms**

It applies `KMeans` clustering on:

- Clean/original data.
- Noisy data with injected random points.
- Data with explicit and subtle outliers.
- Data with shuffled labels (simulating mislabeling).

#### 3. **Visualization**

Custom plotting functions visualize the clustering results with:

- Color-coded cluster labels.
- Visual comparisons between clean and poisoned data clusters.

#### 4. **Evaluation**

For each scenario, clustering quality is evaluated using:

- `silhouette_score`
- `calinski_harabasz_score`
- `davies_bouldin_score`

These metrics are compared across different scenarios to observe how noise affects cluster structure and algorithm performance.

## Summary of the References

### Tian et al. (2022) - A Comprehensive Survey on Poisoning Attacks and Countermeasures in Machine Learning

Tian et al. (2022) provide a comprehensive review of poisoning attacks in machine learning, which involve injecting malicious data into training datasets to degrade model performance. The authors classify these attacks into **availability attacks**, which reduce overall accuracy, and **integrity attacks**, which manipulate predictions for specific inputs. Various attack techniques, such as **label flipping, gradient manipulation, and backdoor attacks**, are discussed alongside their impact on supervised, semi-supervised, and federated learning. The paper also explores countermeasures, including **robust training, anomaly detection, and data sanitization**. However, detecting poisoning attacks remains challenging due to their subtle nature and similarity to normal data. The authors examine the trade-off between model robustness and performance, highlighting the need for more effective defense mechanisms. Additionally, they identify open research questions, such as the necessity for real-world evaluations and adversarially robust architectures. The study concludes that a combination of **proactive defenses and adaptive detection mechanisms** is essential to mitigate the risks posed by poisoning attacks.

### Biggio et al. (2012) - Poisoning Attacks Against Support Vector Machines

Biggio et al. (2012) focus specifically on poisoning attacks targeting **Support Vector Machines (SVMs)**. They propose an **optimization-based attack strategy** that modifies training data to maximize classification errors. Their experiments show that even a small fraction of poisoned data can significantly degrade the performance of SVMs, highlighting vulnerabilities in their learning algorithms. The study evaluates attack effectiveness in both binary and multi-class classification settings and discusses potential countermeasures, such as **robust optimization techniques and data filtering**. A key insight is that poisoning attacks are difficult to detect because they manipulate the **learning process** rather than the **input-output relationship**. The authors emphasize the importance of **continuous monitoring and robustness testing** in real-world applications. They also suggest further research on defenses that integrate **adversarial training and anomaly detection** to protect SVMs from poisoning attacks.

### Demontis et al. (2019) - Why Do Adversarial Attacks Transfer? Explaining Transferability of Evasion and Poisoning Attacks

Demontis et al. (2019) explore the reasons behind the **transferability** of adversarial attacks, including both **evasion and poisoning attacks**. The authors analyze how these attacks generalize across different models by exploiting **shared decision boundaries and feature space similarities**. Their study examines transferability in both **white-box and black-box settings**, showing that adversarial examples can remain effective even without direct knowledge of a target model. They conduct experiments on multiple machine learning models, such as **neural networks and SVMs**, to demonstrate that poisoning attacks can generalize across **datasets and architectures**. The paper discusses the implications of attack transferability for model robustness and proposes new techniques to **measure and mitigate this phenomenon**. The authors highlight how **adversarial training and feature space regularization** can reduce susceptibility to transferred attacks. Ultimately, the study concludes that understanding attack transferability is crucial for designing **more secure machine learning models**.
