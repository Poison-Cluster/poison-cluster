# Poison Cluster

## Overview of Different Attacks

### Mimicry-based Data Poisoning Attack: A Behaviour Borrowed from Nature
![image](https://github.com/user-attachments/assets/1148de88-dab9-4df1-a1f8-5bf3caadd478)

*Image source: [The past, present and future of 'cuckoos versus reed warblers'](https://www.researchgate.net/publication/256655356_The_past_present_and_future_of_'cuckoos_versus_reed_warblers')*

In Statistics and Machine Learning, many estimators and algorithms are sensitive to outliers. For example, calculating the mean of a distribution or performing clustering with KMeans are affected by the presence of extreme values. Oftentimes, it is advised to apply an outlier detection algorithm on the dataset and remove the extreme values or areas with low-density compared to the rest of the data. Applying outlier detection and issuing their blind removal may lead to unexpected and even fatal results in terms of the Machine Learning model that is to be trained by the cleaned data.

To exploit this vulnerability of the general Data Science pipeline, an interesting behaviour is borrowed from nature: *mimicry*, more specifically, the *cuckoo's parasitism*. In nature, many creatures in various environments follow mimicry. A behaviour that allows them to despite being harmless appear as harmful (to avoid being eaten by a predator) or to despite being harmful appear as a prey (to lure others into a trap). These creatures adapted to their environment, its dangers and opportunities to survive. A cuckoo selects another bird's nest to lay its eggs which mimic the characteristics of the eggs of the host as much as possible. The cuckoo's egg is an outlier yet by blending in the host environment, it avoids detection and fatal outcome of rejection by the host.

Similarly, considering outliers in a dataset, a data poisoning attack can ensure that these abnormal data points blend in the context by mimicking properties similar to the normal data, resulting in the misleading of the outlier detection mechanism. This is analogous to a creature to appear harmless or that is trying to avoid a predator (detection and purging mechanism).

Beyond the concealing effect that mimicry provides for the cuckoo’s egg to survive, there is a dark twist: once the cuckoo chick hatches, it may evict the host’s eggs from the nest, resulting in a contextual reversal—the cuckoo is no longer a minority. Similarly, if the mimicking process in outlier-prone regions not only strengthens but becomes dominant over the rest of the data, the autonomous outlier removal process may begin to recognize originally normal data as abnormal and purge them from the dataset.

In the following experiments only the mimicking part is covered, the contextual reversal stands as an idea to make the data poisoning attack more dangerous.

**Limitations:** This attack assumes that a density-based (DBSCAN) outlier detection mechanism is in place of the targeted ML pipeline. If there are no outliers in the dataset then there is nothing to be concealed. In this case, manual injection of outliers may take place during the attack to ensure its success.

|  _Experimental details_  |                                             Iris dataset |                                    Breast cancer dataset |
|:-------------------------|---------------------------------------------------------:|---------------------------------------------------------:|
| Number of features       |                                                        4 |                                                       30 |
| Standardization method   |                                          Min-max scaling |                                          Min-max scaling |
| Training data            |                 All observations (150 ± purge/injection) |                 All observations (569 ± purge/injection) |
| Number of fixed clusters |                                                        3 |                                                        2 |
| Visualization method     |                                                      PCA |                                                      PCA |
| Evaluation metrics       | Silhouette score, Mean within-cluster standard deviation | Silhouette score, Mean within-cluster standard deviation |

#### Iris Dataset
![image](https://github.com/user-attachments/assets/592fb001-6867-423e-b5c5-d5a1b0034c03)

| _Iris dataset_                                       |     Silhouette score | Mean within-cluster std. |
|:-----------------------------------------------------|---------------------:|-------------------------:|
| Normal clustering                                    |               0.4829 |                   0.1029 |
| Outlier removal + Normal clustering                  |               0.5062 |                   0.0982 |
| Data poisoning + Outlier removal + Normal clustering |               0.5518 |                   0.1206 |

#### Breast Cancer Dataset
![image](https://github.com/user-attachments/assets/2d27bf04-5f0b-4ea1-bde7-c4b4cfc6fc8c)

| _Breast cancer dataset_                              |     Silhouette score | Mean within-cluster std. |
|:-----------------------------------------------------|---------------------:|-------------------------:|
| Normal clustering                                    |               0.3845 |                   0.1146 |
| Outlier removal + Normal clustering                  |               0.3853 |                   0.1071 |
| Data poisoning + Outlier removal + Normal clustering |               0.3441 |                   0.1513 |


### Semi-supervised KMeans Classification

Centroids of KMeans are initialized based on a limited yet labeled sample. The centroids are labeled based on this sample. As the centroids fit the data during clustering, the label information is used to assign a label to each formulated cluster. The poisoning attack targets the sample data and flips its labels. This poisoned guide then may not result in the change of clustering but definietly will end up in misclassification. The diagrams below show the before and after attack predictions for the target variable.

**Limitations:** The poisoned model seems to be immune against label flipping of the sample data if the flip ratio is not large enough. According to the experiments, a reliable ratio could be $0.7$ for the _Iris_ and $0.6$ for the _Breast cancer_ dataset.

#### Iris Dataset

##### Initial Centroids of Sample
![image](https://github.com/user-attachments/assets/f3046e9b-7ff9-4d49-9f6e-91dd2d710544)
##### Clustering & Classification
![image](https://github.com/user-attachments/assets/15c91fe4-a651-4492-904d-6a4e82900f5f)
##### Classification Report before and after Poisoning
<img width="1186" alt="image" src="https://github.com/user-attachments/assets/073e18d7-a8f9-43d6-8d4e-b1c131e27837" />

#### Breast Cancer Dataset

##### Initial Centroids of Sample
![image](https://github.com/user-attachments/assets/3923e317-c664-44c0-93a5-90d760ba0c29)
##### Clustering & Classification
![image](https://github.com/user-attachments/assets/255c85a8-7b86-4c6c-a281-cc327ad54392)
##### Classification Report before and after Poisoning
<img width="1186" alt="image" src="https://github.com/user-attachments/assets/f57c7098-7b98-401d-ae5c-22755288dc9d" />

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

### Randomized Injection Attacks

Open the notebook in a jupyter notebbok environment and execute the cells in order.  
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

## Key Definitions and Explanations

- **Poisoning Attack** A type of adversarial attack where an attacker injects malicious data into the training set to manipulate a machine learning model's behavior, either by reducing accuracy (**availability attack**) or steering predictions in a specific direction (**integrity attack**).
- **Availability Attack** A poisoning attack aimed at degrading the overall performance of a machine learning model, making it unreliable for general classification tasks.
- **Integrity Attack** A poisoning attack that manipulates specific inputs to force the model into making incorrect predictions while maintaining overall accuracy.
- **Label Flipping Attack** A poisoning technique where an attacker intentionally flips the class labels of certain training examples to mislead the model.
- **Backdoor Attack** A type of poisoning attack where an attacker embeds hidden triggers in training data, allowing them to control model predictions when these triggers appear in test inputs.
- **Gradient Manipulation Attack** An advanced poisoning method where an attacker crafts malicious training samples that directly influence the model's gradient updates to maximize error.
- **Adversarial Training** A defense mechanism that involves training a model on adversarially modified examples to improve robustness against attacks.
- **Transferability of Adversarial Attacks** The ability of an adversarial attack (such as evasion or poisoning) to remain effective across different machine learning models, datasets, or architectures.
- **White-box Attack** An attack where the adversary has full knowledge of the model architecture, parameters, and training data, allowing for precise adversarial manipulations.
- **Black-box Attack** An attack where the adversary has no direct access to the model's parameters or structure and must infer vulnerabilities based on input-output behavior.

These definitions and explanations provide a fundamental understanding of poisoning attacks, their mechanisms, and possible defenses, highlighting the ongoing challenges in securing machine learning models.
