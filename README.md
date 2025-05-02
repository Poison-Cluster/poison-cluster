# Poison Cluster

## Overview of Different Attacks

### Semi-supervised KMeans Classification

Centroids of KMeans are initialized based on a limited yet labeled sample. The centroids are labeled based on this sample. As the centroids fit the data during clustering, the label information is used to assign a label to each formulated cluster. The poisoning attack targets the sample data and flips its labels. This posioned guide then may not result in the change of clustering but definietly will end up in misclassification. The diagrams below show the before and after attack predictions for the target variable.

![image](https://github.com/user-attachments/assets/bc4d7adc-f3f2-4dc7-848b-99195c87c79e)
<img width="604" alt="image" src="https://github.com/user-attachments/assets/2e5ca48e-cac6-4c2e-9889-e79fe22418ae" />

![image](https://github.com/user-attachments/assets/52414753-8a9e-4e60-83b9-6e94c90d045c)
<img width="604" alt="image" src="https://github.com/user-attachments/assets/2ac8f000-29e6-46d1-b85d-d4896f199ef1" />


## Usage Guide

### Random Noise

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
