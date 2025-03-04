# Stress Modelling

## 1 Data Preprocessing

### 1.1 Class Balancing

The WESAD dataset consists of data labeled into seven classes. There is a significant imbalance in the amount of data for each class. Hence, the data must be resampled to solve this imbalance.

Avoiding the introcution of noise is important so the best method would be to undersample all the classes but the minority class. There are therefore several options:

1. Cluster Centroids
2. Random Under Sampling
3. Near Miss

Due to the size of the dataset and limitations in computational resources, **Random Under Sampling** is the ideal solution.

### 1.2 Outlier Detection and Removal

Outliers can either be defined as local or global, in this instance it is appropriate to define local outliers since only neighbouring data in the timeseries are related.

There are many options for outlier detection, however z-score has been used successfully in other studies on the WESAD dataset.

A sliding window approach should be used to detect local outliers within related data.

Removing outliers could involve either clipping outliers to the expected boundaries or deleting them entirely. Again, it is important to avoid introducing noise so deleting outliers is most appropriate.

### 1.3 Normalisation

The data from each sensor needs to be mapped onto a common scale, ideally into the range between -1 and 1. Hence **min-max normalisation** is best.

### 1.4 Feature Generation
