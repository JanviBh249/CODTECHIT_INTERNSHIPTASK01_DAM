# CODTECHIT_INTERNSHIPTASK01_DAM
This repository contains code and resources of my Task 1.

## Project Report

### Project Name: Exploratory Data Analysis (EDA) on the Palmer Penguins Dataset

### Overview

This project involves conducting an Exploratory Data Analysis (EDA) on the Palmer Penguins dataset. The task is to explore the dataset's characteristics, distributions, correlations, and outliers using Python libraries such as pandas, numpy, matplotlib, and seaborn. This analysis is performed as an internship task (Task One) provided by CodtechIT Solutions.

### Project Details

- **Conducted By**: Janvi Deepak Bhanushali
- **Platform**: Jupyter Notebook
- **Language Used**: Python
- **Libraries Used**: pandas, numpy, matplotlib, seaborn
- **Organization**: CodtechIT Solutions
- **Internship Task**: Task One

### Introduction

The Palmer Penguins dataset contains measurements of various physical attributes of penguins from three different species observed in the Palmer Archipelago, Antarctica. This analysis aims to explore the dataset, identify patterns, understand relationships between features, and uncover meaningful insights.

### Dataset Overview

The dataset includes the following features:
- `species`: Penguin species (Adelie, Chinstrap, Gentoo)
- `island`: Island in the Palmer Archipelago where the penguin was observed (Torgersen, Biscoe, Dream)
- `bill_length_mm`: Length of the penguin's bill (mm)
- `bill_depth_mm`: Depth of the penguin's bill (mm)
- `flipper_length_mm`: Length of the penguin's flipper (mm)
- `body_mass_g`: Body mass of the penguin (g)
- `sex`: Sex of the penguin (male, female)

### Handling Missing Values

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Palmer Penguins dataset
df = sns.load_dataset('penguins')

# Display missing values
print(df.isnull().sum())

# Drop rows with missing values for simplicity
df_clean = df.dropna()
print(df_clean.isnull().sum())
```

### Basic Statistics by Species

```python
# Group by species and calculate summary statistics
species_summary = df_clean.groupby('species').describe()
print(species_summary)
```
**Output:**

![image](https://github.com/JanviBh249/CODTECHIT_INTERNSHIPTASK01_DAM/assets/171580805/91dd5dda-5f5d-470f-b3b4-6f9d1e4ab1e3)


### Distribution Analysis

#### Bill Length Distribution

```python
plt.figure(figsize=(10, 6))
sns.histplot(df['bill_length_mm'].dropna(), kde=True, bins=30)
plt.title('Bill Length Distribution')
plt.xlabel('Bill Length (mm)')
plt.ylabel('Frequency')
plt.show()
```
**Output:**
![image](https://github.com/JanviBh249/CODTECHIT_INTERNSHIPTASK01_DAM/assets/171580805/7127a0ee-e254-4095-99ea-5108a31cee3f)

#### Body Mass Distribution

```python
plt.figure(figsize=(10, 6))
sns.histplot(df['body_mass_g'].dropna(), kde=True, bins=30)
plt.title('Body Mass Distribution')
plt.xlabel('Body Mass (g)')
plt.ylabel('Frequency')
plt.show()
```
**Output:**
![image](https://github.com/JanviBh249/CODTECHIT_INTERNSHIPTASK01_DAM/assets/171580805/3fbde90f-ec7e-46e6-b0a8-9451590fba47)

### Categorical Feature Analysis

#### Species Count

```python
plt.figure(figsize=(10, 6))
sns.countplot(x='species', data=df_clean)
plt.title('Species Count')
plt.xlabel('Species')
plt.ylabel('Count')
plt.show()
```
**Output:**
![image](https://github.com/JanviBh249/CODTECHIT_INTERNSHIPTASK01_DAM/assets/171580805/63fe367c-91fc-4404-a7a0-ccdf8b960a79)

#### Species by Island

```python
plt.figure(figsize=(10, 6))
sns.countplot(x='island', hue='species', data=df_clean)
plt.title('Species Distribution by Island')
plt.xlabel('Island')
plt.ylabel('Count')
plt.legend(title='Species')
plt.show()
```
**Output:**
![image](https://github.com/JanviBh249/CODTECHIT_INTERNSHIPTASK01_DAM/assets/171580805/ddccd0de-723a-43c5-8852-3ec7109b35f1)


### Correlation Analysis

#### Correlation Heatmap

```python
plt.figure(figsize=(12, 8))
corr_matrix = df_clean.drop(columns=['species', 'island', 'sex']).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
```
**Output:**
![image](https://github.com/JanviBh249/CODTECHIT_INTERNSHIPTASK01_DAM/assets/171580805/a334c844-1ddf-4e73-b8b6-9b8f25261bfd)


### Relationships between Features

#### Bill Length vs Bill Depth

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(x='bill_length_mm', y='bill_depth_mm', hue='species', data=df_clean, palette='Set2', s=100)
plt.title('Bill Length vs Bill Depth by Species')
plt.xlabel('Bill Length (mm)')
plt.ylabel('Bill Depth (mm)')
plt.legend(title='Species')
plt.show()
```
**Output:**
![image](https://github.com/JanviBh249/CODTECHIT_INTERNSHIPTASK01_DAM/assets/171580805/f3204f4e-9d1f-416a-9da5-79f4efd8305f)

#### Flipper Length Distribution by Species

```python
plt.figure(figsize=(10, 6))
sns.histplot(data=df_clean, x='flipper_length_mm', hue='species', multiple='stack', palette='Set2', bins=30)
plt.title('Flipper Length Distribution by Species')
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Frequency')
plt.show()
```
**Output:**
![image](https://github.com/JanviBh249/CODTECHIT_INTERNSHIPTASK01_DAM/assets/171580805/6e658205-8195-4cb3-a0ee-fa14d83f4564)

### Bill Length vs Flipper Length
``` python
plt.figure(figsize=(10, 6))
sns.scatterplot(x='bill_length_mm', y='flipper_length_mm', hue='species', style='sex', data=df_clean, palette='Set2', s=100)
plt.title('Bill Length vs Flipper Length by Species and Sex')
plt.xlabel('Bill Length (mm)')
plt.ylabel('Flipper Length (mm)')
plt.legend(title='Species & Sex')
plt.show()
 ```
**Output:**
![image](https://github.com/JanviBh249/CODTECHIT_INTERNSHIPTASK01_DAM/assets/171580805/ad7d43af-b5fa-4f76-8f41-94c753b9d739)

### Advanced Visualization Techniques

#### Pairplot with Hue for Species and Sex

```python
sns.pairplot(df_clean, hue='species', palette='Set2', markers=["o", "s", "D"], diag_kind="kde", height=2.5)
plt.show()
```
![image](https://github.com/JanviBh249/CODTECHIT_INTERNSHIPTASK01_DAM/assets/171580805/00a9a5ff-985a-4cf9-8790-aaaafefec775)

#### Violin Plots for Body Mass by Species and Sex

```python
plt.figure(figsize=(10, 6))
sns.violinplot(x='species', y='body_mass_g', hue='sex', data=df_clean, split=True, palette='Set2')
plt.title('Body Mass Distribution by Species and Sex')
plt.xlabel('Species')
plt.ylabel('Body Mass (g)')
plt.show()
```
**Output:**
![image](https://github.com/JanviBh249/CODTECHIT_INTERNSHIPTASK01_DAM/assets/171580805/50b0b7e3-9d95-4db4-91e3-1e8f0f4ad1b8)

### KDE Plot for Flipper Length
```python
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_clean, x='flipper_length_mm', hue='species', fill=True, common_norm=False, palette='Set2', alpha=0.5)
plt.title('KDE Plot for Flipper Length by Species')
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Density')
plt.show()
```
**Output:**
![image](https://github.com/JanviBh249/CODTECHIT_INTERNSHIPTASK01_DAM/assets/171580805/b3b2e084-1dba-4e0c-8f41-9880f910dfbb)

### Pairwise Relationships with Regression Lines
```python
sns.pairplot(df_clean, hue='species', palette='Set2', kind='reg', plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.6}})
plt.show()
```
**Output:**
![image](https://github.com/JanviBh249/CODTECHIT_INTERNSHIPTASK01_DAM/assets/171580805/fbc011e1-2633-4fd3-a3ea-10c3f7478bea)

### Additional Insights from Subgroup Analyses

#### Body Mass by Sex within Each Species

```python
plt.figure(figsize=(10, 6))
sns.boxplot(x='species', y='body_mass_g', hue='sex', data=df_clean, palette='Set2')
plt.title('Body Mass by Sex within Each Species')
plt.xlabel('Species')
plt.ylabel('Body Mass (g)')
plt.legend(title='Sex')
plt.show()
```
**Output:**
![image](https://github.com/JanviBh249/CODTECHIT_INTERNSHIPTASK01_DAM/assets/171580805/bea42601-bf1a-4cfc-a656-44c9304470b3)

### PCA (Principal Component Analysis)

```python
from sklearn.decomposition import PCA

# Drop categorical columns and handle missing values
df_pca = df_clean.drop(columns=['species', 'island', 'sex'])

# Standardize the data
df_pca = (df_pca - df_pca.mean()) / df_pca.std()

# Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_pca)
df_pca_result = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
df_pca_result['species'] = df_clean['species'].values

# Plot PCA results
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='species', data=df_pca_result, palette='Set2', s=100)
plt.title('PCA of Palmer Penguins Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Species')
plt.show()
```
**Output:**
![Screenshot 2024-06-10 104213](https://github.com/JanviBh249/CODTECHIT_INTERNSHIPTASK01_DAM/assets/171580805/584e3f17-4717-46d2-8dd1-72df32408515)


### Detailed Insights

**Missing Values**:
   - The dataset has some missing values in features like 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', and 'sex'.

**Basic Statistics by Species**:
   - Gentoo penguins have the longest flippers and highest body mass.
   - Adelie penguins have the shortest flippers and lowest body mass.

**Distributions**:
   - Bill length and body mass distributions show normal patterns with some outliers.

**Species Distribution by Island**:
   - Adelie penguins are found on all three islands (Torgersen, Biscoe, Dream), Chinstrap penguins only on Dream, and Gentoo penguins only on Biscoe.

**Correlation Analysis**:
   - Strong positive correlation between flipper length and body mass.
   - Negative correlation between bill depth and bill length.

**Bill Length vs. Bill Depth**:
   - Clear distinction between species based on bill measurements.

**Flipper Length Distribution**:
   - Gentoo penguins have significantly longer flippers compared to the other species.

**Bill Length vs. Flipper Length:**
  - There is a clear distinction between species based on bill length and flipper length.
  - Gentoo penguins generally have longer flippers and bills, while Adelie and Chinstrap penguins show more overlap.


**Pairplot**:
   - Distinct clustering patterns for different species.

**Violin Plots for Body Mass**:
   - Male penguins have higher body mass than females across all species, with the most significant difference in Gentoo penguins.
     
**KDE Plot for Flipper Length:**
  - The KDE plot shows that Gentoo penguins have the highest flipper length density peak, indicating that most of them have longer flippers.
  - Adelie and Chinstrap penguins have overlapping density peaks but are generally lower than Gentoo.

**Pairwise Relationships with Regression Lines:**
  - The pairplot with regression lines shows linear relationships between features, with Gentoo penguins often having distinct slopes compared to the other species.
  - This indicates different growth patterns or physical characteristics between species.
       
**Body Mass by Sex within Each Species**:
  - Male penguins tend to have higher body mass compared to female penguins within each species.
  - The difference in body mass between sexes is most pronounced in Gentoo penguins.

**PCA**:
   - Clear separation between the three penguin species in the PCA plot, with PC1 distinguishing Gentoo from the other two species and PC2 separating Adelie from Chinstrap penguins.
  * PC1 captures most of the variance and distinguishes Gentoo penguins from Adelie and Chinstrap penguins.
  * PC2 helps to further separate Adelie from Chinstrap penguins, although there is some overlap

### Conclusion

The exploratory data analysis of the Palmer Penguins dataset reveals distinct patterns and relationships between various physical measurements of the penguins. These insights can be used as a foundation for further analysis, feature engineering, or predictive modeling tasks. The clear separations observed between species and the relationships between features highlight the importance of comprehensive EDA in understanding and leveraging the data for more advanced analyses.
