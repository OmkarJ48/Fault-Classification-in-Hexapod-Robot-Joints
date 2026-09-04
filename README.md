# Fault Classification in Hexapod Robot Joints

ML pipeline for fault detection and diagnosis in hexapod robot joints — K-Means/DBSCAN
clustering, SVM / decision tree / bagged-ensemble / KNN / neural-network classification,
SMOTE for class imbalance, evaluated with cross-validation, F1 and ROC AUC.

MSc coursework (**Machine Learning — ME7026**), written in MATLAB (R2024b).

---

## Repository layout

```
.
├── docs/                              Coursework reports and appendix
│   ├── K2441768_OMKAR ANANT JOSHI_Fault Classification in Robot Joints_Individual Conclusion Submission.docx
│   ├── APPENDIX.docx
│   ├── Machine Learning-ME7026 Coursework.docx
│   └── Machine Learning-ME7026 Coursework.pdf
│
├── original_dataset/                  Raw, unmodified source data
│   └── Hexapod_One_Joint_data_set.csv
│
├── matlab_codes/
│   ├── classification/                Supervised classifiers + metrics
│   ├── clustering/                    K-Means, DBSCAN, PCA, normalisation
│   └── neural_network/                patternnet / trainNetwork models
│
├── REQUIREMENTS.md                    MATLAB version and required toolboxes
├── .gitignore
└── README.md
```

---

## The data

`original_dataset/Hexapod_One_Joint_data_set.csv` is the single source of truth —
sensor readings for one hexapod joint with a `Label` column.

Every other CSV in `matlab_codes/**` is **derived** from it, and is committed so that
each script can be run on its own without regenerating the whole pipeline:

| Derived file | What it is | Written by |
|---|---|---|
| `binary_classification.csv`, `binary_dataset.csv` | full data relabelled fault (1) vs. no-fault (0); identical content, two names | `Part1ML.m`, `Four_datasets.m` |
| `binary_classification_2.csv` | 1 % random subset of the binary data | `Part1ML.m`, `Classificationmatlabcodefinal.m` |
| `binary_dataset_1%.csv` | a second, independent 1 % random sample of the binary data | `Four_datasets.m` |
| `multiclass_dataset.csv` | full data with `Label` as a categorical fault type | `Part1ML.m`, `Four_datasets.m` |
| `multiclass_classification.csv` | 1 % random subset of the multiclass data | `Part1ML.m`, `Classificationmatlabcodefinal.m` |
| `multiclass_dataset_1%.csv` | a second, independent 1 % random sample of the multiclass data | `Four_datasets.m` |
| `equal_balanced_dataset.csv` | binary data downsampled to equal-sized classes | `Binarydata_patternnet_equal_class.m` |
| `normalized_binary_data.csv`, `normalized_multiclass.csv` | z-score normalised full datasets | `Part1ML.m` |
| `normalized_binary_data1.csv`, `normalized_multiclass1.csv` | z-score normalised 1 % subsets | `Part1ML.m` |

The 1 % samples are drawn with `randperm`, so they are random rather than stratified —
re-running the generator scripts will not reproduce the committed CSVs exactly.

---

## Running the code

The scripts load their data with plain `readtable('<name>.csv', ...)`, i.e. from the
current folder. Add the repo to the MATLAB path once, then `cd` into whichever
folder you want to run:

```matlab
addpath(genpath('.'))          % run from the repository root
cd matlab_codes/clustering     % or classification / neural_network
```

Scripts that read `Hexapod_One_Joint_data_set.csv` need the raw dataset visible too:

```matlab
addpath(fullfile(pwd, 'original_dataset'))   % from the repository root
```

### 1. Data preparation

| Script | Purpose |
|---|---|
| `clustering/Part1ML.m` | loads the raw dataset, builds the binary/multiclass CSVs and the normalised variants |
| `neural_network/Four_datasets.m` | builds the four dataset variants (full + 1 % samples) used by the networks |

### 2. Clustering — `matlab_codes/clustering/`

| Script | Purpose |
|---|---|
| `Part2ML.m`, `part2MLNORM.m` | K-Means driver on raw and normalised features, optimal *K* via `evalclusters` |
| `KMEANSBIN.m`, `KMEANS1BIN.m` | K-Means on the binary data (full / subset), with SMOTE balancing |
| `KMeansMulti.m`, `KMeansMulti1.m` | K-Means on the multiclass data |
| `DBSCANforBIN.m`, `DBSCAforBIN1.m` | DBSCAN on the binary data |
| `DBSCANFORmulticlass.m`, `dbscanMulti1.m` | DBSCAN on the multiclass data |

Evaluated with silhouette scores and the Calinski–Harabasz (VRC) criterion.

### 3. Classification — `matlab_codes/classification/`

| Script | Purpose |
|---|---|
| `Classificationmatlabcodefinal.m` | main driver: builds the dataset variants, then trains a linear SVM and a bagged ensemble; opens Classification Learner |
| `trainedandtestedclassifierbinarydata.m` | decision-tree classifier exported from Classification Learner (binary, full) |
| `trainedandtestedclassifierbinarysubset.m` | same, on the balanced binary subset |
| `trainedandtestedclassifiermulticlassdata.m` | same, multiclass |
| `F1ScoreBinaryClass*.m` | per-class precision / recall / F1 for the binary models (KNN, linear SVM, ensemble, trees) |
| `F1ScoreMultiClass*.m` | the same metrics for the multiclass models |
| `ROCAUC.m` | ROC curves and AUC across all four dataset variants |

Each exported classifier returns a `trainedClassifier` struct plus a
`validationAccuracy` from 10-fold cross-validation:

```matlab
data = readtable('binary_classification.csv', 'VariableNamingRule', 'preserve');
[clf, acc] = trainClassifier(data);
yfit = clf.predictFcn(data);
```

### 4. Neural networks — `matlab_codes/neural_network/`

| Script | Purpose |
|---|---|
| `Binarydata_patternet.m` | `patternnet` on the full binary dataset |
| `Binarydata_1percent_patternnet.m` | `patternnet` on the 1 % binary sample |
| `Binarydata_patternnet_equal_class.m` | `patternnet` on the class-balanced dataset |
| `Binarydata_trainNetwork.m` | `trainNetwork` model: featureInput → FC(10) → ReLU → FC(5) → ReLU → FC(2) → softmax, Adam optimiser |
| `Multiclass_dataset_patternnet.m` | `patternnet` on the full multiclass dataset |
| `Multiclass_dataset_1percent_patternnet.m` | `patternnet` on the 1 % multiclass sample |

---

## Requirements

MATLAB R2024b with the **Statistics and Machine Learning Toolbox** and the
**Deep Learning Toolbox**. See [REQUIREMENTS.md](REQUIREMENTS.md) for the full
list and an installation check. No Python dependencies — SMOTE is implemented
as a local function inside the clustering scripts, so nothing extra to download.

## Author

Omkar Anant Joshi — K2441768
