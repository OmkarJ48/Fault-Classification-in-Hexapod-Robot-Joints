# Requirements

This project is written entirely in **MATLAB**. There are no Python packages,
so there is no `pip` `requirements.txt` to install — the list below is the
equivalent for a MATLAB environment.

## MATLAB

| Item | Version | Notes |
|---|---|---|
| MATLAB | R2024b (developed on) | R2021a or newer should work; the exported classifiers state `Classification Learner R2024b` |

## Required toolboxes

| Toolbox | Used for | Functions used in this repo |
|---|---|---|
| **Statistics and Machine Learning Toolbox** | clustering, classical classifiers, evaluation | `kmeans`, `dbscan`, `evalclusters`, `silhouette`, `pca`, `fitctree`, `fitcsvm`, `fitcensemble`, `cvpartition`, `crossval`, `confusionmat`, `perfcurve` |
| **Deep Learning Toolbox** | shallow and feed-forward neural networks | `patternnet`, `train`, `trainNetwork`, `trainingOptions`, `featureInputLayer`, `fullyConnectedLayer`, `reluLayer`, `softmaxLayer`, `classificationLayer` |

### Apps used
* **Classification Learner** (Statistics and Machine Learning Toolbox) — the
  `trainedandtestedclassifier*.m` files are models exported from this app.
* **Neural Net Pattern Recognition** (Deep Learning Toolbox) — the
  `*_patternnet.m` scripts follow the layout this app generates.

## Not required

* **SMOTE** — no toolbox or File Exchange download is needed. The oversampling
  routine is defined as a local `smote(...)` function at the bottom of each
  clustering script that uses it.

## Checking your installation

Run this in the MATLAB Command Window:

```matlab
v = ver;
installed = {v.Name};
needed = {'Statistics and Machine Learning Toolbox', 'Deep Learning Toolbox'};
for k = 1:numel(needed)
    if any(strcmp(installed, needed{k}))
        status = 'OK';
    else
        status = 'MISSING';
    end
    fprintf('%-45s %s\n', needed{k}, status);
end
```
