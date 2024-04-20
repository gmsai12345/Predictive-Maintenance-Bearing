import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'cwru-bearing-datasets:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F1328389%2F2211974%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240413%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240413T144842Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D7d22740a57793b9b25332117c656f940576e0c13ac69c2809f1555d0ac348939eb67f9458c6c101e465ae14d772eedad148e91f03d5327e679535f3f49a3e86a21d228e924c22043539c10d1c0bda3ee45c7daf22f6a23a34c02ff5c3b81173a39b7dd0f5a7b4ff3aaba00caeac0a2d699dc3113eb0a2f96dc300388bf446ff9d58f77bad43cb7882a3b7a70b68af690f81d48c84b821232995fc60c9ea432d7dbe5b96e41d59738edcaff6edb203440c47673e8403bf247b0c32c6c354fde57ac305e484ea93d316fd75e0b683ae5722d5ed2db4e1eb127caff0e0e0d1b24672c4adfe679070f45e2375be60b2b103d085b26b33ba57094e936e7b9de5fa5fd'

KAGGLE_INPUT_PATH='content/kaggle/input'
KAGGLE_WORKING_PATH='content/kaggle/working'
KAGGLE_SYMLINK='kaggle'
# !umount content/kaggle/input/ 2> /dev/null
shutil.rmtree('content/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)
try:
  os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
  pass
try:
  os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
  pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('content/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

"""## Fault type identification
There are 10 types of faults, linked to each bearing deffect:

- **Ball_007_1**: Ball defect (0.007 inch)
- **Ball_014_1**: Ball defect (0.014 inch)
- **Ball_021_1**: Ball defect (0.021 inch)
- **IR_007_1**: Inner race fault (0.007 inch)
- **IR_014_1**: Inner race fault (0.014 inch)
- **IR_021_1**: Inner race fault (0.021 inch)
- **Normal_1**: Normal
- **OR_007_6_1**: Outer race fault (0.007 inch, data collected from 6 O'clock position)
- **OR_014_6_1**: Outer race fault (0.014 inch, 6 O'clock)
- **OR_021_6_1**: Outer race fault (0.021 inch, 6 O'clock)

## Get the data
The file we will read is the result of preprocessing the raw data files (folder `/kaggle/input/cwru-bearing-datasets/raw/`).

Time series segments contains 2048 points each. Given that the sampling frequency is 48kHz each time serie covers 0.04 seconds.
"""

data_time = pd.read_csv("content/kaggle/input/cwru-bearing-datasets/feature_time_48k_2048_load_1.csv")
data_time

"""## Split into train and test datasets"""

train_data, test_data = train_test_split(data_time, test_size = 750, stratify = data_time['fault'], random_state = 1234)
test_data['fault'].value_counts()

"""## Scale features in train set"""

# Scale each column to have zero mean and standard deviation equal to 1
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data.iloc[:,:-1])
pd.DataFrame(train_data_scaled).describe()

test_data_scaled = (test_data.iloc[:,:-1].values - scaler.mean_)/np.sqrt(scaler.var_)
pd.DataFrame(test_data_scaled).describe()

"""## Train a model using Support Vector Classifier
Call the SVC() model from sklearn and fit the model to the training data.
"""

from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(train_data_scaled, train_data['fault'])

"""## Model Evaluation
Now get predictions from the model and create a confusion matrix and a classification report.
"""

train_predictions = svc_model.predict(train_data_scaled)
test_predictions = svc_model.predict(test_data_scaled)

from sklearn.metrics import classification_report,confusion_matrix

"""Plot confusion matrixes."""

train_confu_matrix = confusion_matrix(train_data['fault'], train_predictions)
test_confu_matrix = confusion_matrix(test_data['fault'], test_predictions)

fault_type = data_time.fault.unique()

plt.figure(1,figsize=(18,8))

plt.subplot(121)
sns.heatmap(train_confu_matrix, annot= True,fmt = "d",
xticklabels=fault_type, yticklabels=fault_type, cmap = "Blues", cbar = False)
plt.title('Training Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.subplot(122)

plt.subplot(122)
sns.heatmap(test_confu_matrix, annot = True,
xticklabels=fault_type, yticklabels=fault_type, cmap = "Blues", cbar = False)
plt.title('Test Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.show()

# Classification report (test set)
class_report = classification_report(y_pred = test_predictions, y_true = test_data['fault'])
print(class_report)

"""- **recall**    =    para cada fallo, proporción de los correctamente identificados sobre el total de los reales = `TP / (TP + sum(FN))`
- **precision** = para cada fallo, proporción de los correctamente identificados sobre el total en la predicción = `TP / (TP + sum(FP))`

Refer to [Understanding Data Science Classification Metrics in Scikit-Learn in Python](https://towardsdatascience.com/understanding-data-science-classification-metrics-in-scikit-learn-in-python-3bc336865019) for the explanation of these metrics

## Tuning hyperparameters for model optimization

We will check a grid of parameters to find the best one. For each parameter combination, 10 fold cross-validation is performed.
- Understand what [10 fold cross-validation](https://machinelearningmastery.com/k-fold-cross-validation/) is
"""

parameters = {"C":[1, 10, 45, 47,49, 50, 51, 55, 100, 300, 500],
             'gamma':[0.01, 0.05, 0.1, 0.5, 1, 5],
             'kernel':["rbf"]}

# Define the Grid Search optimization analysis
tuned_svm_clf = GridSearchCV(SVC(),parameters,n_jobs = -1, cv= 10)
tuned_svm_clf

# Train the move for the every pair of hyperparameters,
#   and determine the best combination
tuned_svm_clf.fit(train_data_scaled, train_data['fault'])

# Hyperparameter of the best model
tuned_svm_clf.best_params_

#Let's select the best model and provide results on them
best_clf = tuned_svm_clf.best_estimator_
best_clf

"""## Best model evaluation"""

# Compute the predictions
train_predictions_best = best_clf.predict(train_data_scaled)
test_predictions_best = best_clf.predict(test_data_scaled)

# Compute confusion matrix for training and test datasets
train_confu_matrix_best = confusion_matrix(train_data['fault'], train_predictions_best)
test_confu_matrix_best = confusion_matrix(test_data['fault'], test_predictions_best)

plt.figure(1,figsize=(18,8))

plt.subplot(121)
sns.heatmap(train_confu_matrix_best, annot= True,fmt = "d",
xticklabels=fault_type, yticklabels=fault_type, cmap = "Blues", cbar = False)
plt.title('Training Confusion Matrix (best model)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.subplot(122)

plt.subplot(122)
sns.heatmap(test_confu_matrix_best, annot = True,
xticklabels=fault_type, yticklabels=fault_type, cmap = "Blues", cbar = False)
plt.title('Test Confusion Matrix (best model)')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.show()

"""### Compare with non optimized versions"""

# Classification report (test set)
class_report_best = classification_report(y_pred = test_predictions_best, y_true = test_data['fault'])
print(class_report_best)

# Remember the metrics for the non-optimized model
print(class_report)

"""## Multinomial logistic regression
This is the alternative model for comparing with SVC performance
"""

# Logistic regression classifier
from sklearn.linear_model import LogisticRegression

# Setup the model
logis_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

# Train the model
logis_model.fit(train_data_scaled, train_data['fault'])

# Compute the predictions
test_predictions_lr = logis_model.predict(test_data_scaled)

# Compute confusion matrix
test_confu_matrix_lr = confusion_matrix(test_data['fault'], test_predictions_lr)

# Classification report
class_report_lr = classification_report(y_pred = test_predictions_lr, y_true = test_data['fault'])
print(class_report_lr)

# Compute the predictions
train_predictions_logis = logis_model.predict(train_data_scaled)
test_predictions_logis = logis_model.predict(test_data_scaled)

# Classification report (test set)
class_report_logis = classification_report(y_pred = test_predictions_logis, y_true = test_data['fault'])
print(class_report_logis)

plt.figure(1,figsize=(8,6))

sns.heatmap(test_confu_matrix_lr, annot = True,
xticklabels=fault_type, yticklabels=fault_type, cmap = "Blues", cbar = False)
plt.title('Test Confusion Matrix (logistic regression)')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.show()