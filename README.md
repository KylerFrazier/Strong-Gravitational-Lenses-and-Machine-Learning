# Strong-Gravitational-Lenses-and-Machine-Learning

## Dependencies
All of the machine learning models were derived from those defined by the Scikit-learn 0.24.1 library \cite{scikit-learn} in Python 3.7.4, with additional usage of the python libraries Numpy 1.19.5, Matplotlib 3.4.1, and Pandas 0.25.1 for data manipulation. The final model used was a stacked ensemble using a random forest, SVM, and gradient boosted tree as the base models and another random forest as the meta model.

## How to Run Everything
The file `main.py` will create a hyperparameter tuner and train many models on the dataset specified by the filepath in `main.py`. Performance of the best model will be outputted at the end.<br>
<br>
The file `clasify.py` will take the pickled model tuner, which contains the best model and performance data, and classify some testing data with it. It then outputs useful plots and information.<br>
<br>
*Note:* The data used for training and testing is not included in this repo, as it is not yet public.
