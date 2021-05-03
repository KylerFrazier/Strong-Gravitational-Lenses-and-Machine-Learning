from __future__ import print_function

# Changes matplotlib's backend to write to file instead of display
import matplotlib
matplotlib.use('Agg')

# Modules for processing, math, and graphing
import numpy as np
from matplotlib import pyplot as plt
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import ( confusion_matrix, 
                              plot_confusion_matrix )

# Hide warnings from sklearn about convergence
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Change plot fonts and enable LaTeX
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"]
# })

# Modules for machine learning
from utils import ( EnsembleBT,
                    RandomForestBT,
                    SupportVectorMachineBT,
                    GradientBoostingBT,
                    NeuralNetworkBT )

def main():

    ###########################################################################

    n_real_data_size = 2000000
    n_predicted_lenses = 2000

    data_features = read_csv("data/ADavailableData.csv", header=0, quotechar='"', skipinitialspace=True).to_numpy()
    data_classes = read_csv("data/ADlabel.csv", header=0, quotechar='"', skipinitialspace=True).to_numpy()

    # The following line assumes that the features and classes are ordered properly
    x, y = data_features[:,1:], np.ravel(data_classes[:,1:])

    # Renaming classes to be binary and checking that they match
    classes = np.array(["QSO", "Lens"])
    QSO = np.where(classes == "QSO")[0][0]
    Lens = np.where(classes == "Lens")[0][0]
    num_entries_per_class = 0
    for i in range(len(classes)):
        y[y == classes[i]] = i
        num_entries_per_class += len(y[y == i])
    y = y.astype(int)
    assert num_entries_per_class == len(y) , f"The classes should be in {classes}"

    x_tr, x_te, y_tr, y_te = train_test_split(x, y, random_state=0, stratify=y)
    print("Classes:", classes)
    print("x_tr.shape =", x_tr.shape)
    print("x_te.shape =", x_te.shape)
    print("y_tr.shape =", y_tr.shape)
    print("y_te.shape =", y_te.shape)
    
    estimators = [
        ('rf',  RandomForestBT( n_estimators = 64, random_state = 0 )),
        ('svm', SupportVectorMachineBT( gamma = 2e-8, kernel = 'rbf', 
                                        probability=True, 
                                        random_state = 0 )),
        ('gbt', GradientBoostingBT( learning_rate = 0.07, 
                                            n_estimators = 64, 
                                            loss = "deviance",  
                                            random_state = 0 )), 
        ('nn',  NeuralNetworkBT( hidden_layer_sizes = (28,),
                               max_iter=2000,
                               learning_rate = "adaptive",
                               learning_rate_init = 0.0001, random_state = 0 )) ]

    ###########################################################################

    ns = (2.0**np.arange(1,8)).astype(int)
    # ns = (2.0**np.arange(6,8)).astype(int)
    models = []
    error_tr = []
    error_te = []

    print(' '*11 + '_'*(len(ns)*2+1))
    print("Progress: [ ", end='', flush=True)
    for n in ns:
        ens_wrapper = RandomForestBT(n_estimators = n, random_state = 0)
        model = EnsembleBT(estimators = estimators, final_estimator = ens_wrapper)
        model.fit(x_tr, y_tr)
        error_tr.append(model.error(x_tr, y_tr))
        error_te.append(model.error(x_tr, y_tr))
        models.append(model)
        print('> ', end='', flush=True)
    print(']')

    model_best = models[np.argmin(error_te)]
    n_best = ns[np.argmin(error_te)]
    print(f"Best n = 2^{int(np.log2(n_best))}")

    plt.semilogx(ns, error_tr, c="blue", base=2, label="Training Error")
    plt.semilogx(ns, error_te, c="red",  base=2, label="Testing Error" )
    plt.title("Model error as a function of $n$")
    plt.xlabel("$n$")
    plt.ylabel("Error")
    plt.legend()
    plt.savefig('output/ensemble/n.pdf')
    plt.clf()

    ###########################################################################
    
    thresholds = np.linspace(0.8, 1.0, 21).astype(float).round(decimals=10)
    models = []
    error_tr = []
    error_te = []
    prec = []
    reca = []

    print(' '*11 + '_'*(len(thresholds)*2+1))
    print("Progress: [ ", end='', flush=True)
    for threshold in thresholds:
        ens_wrapper = RandomForestBT(n_estimators = n_best, random_state = 0, threshold = threshold)
        model = EnsembleBT(estimators = estimators, final_estimator = ens_wrapper)
        model.fit(x_tr, y_tr)
        error_tr.append(model.error(x_tr, y_tr))
        error_te.append(model.error(x_te, y_te))
        prec.append(model.precision(x_te, y_te))
        reca.append(model.recall(x_te, y_te))
        models.append(model)
        print('> ', end='', flush=True)
    print(']')

    model_best = models[np.argmin(error_te)]
    threshold_best = thresholds[np.argmin(error_te)]

    plt.plot(thresholds, error_tr, c="blue", label="Training Error")
    plt.plot(thresholds, error_te, c="red", label="Testing Error")
    plt.title("Model error as a function of threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Error")
    plt.legend()
    plt.savefig('output/ensemble/threshold.pdf')
    plt.clf()
    
    ###########################################################################
    
    plt.plot(thresholds, 0.1*np.array(prec), c="green")
    plt.plot(thresholds, reca, c="orange")
    plt.title("Precision vs Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.savefig('output/ensemble/precision_vs_recall.pdf')
    plt.clf()
    
    ###########################################################################
    
    print("Best Error:", model_best.error(x_te, y_te))
    print("Best n:", n_best)
    print("Best threshold:", threshold_best)
    fp_best, tp_best = np.ravel(confusion_matrix(y_te, model_best.predict(x_te), normalize="true")[:,1])
    print()
    print("Best False Positive:", fp_best)
    print("Best True Positive:", tp_best)
    print()
    print("Expected findings:")
    print("    Selected QSOs   =", int(n_real_data_size*fp_best))
    print("    Selected Lenses =", int(n_predicted_lenses*tp_best))


    disp = plot_confusion_matrix(model_best, x_te, y_te, cmap=plt.cm.Blues, normalize="true", display_labels=classes) 
    disp.ax_.set_title("Ensenmble Model || CM Normalized")
    plt.savefig('output/ensemble/CM.pdf')
    plt.clf()

    ###########################################################################

    y_pr_tr = model_best.predict_proba(x_tr).T[Lens]
    plt.figure()
    plt.hist(y_pr_tr[y_tr == QSO], bins = 15, alpha = 0.5, label = "QSO")
    plt.hist(y_pr_tr[y_tr == Lens], bins = 15, alpha = 0.5, label = "Lens")
    plt.legend(loc='upper right')
    plt.title("Histogram of Best Model's Scores || X-Train")
    plt.xlabel("Score || Probabalistic guess of it being a Lens")
    plt.ylabel("Count")
    plt.savefig('output/ensemble/hist_train.pdf')
    plt.clf()

    y_pr_te = model_best.predict_proba(x_te).T[Lens]
    plt.figure()
    plt.hist(y_pr_te[y_te == QSO], bins = 10, alpha = 0.5, label = "QSO")
    plt.hist(y_pr_te[y_te == Lens], bins = 10, alpha = 0.5, label = "Lens")
    plt.legend(loc='upper right')
    plt.title("Histogram of Best Model's Scores || X-Test")
    plt.xlabel("Score || Probabalistic guess of it being a Lens")
    plt.ylabel("Count")
    plt.savefig('output/ensemble/hist_test.pdf')
    plt.clf()

    ###########################################################################

if __name__ == "__main__":
    main()