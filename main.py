# Changes matplotlib's backend to write to file instead of display
# import matplotlib
# matplotlib.use('Agg')

# Modules for processing, math, and graphing
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ( confusion_matrix, 
                              plot_confusion_matrix )

# Hide warnings from sklearn about convergence
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Change plot fonts and enable LaTeX
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]
})

# Modules for machine learning and other utilities
from utils import ( EnsembleBT,
                    RandomForestBT,
                    SupportVectorMachineBT,
                    GradientBoostingBT,
                    NeuralNetworkBT )
from utils import utils

def main():

    ###########################################################################

    # Approximations of the larger data set
    n_real_data_size = 2000000
    n_predicted_lenses = 2000

    # Renaming classes to be binary to keep things clean
    classes = np.array(["QSO", "Lens"])
    QSO = np.where(classes == "QSO")[0][0]
    LENS = np.where(classes == "Lens")[0][0]

    # Loading data
    x_tr, x_te, y_tr, y_te = utils.load_binary_data(
        "data/ADavailableData.csv",
        "data/ADlabel.csv",
        classes, verbose=True )

    base_models = [
        ('rf',  RandomForestBT( 
            n_estimators = 64, 
            random_state = 0 )),
        ('svm', SupportVectorMachineBT( 
            gamma = 2e-8, 
            kernel = 'rbf', 
            probability=True, 
            random_state = 0 )),
        ('gbt', GradientBoostingBT( 
            learning_rate = 0.07, 
            n_estimators = 64, 
            loss = "deviance",  
            random_state = 0 )) 
    ]
        # ('nn',  NeuralNetworkBT( 
        #     hidden_layer_sizes = (28,),
        #     max_iter=2000,
        #     learning_rate = "adaptive",
        #     learning_rate_init = 0.0001,
        #     random_state = 0 ))
    
    trainer = utils.EnsembleTrainer( base_models, x_tr, y_tr, x_te, y_te, 
                                     file_path="./output/ensemble/" )

    ###########################################################################
    
    temp_weight = 100

    ns = (2.0**np.arange(1,12+1)).astype(int)
    n_best = trainer.find_n_linear(ns, weight=temp_weight)

    ###########################################################################
    
    thresholds = np.linspace(0.7, 1.0, 31).astype(float).round(decimals=10)
    weights    = np.linspace( 50, 350, 31).astype(float).round(decimals=10)
    weight_best, threshold_best = trainer.find_weights_and_threshold_grid( 
                                          thresholds, weights )

    # Best results so far: 
    #     n = 2^7
    #     qso_weight = 1e2
    #     threshold = 0.78

    sample_weights = np.ones(y_tr.shape)
    sample_weights[y_tr == QSO] = 1e2

    print(' '*11 + '_'*(len(thresholds)*2+1))
    print("Progress: [ ", end='', flush=True)
    for threshold in thresholds:
        ens_wrapper = RandomForestBT(n_estimators = n_best, random_state = 0)
        model = EnsembleBT(estimators = base_models, final_estimator = ens_wrapper, threshold = threshold)
        model.fit(x_tr, y_tr, sample_weight=sample_weights)
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

    y_pr_tr = model_best.predict_proba(x_tr).T[LENS]
    plt.figure()
    plt.hist(y_pr_tr[y_tr == QSO], bins = 15, alpha = 0.5, label = "QSO")
    plt.hist(y_pr_tr[y_tr == LENS], bins = 15, alpha = 0.5, label = "Lens")
    plt.legend(loc='upper right')
    plt.title("Histogram of Best Model's Scores || X-Train")
    plt.xlabel("Score || Probabalistic guess of it being a Lens")
    plt.ylabel("Count")
    plt.savefig('output/ensemble/hist_train.pdf')
    plt.clf()

    y_pr_te = model_best.predict_proba(x_te).T[LENS]
    plt.figure()
    plt.hist(y_pr_te[y_te == QSO], bins = 10, alpha = 0.5, label = "QSO")
    plt.hist(y_pr_te[y_te == LENS], bins = 10, alpha = 0.5, label = "Lens")
    plt.legend(loc='upper right')
    plt.title("Histogram of Best Model's Scores || X-Test")
    plt.xlabel("Score || Probabalistic guess of it being a Lens")
    plt.ylabel("Count")
    plt.savefig('output/ensemble/hist_test.pdf')
    plt.clf()

    ###########################################################################

if __name__ == "__main__":
    main()