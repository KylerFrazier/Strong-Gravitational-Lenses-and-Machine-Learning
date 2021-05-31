# Changes matplotlib's backend to write to file instead of display
# import matplotlib
# matplotlib.use('Agg')

# Modules for processing, math, and graphing
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from os import path
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
import pickle


def model_performance(n,threshold,weight):

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

    sample_weight = np.ones(y_tr.shape)
    sample_weight[y_tr == 0] = weight

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

    ens_wrapper = RandomForestBT( n_estimators = n, 
                                  random_state = 0 )
    model = EnsembleBT( estimators = base_models, 
                        final_estimator = ens_wrapper,
                        threshold = threshold )
    model.fit(x_tr, y_tr, sample_weight = sample_weight)

    y_hat = model.predict(x_te)
    fp_best, tp_best = np.ravel(confusion_matrix(y_te, 
        model.predict(x_te), normalize="true")[:,1])
    print("Error:", model.error(x_te, y_te))
    print("Best False Positive:", fp_best)
    print("Best True Positive:", tp_best)
    print()
    print("Expected findings:")
    print("    Selected QSOs   =", int(n_real_data_size*fp_best))
    print("    Selected Lenses =", int(n_predicted_lenses*tp_best))

    disp = plot_confusion_matrix(model, x_te, y_te, 
        cmap=plt.cm.Blues, normalize="true", display_labels=classes) 
    disp.ax_.set_title("Normalized Confusion Matrix")
    plt.savefig("CM.pdf")
    plt.clf()

    ###########################################################################

    y_pr_tr = model.predict_proba(x_tr).T[LENS]
    plt.figure()
    plt.hist(y_pr_tr[y_tr == QSO], bins = 12, alpha = 0.5, label = "QSO")
    plt.hist(y_pr_tr[y_tr == LENS], bins = 12, alpha = 0.5, label = "Lens")
    plt.legend(loc='upper right')
    plt.title("Histogram of Soft Predict on Training Data")
    plt.xlabel("P(x=Lens)")
    plt.ylabel("Count")
    plt.savefig("hist_train.pdf")
    plt.clf()

    y_pr_te = model.predict_proba(x_te).T[LENS]
    plt.figure()
    plt.hist(y_pr_te[y_te == QSO], bins = 9, alpha = 0.5, label = "QSO")
    plt.hist(y_pr_te[y_te == LENS], bins = 9, alpha = 0.5, label = "Lens")
    plt.legend(loc='upper right')
    plt.title("Histogram of Soft Predict on Testing Data")
    plt.xlabel("P(x=Lens)")
    plt.ylabel("Count")
    plt.savefig("hist_test.pdf")
    plt.clf()

def main():

    trainer_path = "./archive/ensemble_3035_long/EnsembleData.p"

    # Approximations of the larger data set
    n_real_data_size = 2000000
    n_predicted_lenses = 2000

    with open( trainer_path, 'rb' ) as handle:
        trainer = pickle.load(handle)
    
    fp_best, tp_best = np.ravel(confusion_matrix(trainer.y_te, 
        trainer.model_best.predict(trainer.x_te), normalize="true")[:,1])
    print()
    print("Best False Positive:", fp_best)
    print("Best True Positive:", tp_best)
    print()
    print("Expected findings:")
    print("    Selected QSOs   =", int(n_real_data_size*fp_best))
    print("    Selected Lenses =", int(n_predicted_lenses*tp_best))

    print("\nAll min error:")
    error_te = trainer.data['weights_and_threshold']['error_te']
    weights = trainer.data['weights_and_threshold']['weights']
    thresholds = trainer.data['weights_and_threshold']['thresholds']

    inds = np.where(error_te == error_te.min())
    print(f"Error:\n\t{min(error_te[inds])}")
    print(f"Weights:\n\t{weights[inds[0]]}")
    print(f"Thresholds:\n\t{thresholds[inds[1]]}")

    plt.contourf(weights, thresholds, np.log(error_te.T), cmap=cm.coolwarm)
    plt.colorbar()
    plt.title(f"Log of Testing Error over Weights and Thresholds")
    plt.xlabel("Weights")
    plt.ylabel("Thresholds")
    plt.scatter(weights[inds[0]], thresholds[inds[1]], 
                s=1, marker='o', c='black', label = "Minima")
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig("weights_vs_thresh.pdf")
    plt.clf()

    from scipy import signal
    kernel = np.array( [[1,1,1,1,1],
                        [1,1,1,1,1],
                        [1,1,1,1,1],
                        [1,1,1,1,1],
                        [1,1,1,1,1]] ) 
    # Perform 2D convolution with input data and kernel 
    avg = signal.convolve2d(error_te.T, kernel, boundary='wrap', mode='same')/kernel.sum()
    
    print("\nAll min error based on local average:")
    inds = np.where(avg.T == avg.T.min())
    print(f"Error:\n\t{min(error_te[inds])}")
    print(f"Weights:\n\t{weights[inds[0]]}")
    print(f"Thresholds:\n\t{thresholds[inds[1]]}")

    plt.contourf(weights, thresholds, avg, cmap=cm.coolwarm)
    plt.colorbar()
    plt.title(f"Local Average of Testing Error over Weights and Thresholds")
    plt.xlabel("Weights")
    plt.ylabel("Thresholds")
    plt.scatter(weights[inds[0]], thresholds[inds[1]], 
                s=1, marker='o', c='black', label = "Minima")
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig("weights_vs_thresh_avg.pdf")
    plt.clf()

    
    plt.semilogx(trainer.data["n"]["ns"], trainer.data["n"]["error_te"], 
                 c="blue", base=2)
    plt.title(f"Testing Error as a function of $n$")
    plt.xlabel("$n$")
    plt.ylabel("Error")
    # plt.legend()
    plt.savefig("n_error.pdf")
    plt.clf()

    model_performance(
        n           = 111,
        threshold   = 0.85,
        weight      = 125
    )

if __name__ == "__main__":
    main()