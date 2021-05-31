# Modules for processing, math, and graphing
import numpy as np
from matplotlib import pyplot as plt
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
                    GradientBoostingBT )
from utils import utils

def main():

    ### Setting up data and base models ###

    output_path = "./output/ensemble/"

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

    trainer = utils.EnsembleTrainer( base_models, x_tr, y_tr, x_te, y_te, 
                                     file_path = output_path )

    ### Tuning n ###
    
    temp_weight = 100

    ns = np.unique((2.0**np.linspace(0,12,12*5+1)[1:]).astype(int))
    n_best = trainer.find_n_linear(ns, weight=temp_weight)

    ### Tuning weights and thresholds ###
    
    thresholds = np.linspace(0.75, 0.95, 2**7+1).astype(float).round(decimals=10)
    weights    = np.linspace(   0,  200, 2**7+1).astype(float).round(decimals=10)[1:]
    weight_best, threshold_best = trainer.find_weights_and_threshold_grid( 
                                          thresholds, weights )

    ### Displaying results ###

    # Best results so far: 
    #     n = 111
    #     weight = 125
    #     threshold = 0.85
    #     tp = 0.3036
    
    print("Best Error:", trainer.model_best.error(x_te, y_te))
    print("Best n:", n_best)
    print("Best weight:", weight_best)
    print("Best threshold:", threshold_best)
    fp_best, tp_best = np.ravel(confusion_matrix(y_te, 
        trainer.model_best.predict(x_te), normalize="true")[:,1])
    print()
    print("Best False Positive:", fp_best)
    print("Best True Positive:", tp_best)
    print()
    print("Expected findings:")
    print("    Selected QSOs   =", int(n_real_data_size*fp_best))
    print("    Selected Lenses =", int(n_predicted_lenses*tp_best))

    disp = plot_confusion_matrix(trainer.model_best, x_te, y_te, 
        cmap=plt.cm.Blues, normalize="true", display_labels=classes) 
    disp.ax_.set_title("Ensenmble Model || CM Normalized")
    plt.savefig(path.join(output_path, "CM.pdf"))
    plt.clf()


if __name__ == "__main__":
    main()