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

def main():

    trainer_path = "./archive/ensemble_3035_long/EnsembleData.p"

    # Approximations of the larger data set
    n_real_data_size = 2000000
    n_predicted_lenses = 2000

    with open( trainer_path, 'rb' ) as handle:
        trainer = pickle.load(handle)
    
    print("Best Error:", trainer.error_best)
    print("Best n:", trainer.n_best)
    print("Best weight:", trainer.weight_best)
    print("Best threshold:", trainer.threshold_best)
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
    print(inds)
    print(error_te[inds])
    print(weights[inds[0]])
    print(thresholds[inds[1]])
    
    plt.contourf(weights, thresholds, error_te.T, cmap=cm.coolwarm)
    plt.title(f"Model training error as a function of weights and thresholds")
    plt.xlabel("weights")
    plt.ylabel("thresholds")
    plt.scatter(weights[inds[0]], thresholds[inds[1]], c='black')
    plt.show()


if __name__ == "__main__":
    main()