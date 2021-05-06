import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from pandas import read_csv
from sklearn.model_selection import train_test_split
from os import path
from utils import EnsembleBT, RandomForestBT


def load_binary_data(feature_path, class_path, classes=[0,1], verbose=False):
    data_features = read_csv( feature_path, header=0, quotechar='"', 
                              skipinitialspace=True ).to_numpy()
    data_classes = read_csv( class_path, header=0, quotechar='"', 
                             skipinitialspace=True ).to_numpy()

    # The below assumes that the features and classes are ordered properly
    x, y = data_features[:,1:], np.ravel(data_classes[:,1:])
    
    num_entries_per_class = 0
    for i in range(len(classes)):
        y[y == classes[i]] = i
        num_entries_per_class += len(y[y == i])
    y = y.astype(int)
    assert num_entries_per_class == len(y) , f"The classes should be in {classes}"

    x_tr, x_te, y_tr, y_te = train_test_split(x, y, random_state=0, stratify=y)

    if verbose:
        print("Classes:", classes)
        print("x_tr.shape =", x_tr.shape)
        print("x_te.shape =", x_te.shape)
        print("y_tr.shape =", y_tr.shape)
        print("y_te.shape =", y_te.shape)
    
    return x_tr, x_te, y_tr, y_te

def plot_errors(x, x_label, error_tr, error_te, output_path):
    plt.semilogx(x, error_tr, c="blue", base=2, label="Training Error")
    plt.semilogx(x, error_te, c="red",  base=2, label="Testing Error" )
    plt.title(f"Model error as a function of {x_label}")
    plt.xlabel(x_label)
    plt.ylabel("Error")
    plt.legend()
    plt.savefig(output_path)
    plt.clf()

def plot_errors_3D(x, y, x_label, y_label, error_tr, error_te, path1, path2):
    plt.contourf(x, y, error_tr.T, cmap=cm.coolwarm)
    plt.title(f"Model training error as a function of {x_label} and {y_label}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(path1)
    plt.clf()

    plt.contourf(x, y, error_te.T, cmap=cm.coolwarm)
    plt.title(f"Model testing error as a function of {x_label} and {y_label}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(path2)
    plt.clf()

class ProgressBar(object):

    def __init__(self, length):
        self.length = length
        self.tics = 0
        print(' '*11 + '_'*(length*2+1))
        print("Progress: [ ", end='', flush=True)
    
    def incriment(self):
        if self.tics < self.length:
            print('> ', end='', flush=True)
            self.tics += 1
            if self.tics >= self.length: 
                print(']')

class EnsembleTrainer(object):

    def __init__( self, base_models, x_tr, y_tr, x_te, y_te, 
                  file_path=".", file_name="EnsembleData.p",
                  model_name="Model.p" ):
        
        assert path.exists(file_path), f"File path {file_path} does not exist."
        
        self.base_models = base_models
        self.x_tr, self.y_tr = x_tr, y_tr
        self.x_te, self.y_te = x_te, y_te

        self.data = {}
        self.file_path = file_path
        self.file_name = file_name
        self.model_name = model_name

        self.model_best = None
        self.error_best = np.inf

        self.n_best = None
        self.weight_best = None
        self.threshold_best = None
    
    def save(self, file_path=None, file_name=None):
        if file_path == None or file_name == None:
            file_path = self.file_path
            file_name = self.file_name
        with open(path.join(file_path, file_name), 'wb') as handle:
            pickle.dump(self, handle)
    
    def save_model(self, file_path=None, model_name=None):
        if file_path == None or model_name == None:
            file_path = self.file_path
            model_name = self.model_name
        with open(path.join(file_path, model_name), 'wb') as handle:
            pickle.dump(self.model_best, handle)

    def find_n_linear( self, ns, threshold=None, weight=None,
                       verbose=True, save=True, plot=True ):
        models = np.empty((len(ns)), dtype=object)
        error_tr = np.zeros((len(ns)))
        error_te = np.zeros((len(ns)))

        if self.threshold_best != None:
            threshold = self.threshold_best
        elif threshold == None:
            threshold = 0.80
        if self.weight_best != None:
            weight = self.weight_best
        elif threshold == None:
            weight = 1

        sample_weight = np.ones(self.y_tr.shape)
        sample_weight[self.y_tr == 0] = weight

        if verbose: bar = ProgressBar(len(ns))
        for i, n in enumerate(ns):
            ens_wrapper = RandomForestBT(n_estimators = n, random_state = 0)
            model = EnsembleBT( estimators = self.base_models, 
                                final_estimator = ens_wrapper,
                                threshold = threshold )
            model.fit(self.x_tr, self.y_tr, sample_weight = sample_weight)
            error_tr[i] = model.error(self.x_tr, self.y_tr)
            error_te[i] = model.error(self.x_te, self.y_te)
            models[i] = model
            if verbose: bar.incriment()

        model_best = models[np.argmin(error_te)]
        n_best = ns[np.argmin(error_te)]
        self.n_best = n_best

        if np.min(error_te) < self.error_best:
            self.error_best = np.min(error_te)
            self.model_best = model_best

        if verbose:
            print(f"Best n: {n_best}\n")

        n_data = {
            "ns"         : ns,
            "error_tr"   : error_tr,
            "error_te"   : error_te,
            "model_best" : model_best,
            "n_best"     : n_best
        }
        self.data["n"] = n_data

        if save:
            self.save()

        if plot:
            plot_errors( ns, "$n$", error_tr, error_te, 
                         path.join(self.file_path, "n.pdf") )

        return n_best

    def find_weights_and_threshold_grid( self, thresholds, weights, n=None,
                                         verbose=True, save=True, plot=True ):
        error_tr = np.zeros((len(weights), len(thresholds)))
        error_te = np.zeros((len(weights), len(thresholds)))

        if self.n_best != None:
            n = self.n_best
        elif n == None:
            n = 2**7

        error_best = np.inf
        model_best = None
        ind_best = (0,0)

        if verbose: bar = ProgressBar(len(weights))
        for i, weight in enumerate(weights):
            sample_weight = np.ones(self.y_tr.shape)
            sample_weight[self.y_tr == 0] = weight

            for j, threshold in enumerate(thresholds):
                ens_wrapper = RandomForestBT( n_estimators = n, 
                                              random_state = 0 )
                model = EnsembleBT( estimators = self.base_models, 
                                    final_estimator = ens_wrapper,
                                    threshold = threshold )
                model.fit(self.x_tr, self.y_tr, sample_weight = sample_weight)

                error_test = model.error(self.x_te, self.y_te)
                error_tr[i,j] = model.error(self.x_tr, self.y_tr)
                error_te[i,j] = error_test
                if error_test < error_best:
                    error_best = error_test
                    model_best = model
                    ind_best = (i,j)
            if verbose: bar.incriment()

        ind = np.unravel_index(np.argmin(error_te), error_te.shape)
        if ind != ind_best: 
            print("The selected model didn't match the selected error values!")

        weight_best = weights[ind[0]]
        threshold_best = thresholds[ind[1]]
        self.weight_best = weight_best
        self.threshold_best = threshold_best

        if error_te[ind] < self.error_best:
            self.error_best = error_te[ind]
            self.model_best = model_best

        if verbose:
            print(f"Best weights:   {weight_best}")
            print(f"Best threshold: {threshold_best}\n")

        weights_and_threshold_data = {
            "weights"        : weights,
            "thresholds"     : thresholds,
            "error_tr"       : error_tr,
            "error_te"       : error_te,
            "model_best"     : model_best,
            "weight_best"    : weight_best,
            "threshold_best" : threshold_best
        }
        self.data["weights_and_threshold"] = weights_and_threshold_data

        if save:
            self.save()

        if plot:
            plot_errors_3D( 
                weights, thresholds, "Weights", "Thresholds", 
                error_tr, error_te, 
                path.join(self.file_path, "weights_thresholds_training.pdf"),
                path.join(self.file_path, "weights_thresholds_testing.pdf") 
            )
        return weight_best, threshold_best