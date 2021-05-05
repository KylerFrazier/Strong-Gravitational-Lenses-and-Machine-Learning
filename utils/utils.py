import pickle
import numpy as np
from matplotlib import pyplot as plt
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
        self.threshold_best = None
        self.weights_best = None
    
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

    def find_n_linear( self, ns, threshold=None, weights=None,
                       verbose=True, save=True, plot=True ):
        models = []
        error_tr = []
        error_te = []

        if self.threshold_best != None:
            threshold = self.threshold_best
        elif threshold == None:
            threshold = 0.80
        if self.weights_best != None:
            weights = self.weights_best
        elif threshold == None:
            weights = np.ones(self.y_tr.shape)

        if verbose: print( ' '*11 + '_'*(len(ns)*2+1) + '\n' + \
                           "Progress: [ ", end='', flush=True )
        for n in ns:
            ens_wrapper = RandomForestBT(n_estimators = n, random_state = 0)
            model = EnsembleBT( estimators = self.base_models, 
                                final_estimator = ens_wrapper,
                                threshold = threshold )
            model.fit(self.x_tr, self.y_tr, sample_weight = weights)
            error_tr.append(model.error(self.x_tr, self.y_tr))
            error_te.append(model.error(self.x_te, self.y_te))
            models.append(model)
            if verbose: print('> ', end='', flush=True)
        if verbose: print(']')

        model_best = models[np.argmin(error_te)]
        n_best = ns[np.argmin(error_te)]
        self.n_best = n_best

        if np.min(error_te) < self.error_best:
            self.error_best = np.min(error_te)
            self.model_best = model_best

        if verbose:
            print(f"Best n: {n_best}\n")

        n_data = {
            "models"     : models,
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
            plot_errors(ns, "$n$", error_tr, error_te, "output/ensemble/n.pdf")
        
        return n_best
    
    def find_weights_and_threshold_grid( self, thresholds, weights, n=None,
                                         verbose=True, save=True, plot=True ):
        models = np.empty((len(weights), len(thresholds)))
        error_tr = np.zeros((len(weights), len(thresholds)))
        error_te = np.zeros((len(weights), len(thresholds)))

        if self.n_best != None:
            n = self.n_best
        elif n == None:
            n = 2**7

        if verbose: print( ' '*11 + '_'*(len(weights)*2+1) + '\n' + \
                           "Progress: [ ", end='', flush=True )
        for i, weight in enumerate(weights):
            for j, threshold in enumerate(thresholds):
                ens_wrapper = RandomForestBT( n_estimators = n, 
                                              random_state = 0 )
                model = EnsembleBT( estimators = self.base_models, 
                                    final_estimator = ens_wrapper,
                                    threshold = threshold )
                model.fit(self.x_tr, self.y_tr, sample_weight = weight)
                error_tr[i,j] = model.error(self.x_tr, self.y_tr)
                error_te[i,j] = model.error(self.x_te, self.y_te)
                models[i,j] = model
            if verbose: print('> ', end='', flush=True)
        if verbose: print(']')

        model_best = models[np.argmin(error_te)]
        n_best = ns[np.argmin(error_te)]
        self.n_best = n_best

        if np.min(error_te) < self.error_best:
            self.error_best = np.min(error_te)
            self.model_best = model_best

        if verbose:
            print(f"Best n: {n_best}\n")

        weights_and_threshold_data = {
            "models"     : models,
            "weights"    : weights,
            "thresholds" : thresholds,
            "error_tr"   : error_tr,
            "error_te"   : error_te,
            "model_best" : model_best,
            "n_best"     : n_best
        }
        self.data["weights_and_threshold"] = weights_and_threshold_data

        if save:
            self.save()

        # if plot:
        #     plot_errors(ns, "$n$", error_tr, error_te, "output/ensemble/n.pdf")
        
        return n_best