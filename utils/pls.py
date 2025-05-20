import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import BaseEnsemble
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, check_random_state

class PLSClassifier(BaseEnsemble, ClassifierMixin):
    def __init__(self,
                 n_components=2,
                 random_state=None):
        self.n_components = n_components
        self.random_state = random_state
        self.labels_to_binary = {}
        self.binary_to_labels = {}

    def get_hyperparams(self):
        """Return the model hyperparameters"""
        hyperparams = {
            'n_components' : self.n_components
        }
        return hyperparams

    def labels_conversion(self, labels_list):
        """
        Return the equivalence between labels and binaries
        """
        l = list(set(labels_list))
        labels_dict = {c:idx for idx, c in enumerate(l)}
        if len(l) < 2:
            raise ValueError("Only 1 classe given to the model, needs 2")
        elif len(l) > 2:
             raise ValueError("{} classes were given, multiclass prediction is not implemented".format(len(l)))
        return np.array(l), labels_dict

    def fit(self, X, y):
        """
        Fit the model with the given data
        """
        # Check if 2 classes are inputed and convert labels to binary labels
        X, y = check_X_y(X, y)
        self.classes_, self.labels_to_binary = self.labels_conversion(y)
        self.binary_to_labels = {bin_label:str_label for str_label, bin_label in self.labels_to_binary.items()}
        y = np.array([self.labels_to_binary[l] for l in y])
        
        self.model = PLSRegression(n_components=self.n_components)
        self.model.fit(X, y)

    def predict(self, X):
        """
        Compute model predictions for data in X

        Returns:
        ----------
        predictions : array
            predictions[i] is the predicted class for he sample i in X
        """
        check_is_fitted(self, ["model"])
        X = check_array(X)
        predicted_proba = np.array([np.array([1 - pred, pred]) for pred in self.model.predict(X)])
        predictions = np.array(np.argmax(predicted_proba, axis=1), dtype=int)
        return predictions
    
    def get_coefs(self):
        """
        Returns coefficients used to compute a prediction for a sample

        Returns:
        ----------
        coefs : array
            coefs[i] is the multiplicative coefficient for feature i of each samples in X
            ndarray of shape (n_features, n_targets)
        """
        check_is_fitted(self, ["model"])
        return self.model.coef_
