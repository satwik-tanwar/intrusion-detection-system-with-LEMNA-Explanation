import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore",category=DeprecationWarning)

import numpy as np

np.random.seed(1234)

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
import lime
import lime.lime_tabular
from LEMNA import RegressionMixture
from LEMNA.fusedlasso import fused_lasso

class LemnaExplainer(object):
    """class for explaining the predictions --> explains only one sample at a time"""
    def __init__(self, model, training_data, feature_names, class_names, predFun,fusedLassoGM=True,n_components=2, alpha1=1, alpha2=0.5):
        """
        Args:
            model: target model
            training_data: Data on which the model was trainded - 2D array-like
            feature_names: Names of the features in the data
            class_names: Names of the classes in which data is classified.
            predFun: a function with 2 parameters -> (model,data)
                        returns predictions(probablities) of the model for the data.
            
            fusedLassoGM: Use Gaussian Mixture Model with fused lasso reguralization or not.
            alpha1: first regularization parameter for fused lasso
            alpha2: second regularization parameter for fused lasso
        """
       
        self.model = model
        self.training_data = np.array(training_data)
        self.seq_len = self.training_data[0].shape[0]
        self.feature_names=feature_names
        self.class_names=class_names
        self.predFun=predFun
        
        self.fusedLassoGM=fusedLassoGM
        self.n_components=n_components
        self.alpha1=alpha1
        self.alpha2=alpha2
    

    def __data_inverse(self,l,data_row,num_samples):
        """Generates a neighborhood around a prediction.

        For numerical features, perturb them by sampling from a Normal(0,1) and
        doing the inverse operation of mean-centering and scaling, according to
        the means and stds in the training data. For categorical features,
        perturb by sampling according to the training distribution, and making
        a binary feature that is 1 when the value is the same as the instance
        being explained.

        Args:
            l: lime object
            data_row: 1d numpy array, corresponding to a row
            num_samples: size of the neighborhood to learn the linear model

        Returns:
            A tuple (data, inverse), where:
                data: dense num_samples * K matrix, where categorical features
                are encoded with either 0 (not equal to the corresponding value
                in data_row) or 1. The first row is the original instance.
                inverse: same as data, except the categorical features are not
                binary, but categorical (as the original data)
        """
        
        num_cols = data_row.shape[0]
        data = np.zeros((num_samples, num_cols))
        features = range(num_cols)
        
        first_row = l.discretizer.discretize(data_row)
        data[0] = data_row.copy()
        inverse = data.copy()
        for column in features:
            values = l.feature_values[column]
            freqs = l.feature_frequencies[column]
            inverse_column = l.random_state.choice(values, size=num_samples,replace=True,p=freqs)
            binary_column = (inverse_column == first_row[column]).astype(int)
            binary_column[0] = 1
            inverse_column[0] = data[0, column]
            data[:, column] = binary_column
            inverse[:, column] = inverse_column

        
        inverse[1:] = l.discretizer.undiscretize(inverse[1:])
        StandardScaler().fit_transform(inverse)
        inverse[0] = data_row
        return data, inverse

    

    def extract_feature(self,data_row,num_samples):
        """extract the important features from the input data
        Arg:
            data_row: data sample to be explained
            num_samples: number of data used for explanation
        return:
            importance_score: returns importance scores of each feature
        """

        self.data_row=np.array(data_row)
        self.pred = self.predFun(self.model,self.data_row.reshape(1,-1))
        self.label_index = np.argmax(self.pred)

        l=lime.lime_tabular.LimeTabularExplainer(self.training_data,class_names=self.class_names,feature_names=self.feature_names)
        data,data_explain=self.__data_inverse(l,self.data_row,num_samples)
       
        label_sampled = self.predFun(self.model,data_explain)[:, self.label_index]
        
        self.label_sampled=label_sampled
        self.data_explain=data_explain

        #Mixture model fitting
        if self.fusedLassoGM:
          gmm = RegressionMixture.GMM(label_sampled,n_components=self.n_components,covariance_type='full').fit(data_explain)
        else:
            gmm = GaussianMixture(n_components=self.n_components).fit(data_explain)

        # find the index for the best component
        self.gmm_pred=gmm.predict_proba(data_explain)
        res = np.argmax(self.gmm_pred, axis=1)
        best_component_idx = res[0]

        importance_score=gmm.means_[best_component_idx]
        self.ranks=np.flip(np.argsort(importance_score))
        self.rankedFeatures={
            'Features':[],
            'Importance':[]
        }
        for x in self.ranks:
            self.rankedFeatures['Features'].append(self.feature_names[x])
            self.rankedFeatures['Importance'].append(importance_score[x])
                
        return self.rankedFeatures