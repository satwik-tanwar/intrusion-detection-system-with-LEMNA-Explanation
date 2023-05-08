from LEMNA import lemna
from sklearn.metrics import mean_squared_error,recall_score,confusion_matrix
import numpy as np

class fid_test(object):
    def __init__(self,lemna):
        self.lemna=lemna
  
    def feature_deduction_test(self, num_fea):
        test_data = np.copy(self.lemna.data_row.reshape(1,-1))
        selected_fea = self.lemna.ranks[list(range(num_fea))]
        test_data[:,selected_fea] = 0
        pred = self.lemna.predFun(self.lemna.model,test_data)[:, self.lemna.label_index]
        return pred
    
    def synthetic_feature_test(self,num_fea):
        test_data=np.zeros((1,self.lemna.data_row.shape[0]))
        selected_fea = self.lemna.ranks[list(range(num_fea))]
        test_data[:,selected_fea] = self.lemna.data_row.reshape(1,-1)[:,selected_fea]
        pred = self.lemna.predFun(self.lemna.model,test_data)[:, self.lemna.label_index]
        return pred
    
    def feature_augmentation_test(self,num_fea):
        random_indices = np.random.choice(len(self.lemna.training_data),size=1,replace=False)
        test_seed=self.lemna.training_data[random_indices]
        test_data=np.array(test_seed).reshape(1,-1)
        selected_fea = self.lemna.ranks[list(range(num_fea))]
        test_data[:,selected_fea] = self.lemna.data_row.reshape(1,-1)[:,selected_fea]
        pred = self.lemna.predFun(self.lemna.model,test_data)[:, self.lemna.label_index]
        return pred