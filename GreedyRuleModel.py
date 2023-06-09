from learned_rule_model import LearnedRuleModel
import numpy as np

#define some constants for where things are in a rule
FEATURE=0
OPERATOR=1
THRESHOLD=2
LABEL = 3

class GreedyRuleInductionModel(LearnedRuleModel):
    '''
    sub-class that uses greedy constrcutive search to find a set of rules
    that classify a dataset
    '''
    def __init__(self,max_rules=10, increments=25):
        '''constructor 
        calls the init function for the super class
        and inherit all the other methods
        ''' 
        super().__init__(max_rules=max_rules, increments=increments)
    
    def fit(self, train_X:np.ndarray, train_y:list):
        '''
        Learns a set of rules that fit the training data
        Calls then extends the superclass method.
        Makes repeated use of the method __get_examples_covered_by()

        Parameters
        ----------
        train_X - 2D numpy array of instance feature values
                  shape (num_examples, num_features)
        train_y - 1D numpy array of labels, shape(num_examples,0)
        '''

        # Call superclass method to preprocess the training set  
        super().fit(train_X,train_y)
        
        
        self.default_prediction = np.bincount(train_y.astype(int)).argmax()
        self.not_covered = np.arange(train_X.shape[0])

        improved = True
        while len(self.not_covered) > 0 and len(self.rule_set) < self.max_rules and improved:
            improved = False
            best_new_rule = None
            best_covered = None

            for feature in range(train_X.shape[1]):
                for op in range(len(self.operator_set)):
                    for threshold in range(self.num_thresholds):
                        for label in range(self.num_classes):
                            rule = np.array([feature, op, threshold, label])
                            covered = self._get_examples_covered_by(rule, train_X[self.not_covered], train_y[self.not_covered])

                            if best_covered is None or len(covered) > len(best_covered):
                                best_covered = covered
                                best_new_rule = rule
                                improved = True

            if improved:
                self.not_covered = np.delete(self.not_covered, best_covered , axis=0)  
                self.rule_set= np.row_stack((self.rule_set, best_new_rule)).astype(int)  

    def predict_one(self, example:np.ndarray)->int:
        '''
        Method that overrides the naive code in the superclass
        function GreedyRuleInduction.

        Parameters
         ---------
        example: numpy array of feature values that represent one exanple

        Returns: valid label in form of int index into set of values found in the training set
        '''
     
        prediction = self.default_prediction

        for rule in self.rule_set:
            if self._meets_conditions(example, rule):
                prediction = rule[LABEL]
                break

        return prediction
