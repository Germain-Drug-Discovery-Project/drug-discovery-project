import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, cross_validate
warnings.simplefilter('ignore')




def premodeling_processing(bac_df: pd.DataFrame, fp_df: pd.DataFrame):
    ''' This function serves to do a bit of preprocessing after being fed and merging the fingerprint df with the
        bioactivity df. Drops pre-encoded canonical_smiles, repeat molecule_chembl_id, encodes bioactivity class,
        and seperates target pIC50 into a series.
        
        Returns in this order: Processed dataframe without target(pIC50), and series of the target
    '''
    # Merge bioactivity df and fingerprint df on name/molecule_chembl_id
    df = fp_df.merge(bac_df, right_on='molecule_chembl_id', left_on='Name')
    
    # Init empty list to fill with unique bioactivity class ids
    bac = []
    for cls in df.bioactivity_class.unique(): # gets unique class ids
        bac.append(cls) # appends to bioactivity list
    
    # Encoding bioactivity class ID into
    for i, c in enumerate(bac): # could have anywhere between 1 and 3 bioactivity classes so enumerate counts the number in bac
        df.bioactivity_class.replace([c], [i], inplace = True)
    
    # Creating target as a pandas Series of pIC50
    target = pd.Series(df.pIC50)
    
    # Renaming Name column to molecule_id for clarity
    df.rename(columns = {'Name': 'molecule_id'}, inplace = True)
    
    # Removing pre-encoded smiles, repeated molecule_chemble_id cols, and target col
    df = df.drop(columns = ['molecule_chembl_id', 'canonical_smiles', 'pIC50'])
    return df, target


class Regression_model():
    ''' Creates regression models using a variety of Sklearn models.

        Methods:
        ----------------------------------------------------------------
        > split: preforms train/test split. Can also preform X/y split if given a target array.
        
        > scaling: scales data using a scaler method determined by user input

        > metrics: utilizes cross-validation to fit and train models on a train data set that is scaled using the scaling method
        ----------------------------------------------------------------
        
        Arguments:
            - data: Pandas DataFrame
            - classifiers: List of regression models
            - names: Names of regression models
    '''
    def __init__(self, data:pd.DataFrame, classifiers: list, names: list):
        ''' Passes dataframe, list of actual classifiers and their names, as well as checks 
            for kwargs lang or top_lang
            Creates a zip of classifiers and their names
        '''
        # Creating class instance of df
        self.df = data.copy(deep = True)

        # Creating class attributes
        self.classifiers = classifiers
        self.names = names
        
        models = [(classifiers[n], names[n]) for n in range(len(names))] # creating tuple list of models and names
        self.models = models

        
    def split(self, df, target = None):
        '''
        This function takes in a dataframe and, optionally, a target_var array. Performs a train,
        test split with no stratification. Returns train and test dfs.
        '''
        
        # Checking for y specified
        if target is None: # if no y, preform regular train, validate, test split
            train, test = train_test_split(df, test_size=.25, 
                                          random_state=1312)
            
            self.train, self.test = train, test # setting self versions of each df
            return train, test
        
        # If y is specified preform X/y train, validate, test split
        else:
            X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=.25, random_state=1312)
            self.X_train, self.X_test,\
            self.y_train, self.y_test = X_train, X_test, y_train, y_test # attributes for each X/y df and array
            
            return X_train, X_test, y_train, y_test
    

    def scaling(self, scaler = MinMaxScaler()):

    
    def metrics(self, metric_types = ['neg_root_mean_squared_error', 'r2'], splits = 10, scaler_type = None, grid_search = True):
        ''' Checks for scaled flag, if False, scales data using scaler of users choice defined in scaler_type. (Read warning)
            Creates a metrics df measuring metric_type, accuracy by default. Preforms a kfold a number of times 
            determined by splits, default 10. Returns a metrics df with the metric types chosen for the models 
            defined in the initialization of the object

            ----------------------------------------------------------------------------------------------
            Arguments:
            metric_type: Default is RMSE (neg_root_mean_squared_error), R-Squared (r2), explained varianece (explained_variance)
                Can be any combination of MAE (neg_mean_absolute_error), MSE(neg_mean_squared_error) and MSELog(neg_mean_squared_log_error),
                RMSE (neg_root_mean_squared_error), Median AE (neg_median_absolute_error), and R-Squared(r2)

            splits: Int, default set to 10. Determines the number of K-folds used.

            scaler_type: Dependent upon scaled flag. If scaler is specified but scaled = True, will do nothing. If scaled = False but no
                scaler_type is defined, will choose the default MinMaxScaler. Read warning for more info.

            grid_search: Boolean, 

            WARNING: If you haven't run scaling method prior to this step, please define a scaler_type if you choose. 
            If no scaler_type defined and you havent run scaling, will use default scaler MinMaxScaler to scale data.
            ----------------------------------------------------------------------------------------------
        '''
        try: # checking if pIC50 column exists, if not raise KeyError, didnt specify a lang or top_langs
            self.df['pIC50']
        except KeyError:
            return KeyError('Missing pIC50 column in your dataframe, make sure you are pulling a dataset with pIC50 present.')
        
        try: # Checking if vectorization has already run, if yes there will be an attribute vectorized df
            self.vectorized
        except AttributeError: # If no vectorized attribute exists get vectorized df calling self.count_vectorize
            print('Have not run count_vectorize method yet, running now...')
            self.vectorized = self.count_vectorize()
            print('All done! Moving on to modeling, this may take a while...')
        target = 'label' # Setting target to label
        
        # checking for lang or top_langs
        if self.df[target].nunique() == 2: # If one lang chosen
            s = self.df[target].replace([f'{self.lang.lower()}', f'not_{self.lang.lower()}'], [1,0]) # Endode lang as 1 not_lang as 0
        else: # if top_langs
            lang_list = [l.lower() for l in list(self.top_langs.index)] # getting a list of all lower case langs in top lang
            lang_list.append('other') # appending 'other' label
            
            lang_encode = list(range(1, len(self.top_langs)+1)) # list of numbers to encode top_langs as
            lang_encode.append(0) # appending 0 for other
            s = self.df[target].replace(lang_list, lang_encode) # encoding top_langs
            
        
        X_train, X_test, y_train, y_test = self.split(self.vectorized, s)
        
        
        result = [] # init empty results list
        for (classifier, name) in self.models: # iterate through zipped models
            kfold = KFold(n_splits = splits) # number of kfolds set to splits
            scores = cross_validate(classifier, X_train, y_train, cv = kfold, scoring = metric_type, return_estimator=True) # cross validate on each kfold
            result.append(scores) # append to results
            
            msg = "{0}: Validate accuracy: {1}".format(name, scores['test_score'].mean())
            print(msg)
        
        estimators = [res['estimator'] for res in result] # list comp for estimators/classifiers
        results = [res['test_score'] for res in result] # results of validation scores
        avg_res = [round(res['test_score'].mean(), 4) * 100 for res in result] # list comp to get mean of cross val tests for each model
        metrics_df = pd.DataFrame(data = zip(self.names, avg_res), columns = ['model', f'average_{metric_type}%']) # wrap zipped model names and results in dataframe
        
        model_scores = [(estimators[n], results[n]) for n in range(len(estimators))] # Creating list of tuples for model objects and their scores
        
        # Creating attribute for testing
        self.model_scores = model_scores
        return metrics_df.sort_values(by = [f'average_{metric_type}%'], ascending = False) # return sorted metric df
    
    
    def test_on_best(self):
        ''' Gets best preforming model from a list of estimators garnered from cross validation
            and tests model accuracy on Test dataset provided as an arg. Returns model.
        '''
        # Making list of models from models_scores
        models = []
        for m in self.model_scores:
            for mdl in m[0]:
                models.append(mdl)
        # Making list of scores from cross_val
        scores = []
        for m in self.model_scores:
            for score in m[1]:
                scores.append(score)
        
        # Creating list of tuples for models and scores
        estimator_scores = [(models[n], scores[n]) for n in range(len(scores))]

        # Creating helper list to get max score
        maxs = [tup[1] for tup in estimator_scores]
        # Getting best model and score on test
        for tup in estimator_scores:
            if tup[1] == max(maxs):
                mdl = (tup[0])
                print(f'Best model: {tup[0]}\nValidate score: {round(tup[1], 4) *100}%\nTest Score: {round(mdl.score(self.X_test, self.y_test), 3) *100}%')
                return mdl