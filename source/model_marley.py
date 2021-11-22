import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, cross_validate

from math import sqrt
import statistics as st
from sklearn.metrics import mean_squared_error

import itertools

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
warnings.simplefilter('ignore')
pd.set_option('display.float_format', lambda x: f'{x:,.3f}')



def premodeling_processing(bac_df: pd.DataFrame, fp_df: pd.DataFrame) -> pd.DataFrame:
    ''' This function serves to do a bit of preprocessing after being fed and merging the fingerprint df with the
        bioactivity df. Drops pre-encoded canonical_smiles, repeat molecule_chembl_id, encodes bioactivity class,
        and seperates target pIC50 into a series.
        
        Returns, in this order: Processed dataframe without targets (pIC50 and bioactivity_class), and Dataframe of the targets,
        both bioactivity_class and pIC50.
    '''
    # Merge bioactivity df and fingerprint df on name/molecule_chembl_id
    df = fp_df.merge(bac_df, right_on='molecule_chembl_id', left_on='Name').dropna()
    
    # Init empty list to fill with unique bioactivity class ids
    bac = []
    for cls in df.bioactivity_class.unique(): # gets unique class ids
        bac.append(cls) # appends to bioactivity list
    
    raw_bac = df.bioactivity_class.copy()
    # Encoding bioactivity class ID into
    for i, c in enumerate(bac): # could have anywhere between 1 and 3 bioactivity classes so enumerate counts the number in bac
        df['bioactivity_class'].replace([c], [i], inplace = True)
    
    # Creating target as a df with all target vars
    target = pd.DataFrame({'pIC50': df.pIC50, 'standard_value': df.standard_value, 'bioactivity_class': raw_bac, 'encoded_bac': df.bioactivity_class})
    
    # Renaming Name column to molecule_id for clarity
    df.rename(columns = {'Name': 'molecule_id'}, inplace = True)
    
    # Removing pre-encoded smiles, repeated molecule_chemble_id cols, and target cols. Have to remove bioactivity to prevent data leakage
    df = df.drop(columns = ['molecule_chembl_id', 'canonical_smiles', 'pIC50', 'bioactivity_class', 'standard_value'])
    return df, target


class Modeling_class():
    ''' This class can create regression models or classification models from a variety of Sklearn models. Includes methods to process data for modeling.

        Methods:
        ----------------------------------------------------------------
        > split: Preforms train/test split. Can also preform X/y split if given a target array.
        
        > scaling: Scales data using a scaler method determined by user input

        > metrics: Utilizes cross-validation to fit and train models on a train data set that is scaled using the scaling method

        > test_on_best: Chooses best model from metrics method and tests on out-of-sample test dataset
        ----------------------------------------------------------------
        
        Arguments:
            - x_data: Pandas DataFrame without targets pIC50 and bioactivity_class
            - y_data: Pandas DataFrame of targets pIC50 and bioactivity_class
            - model_types: List of sklearn models, should be classification or regression models
            - names: Names of regression models
    '''
    def __init__(self, x_data:pd.DataFrame, y_data: pd.DataFrame, regressor_types: list, reg_names: list, classifier_types: list, cls_names: list):
        ''' Passes dataframe without target, target series, list of actual regressors and their names, as well as checks 
            Creates a list of tuples of regressors and their names.

            ----------------------------------------------------------------
            Attributes initialized:
                - self.df: Copy of x_data DataFrame
                - self.regressors: Copy of list of regressors
                - self.names: Copy of list of regressor names
                - self.models: List of tuples of regressors and their names
            ----------------------------------------------------------------
        '''
        # Creating class instance of df and y_data
        self.df = x_data.reset_index(drop = True).copy(deep = True)
        self.y_data = y_data.reset_index(drop = True).copy(deep = True)
        try: # checking if pIC50 column exists or bioactivity column exists, if they do, raise error
            self.df['pIC50']
            self.df['bioactivity_class']
            raise AttributeError('bioactivity class or pIC50 are present in dataframe, must remove them and input them as a dataframe using standalone premodeling_processing function')
        except KeyError:
            pass
        
        # Creating base class attributes
        self.regressor_types = regressor_types
        self.reg_names = reg_names
        self.classifier_types = classifier_types
        self.cls_names = cls_names
        
        reg_models = [(regressor_types[n], reg_names[n]) for n in range(len(reg_names))] # creating tuple list of models and names
        self.reg_models = reg_models

        cls_models = [(classifier_types[n], cls_names[n]) for n in range(len(cls_names))] # creating tuple list of models and names
        self.cls_models = cls_models
        
    def split(self, df: pd.DataFrame, target = None, testsize = .25):
        '''
        This method takes in a dataframe and, optionally, a target_var array. Performs a train,
        test split with no stratification. Returns train and test dfs.

        ----------------------------------------------------------------
        Arguments:
            - df: Dataframe to split
            - target: Optional, default is None. Array-like, if present, preforms X/y split, returning split target 
                array along with split dataframe.
            - testsize: Float : 1.0 > range > 0.0. Determines size of test set when preforming train/test split
        ----------------------------------------------------------------
        Attributes created:
            > If no Target:
                - self.train: Dataframe, train split
                - self.test: Dataframe, test split
            > If Target:
                - self.X_train: Dataframe, train split, no target included
                - self.X_train: Dataframe, test split, no target included
                - self.y_train: Series, train split of target array
                - self.y_test: Series, test split of target array
        ----------------------------------------------------------------
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
        '''This method will scale a dataframe given a scaler, default scaler is MinMaxScaler
        '''
        scaler = scaler
        scaled_df = scaler.fit_transform(self.df[['MW','LogP','NumHDonors','NumHAcceptors']])
        scaled_df = pd.DataFrame(scaled_df, columns = ['MW','LogP','NumHDonors','NumHAcceptors'])

        conc_df = pd.concat([scaled_df, self.df.drop(columns = ['MW','LogP','NumHDonors','NumHAcceptors'])], axis = 1).drop(columns = ['molecule_id'])
        self.scaled_df = conc_df
        return scaled_df
        
    
    def regression_modeling(self, metric_types = ['neg_root_mean_squared_error', 'r2'], splits = 10, scaler_type = None):
        ''' This method will utilize K-fold cross-validation and grid search to create, fit, and train models. Will return a dataframe
            of models and the average metric score across all K-folded cross-validation splits.

            ----------------------------------------------------------------------------------------------

            Checks for scaled flag, if False, scales data using scaler of users choice defined in scaler_type. (Read warning)
            Creates a metrics df measuring metric_type, accuracy by default. Preforms a kfold a number of times 
            determined by splits, default 10. Returns a metrics df with the metric types chosen for the models 
            defined in the initialization of the object

            ----------------------------------------------------------------------------------------------
            Arguments:
            metric_type: Default is RMSE (neg_root_mean_squared_error), R-Squared (r2).
                Can be any combination of explained varianece (explained_variance), MAE (neg_mean_absolute_error), MSE(neg_mean_squared_error) and MSELog(neg_mean_squared_log_error),
                RMSE (neg_root_mean_squared_error), Median AE (neg_median_absolute_error), and R-Squared(r2)

            splits: Int, default set to 10. Determines the number of K-folds used.

            scaler_type: Dependent upon scaled flag. If scaler is specified but scaled = True, will do nothing. If scaled = False but no
                scaler_type is defined, will choose the default MinMaxScaler. Read warning for more info.


            WARNING: If you haven't run scaling method prior to this step, please define a scaler_type if you choose. 
            If no scaler_type defined and you havent run scaling, will use default scaler MinMaxScaler to scale data.
            ----------------------------------------------------------------------------------------------
        '''
        # Getting regression target
        target = pd.Series(self.y_data.pIC50) # Setting target to pIC50
        self.reg_target = target # seting attribute for reg_target


        try: # Checking if scaling has already run, if yes there will be an attribute scaled df
            self.scaled_df
            print('Scaling has already been run. Moving on to modeling, this may take a while...')
        except AttributeError: # If no scaled_df attribute exists create scaled_df calling self.scaling
            print('Have not run scaling method yet, running now...')
            if scaler_type != None:
                self.scaling(scaler = scaler_type)
            else:
                self.scaling()
            print('All done! Moving on to modeling, this may take a while...')
        

        # Preforming X/y split on reg_df and target using split method
        X_train, X_test, y_train, y_test = self.split(self.scaled_df, target)

        # Initializing empty scores dictionaries, will be filled when iterating through models for cross-validation
        model_scores_dict = {'model_type': [], 'metric': [], 'scores': []}
        avg_scores_dict = {'model_type': [], 'metric': [], 'avg_score': []}
        
        outputs = [] # init empty output list, contains entire output from cross validation, all models and metrics
        for (model_type, name) in self.reg_models: # iterate through zipped models
            model_spec_out = [] # model specific output, refreshes each iteration for use in getting model specific metrics

            kfold = KFold(n_splits = splits) # number of kfolds set to splits
            scores = cross_validate(model_type, X_train, y_train, cv = kfold, scoring = metric_types, return_estimator=True) # cross validate on each kfold
            outputs.append(scores) # append to outputs
            model_spec_out.append(scores) # append to model specific output

            # Iterate through to get the acutual results for each metric type as well as the average of each metric type, append to respective dicts
            # for each model type in the self.models list
            for metric in metric_types:
                if metric != 'r2':
                    metric_score = [abs(out[f'test_{metric}']) for out in model_spec_out] # list of all validation scores for this metric for this model iteration
                    avg_result = [abs(np.round(out[f'test_{metric}'], 5)).mean() for out in model_spec_out][0] # Average value of scores for this metric for this df
                else:
                    metric_score = [out[f'test_{metric}'] for out in model_spec_out] # list of all validation scores for this metric for this model iteration
                    avg_result = [np.round(out[f'test_{metric}'], 5).mean() for out in model_spec_out][0] # Average value of scores for this metric for this df
                # Appending to average scores dict the name, metric type, and average score
                avg_scores_dict['model_type'].append(name)
                avg_scores_dict['metric'].append(metric)
                avg_scores_dict['avg_score'].append(avg_result)

                # Appending to model scores the name, metric type and list of scores
                model_scores_dict['model_type'].append(name)
                model_scores_dict['metric'].append(metric)
                model_scores_dict['scores'].append(metric_score)

        
        estimators = [out['estimator'] for out in outputs] # list comp for estimators/regressors

        model_scores = [(estimators[n], outputs[n]) for n in range(len(estimators))] # Creating list of tuples for model objects and their scores
        
        # Creating attribute for testing
        self.model_scores_reg = model_scores
        
        # Creating dataframe from average scores dict
        avg_metric_df = pd.DataFrame.from_dict(avg_scores_dict)

        # List concat to create list of tuples of model type and the metrics for use in multi-index
        tuples = [(avg_metric_df.model_type.values[i], avg_metric_df.metric.values[i]) for i in range(len(avg_metric_df.index))]
        index = pd.MultiIndex.from_tuples(tuples, names = ['model', 'metric']) # creating multi-index to replace metric_df index


        # Creating attribute for metric_df with updated multi-index
        self.reg_metric_df = pd.DataFrame(data = avg_metric_df.avg_score.values, index = index, columns = ['avg_score'])


        print('Modeling done! Average scores are abstract represntations of how well this model type did, not actual scores.')
        return self.reg_metric_df

    def evaluaton_helpers(self):
        ''' A helper function that gets the best estimators per model type from regression cross-validation,
            the score that corresponds to that estimator, and returns the bestimators predictions for use 
            in plotting error visualizations.
        '''
        # Baseline creation
        dr = DummyRegressor(strategy='mean').fit(self.df, self.y_data.pIC50) #Baseline
        yhat_baseline = self.reg_target.mean()

        # Getting baseline RMSE
        y_pred_baseline = [yhat_baseline for i in range(len(self.reg_target))]
        base_rmse = sqrt(mean_squared_error(self.reg_target, y_pred_baseline))

        y_actual = self.reg_target

        # Creating list of tuples for models and scores
        estimator_scores = {'estimator': [], 'scores': []}

        # Making list of models from models_scores
        for out in self.model_scores_reg:
            # Making list of scores from cross_val
            estimator_scores['scores'].append(out[1]['test_neg_root_mean_squared_error'])
        
        # Getting the estimators from the model scores gegression dictionary
        for out in self.model_scores_reg:
            for mdl in out[0]:
                estimator_scores['estimator'].append(mdl)
        
        # Getting list of best estimators for each model type by grabbing best score for each model type
        bestimators = [] # empty list of best estimator for each model type
        for j, scores in enumerate(estimator_scores['scores']):
            for i, score in enumerate(scores):
                if score == max(scores):
                    bestimators.append(estimator_scores['estimator'][i+3*j])

        # Making a list of predictions from the bestimators for each model type
        yhat_preds = []
        for estimator in bestimators:
            yhat_preds.append(estimator.predict(self.df.drop(columns = ['molecule_id'])))


        # Initialize helper attributes
        self.yhat_preds = yhat_preds
        self.bestimators = bestimators
        self.y_actual = y_actual
        self.y_pred_baseline = y_pred_baseline
        self.yhat_baseline = yhat_baseline
        self.baseline_rmse = base_rmse

        
    def plot_actual_vs_pred(self):
        ''' Plots the acutal values vs the predicted values 
        '''

        self.evaluaton_helpers()

        # Baseline and line 
        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot(111)
        
        plt.plot(self.y_actual, self.y_pred_baseline, alpha=1, color="gray", label='predicted')
        plt.annotate("Baseline: Predict Using Mean", (2, 5))
        plt.plot(self.y_actual, self.y_actual, alpha=1, color="blue", label='actual')
        plt.annotate("The Ideal Line: Predicted = Actual", (1, .8), rotation=25.8)

        colors = itertools.cycle(cm.rainbow(np.linspace(0, 1, len(self.reg_names))))
        for i, (estimator, out) in enumerate(self.model_scores_reg):
            ax.scatter(self.y_actual, self.yhat_preds[i],
                    alpha=.5, color=next(colors), s=100, label=f"Model: {self.reg_names[i]}")

        plt.legend()
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted")
        plt.show()


    def plot_errors(self):
        
        self.evaluaton_helpers()

        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot(111)
        plt.axhline(label="No Error")

        colors = itertools.cycle(cm.rainbow(np.linspace(0, 1, len(self.reg_names))))
        for i, (estimator, out) in enumerate(self.model_scores_reg):
            plt.scatter(self.y_actual, (self.yhat_preds[i] - self.y_actual),
                    alpha=.5, color=next(colors), s=100, label=f"Model: {self.reg_names[i]}")
        plt.legend()
        plt.xlabel("Actual")
        plt.ylabel("Residual/Error: Predicted - Actual")
        plt.title("Plotting Errors in Predictions")
        plt.show()


    def classification_modeling(self, metric_type = 'accuracy', splits = 3):
        ''' Checks for and encodes label column
            Creates a metrics df measuring metric_type, accuracy by default.
            Preforms a kfold a number of times determined by splits.
        '''
        # Getting regression target
        target = pd.Series(self.y_data.encoded_bac) # Setting target to pIC50
        self.classes = target # seting attribute for reg_target


        try: # Checking if scaling has already run, if yes there will be an attribute scaled df
            self.scaled_df
            print('Scaling has already been run. Moving on to modeling, this may take a while...')
        except AttributeError: # If no scaled_df attribute exists create scaled_df calling self.scaling
            print('Have not run scaling method yet, running now...')
            if scaler_type != None:
                self.scaling(scaler = scaler_type)
            else:
                self.scaling()
            print('All done! Moving on to modeling, this may take a while...')
        
        X_train, X_test, y_train, y_test = self.split(self.scaled_df, target)
        
        
        result = [] # init empty results list
        for (classifier, name) in self.cls_models: # iterate through zipped models
            kfold = KFold(n_splits = splits) # number of kfolds set to splits
            scores = cross_validate(classifier, X_train, y_train, cv = kfold, scoring = metric_type, return_estimator=True) # cross validate on each kfold
            result.append(scores) # append to results
        
        estimators = [res['estimator'] for res in result] # list comp for estimators/classifiers
        results = [res['test_score'] for res in result] # results of validation scores
        avg_res = [round(res['test_score'].mean(), 4) * 100 for res in result] # list comp to get mean of cross val tests for each model
        metrics_df = pd.DataFrame(data = zip(self.cls_names, avg_res), columns = ['model', f'average_{metric_type}%']) # wrap zipped model names and results in dataframe
        
        model_scores = [(estimators[n], results[n]) for n in range(len(estimators))] # Creating list of tuples for model objects and their scores
        
        # Creating attribute for testing
        self.cls_model_scores = model_scores
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