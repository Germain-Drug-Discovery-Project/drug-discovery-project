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
            - classifiers: List of classification models
            - names: Names of classification models
            - lang: Specifies a language to create a lang/not_lang label from
            - top_langs: Specifies the top n langs to create labels for, non-top_langs will be labeled 'other'
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

    
    def metrics(self, metric_type = 'accuracy', splits = 10, scaled = False, scaler_type = None, grid_search = True):
        ''' Checks for scaled flag, if False, scales data using scaler of users choice defined in scaler_type. (Read warning)
            Creates a metrics df measuring metric_type, accuracy by default.
            Preforms a kfold a number of times determined by splits, default 10.

            -----------------------------------------------
            Arguments:
            metric_type: Default is R-Squared, can be RMSE, MSE, and SSE

            WARNING: If scaler type is defined but scaled is True, will not scale data a second time. If scaled is False
            but no scaler_type defined, will use default scaler_type MinMaxScaler
        '''
        try: # checking if label exists, if not raise KeyError, didnt specify a lang or top_langs
            self.df['label']
        except KeyError:
            return KeyError('Must specify language target in class to create models')
        
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