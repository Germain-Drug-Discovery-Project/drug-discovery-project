from chembl_webresource_client.new_client import new_client
import numpy as np
import pandas as pd
import os
from source.utils import *

class wrangle():

    def __init__(self, disease: str):
        self.disease = disease

    def acquire_data(self):
        '''
        This function gets disease data from local csv, or otherwise from ChEMBL database. Only pulls for single proteins
        '''
        
        if os.path.isfile(f'{self.disease}_chembl_data.csv'):
            df = pd.read_csv(f'{self.disease}_chembl_data.csv', index_col = 0)
        
        else:
            # Create and use new_client object to access ChEMBL database
            target = new_client.target
            target_query = target.search(self.disease).filter(target_type = 'SINGLE PROTEIN') # filters for single proteins only
            df = pd.DataFrame.from_dict(target_query).reset_index()
            df.to_csv(f'{self.disease}_chembl_data.csv')
        
        self.disease_df = df
        return df

    def get_bioactivity_data(self):
        '''
        This function get the bioactivity of compounds against the disease target.
        '''

        try:
            self.disease_df
        except AttributeError:
            self.acquire_data()
        

        print(f'List of single protein ChEMBL ID\'s from chosen disease:\n{self.disease_df.target_chembl_id}')
        user_target = input('Input a single protein target ID from the list:')
        self.user_target = user_target


        if os.path.isfile(f'{user_target}_bioactivity_data.csv'):
            df = pd.read_csv(f'{user_target}_bioactivity_data.csv', index_col = 0)
        
        else:
            # Get bioactivity data for our target coronavirus protein
            # The standard_type='IC50' filters for bioactivity tests using the IC50 standard of measuring
            # 'activity' will be a list of dictionaries
            activity = new_client.activity.filter(target_chembl_id = user_target).filter(standard_type='IC50')
            
            # Turn the 'activity' list of dictionaries into a pandas dataframe
            df = pd.DataFrame.from_dict(activity)
            # standard_value column represents potency
            # A smaller number means a smaller dose is needed to exhibit and effect
            # Lower value means more potent, higher value means less potent
            
            # Save a local copy of our bioactivity data
            df.to_csv(f'{user_target}_bioactivity_data.csv', index=False)
        
        self.bioactivity_df = df
        return df
    
    def prepare_dataframe(self):
        ''' Add Lapinski descriptors and pIC50. 
            Narrows df columns to 'molecule_chembl_id', 'canonical_smiles', 'standard_value'
        '''

        try:
            self.bioactivity_df
        except AttributeError:
            self.get_bioactivity_data()
        
        #Subset data
        subset = ['molecule_chembl_id', 'canonical_smiles', 'standard_value']
        df = self.bioactivity_df[subset]
        df.standard_value = df.standard_value.astype('float64')

        #Add
        df = bioactivity_class(df) #add column
        df = df.reset_index(drop=True)
        df = pd.concat([df, lipinski(df.canonical_smiles)], axis=1) #add descriptors
        
        df = pIC50(df) #add column

        self.prepped_df = df
        return df
    

    def preprocess_bioactivity_data(self, tests=False, fingerprints=True):
        '''
        Args:
            TARGET_ID: The number part of the ChEMBL ID
            test: Set to True to show U test results (bool)
            fingerprints: Set to True to ouput fingerprints (bool)
        '''

        try:
            self.prepped_df
        except AttributeError:
            self.prepare_dataframe()
        
        df = self.prepped_df

        print(f'Saving {len(df)} molecules.')

        #save results of query to csv
        df.to_csv(f'{self.user_target}_bioactivity_preprocessed.csv', index=False)

        if tests:
            print("\nMann-Whitney U tests for molecular descriptors (active vs. inactive)...")
            for column in ['MW', 'LogP', 'NumHDonors', 'NumHAcceptors']:
                kruskal_wallace(column, self.prepped_df)
        if fingerprints:
            print(f'\nComputing fingerprints (takes several minutes if molecules > 1000)...')
            #save PubChem fingerprint results to csv
            output_file = f'{self.user_target}_pubchem_fp.csv'
            compute_fingerprints(self.prepped_df, output_file)
            print("Success!\n")


