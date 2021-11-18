
import pandas as pd
import numpy as np
import argparse

try:
    from acquire import query_chembl
    from utils import *
except:
    from source.acquire import query_chembl
    from source.utils import *


def prepare_dataframe(bioactivity_df):
	'''Add Lapinski descriptors and pIC50, drop intermediate class.
	'''
	#Subset data
	subset = ['molecule_chembl_id', 'canonical_smiles', 'standard_value']
	df = bioactivity_df.copy()[subset]
	df.standard_value = df.standard_value.astype('float64')
	#Add
	df = bioactivity_class(df) #add column
	df = lipinski(df) #add descriptors
	df = pIC50(df) #add column
	#commented out because these examples are useful for linear modeling
	#df = df[df.bioactivity_class != 'intermediate'] #drop middle class

	return df


def preprocess_bioactivity_data(TARGET_ID, fingerprints=True,
								fp="PubchemFingerprinter", tests=False):
	'''Queries database and outputs two csv files to data folder:
	   a preprocessed dataframe and molecular fingerprints
	Args:
		TARGET_ID: The number part of the ChEMBL ID
		fingerprints: Set to True to output fingerprints (bool)
		descriptor: Index of descriptor type, defaults to Pubchem (int)
		test: Set to True to show U test results (bool)
	'''
	#Acquire data
	bioactivity_df = query_chembl(TARGET_ID)
	df = prepare_dataframe(bioactivity_df)
	
	print(f'Saving {len(df)} molecules.')
	#save results of query to csv
	df.to_csv(f'{TARGET_ID}_bioactivity_preprocessed.csv', index=False)

	if tests:
		print("\nMann-Whitney U tests for molecular descriptors (active vs. inactive)...")
		for column in ['MW', 'LogP', 'NumHDonors', 'NumHAcceptors']:
			mannwhitney(column, df)

	if fingerprints:
		print(f'\nComputing fingerprints (takes several minutes if molecules > 1000)...')
		#save PubChem fingerprint results to csv
		output_file = f'{TARGET_ID}_chemfingerprint.csv'
		compute_fingerprints(df, output_file, fp)
		print("Success!\n")


if __name__ == "__main__":

	#User should specify molecule ID(s) after filename
	parser = argparse.ArgumentParser(description='Preprocess ChEMBL molecule data.')
	parser.add_argument('id', metavar='N', type=int, nargs='+',
                    help='the ChEMBL molecule ID, e.g. CHEMBL3199')
	args = parser.parse_args()
	ids = [getattr(args, a) for a in vars(args)][0]

	for target_id in ids:
	 	preprocess_bioactivity_data(target_id, tests=False, fingerprints=True)
