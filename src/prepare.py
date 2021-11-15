
import pandas as pd
import numpy as np
from rdkit import Chem #install library: conda install -c rdkit rdkit
from rdkit.Chem import Descriptors, Lipinski
from acquire import query_chembl


def bioactivity_class(bioactivity_df):
	'''Divide compounds into potency classes.
	Args:
		bioactivity_df: The dataframe returned from querying ChEMBL
	Returns:
		bioactivity_df: The dataframe with added potency class column
	'''

	class_names = []

	for i in bioactivity_df.standard_value:
		if i >= 10000:
			class_names.append('INACTIVE')
		elif i <= 1000:
			class_names.append('ACTIVE')
		else:
			class_names.append('intermediate')

	bioactivity_df['bioactivity_class'] = class_names

	return bioactivity_df


def lipinski(smiles):
	'''Using SMILES notation, returns the four parameters described by
	Lipinski's Rule of Five in dataframe.
	Args:
		smiles: canonical_smiles column with molecular structure (ser)
	Returns:
		descriptors: dataframe with Lipinski descriptors
	'''

	moldata = [Chem.MolFromSmiles(elem) for elem in smiles]
	descriptors = pd.DataFrame(data=np.zeros((len(moldata), 4)),
  					columns=['MW', 'LogP', 'NumHDonors', 'NumHAcceptors'])

	for ix, mol in enumerate(moldata):
		descriptors.loc[ix] = [Descriptors.MolWt(mol),Descriptors.MolLogP(mol),
  							Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol)]
	
	return descriptors


def pIC50(bioactivity_df):
	'''Convert IC50 to pIC50 scale and capping input at 100M,
	which would give negative values after negative logarithm.
	Args:
		smiles: canonical_smiles column with molecular structure (ser)
	Returns:
		bioactivity_df: The dataframe with added pIC50 class column
	'''

	pIC50 = []

	for ic in bioactivity_df.standard_value:
		ic = min(ic, 1e8) #caps values
		molar = ic * 1e-9 #converts nanomolar to molar
		pIC50.append(round(-np.log10(molar), 2)) #uses 3 significant digits
	
	bioactivity_df['pIC50'] = pIC50

	return bioactivity_df


def preprocess_bioactivity_data(TARGET_ID, save=False):
	'''Return preprocessed dataframe.
	Args:
		bioactivity_df: The dataframe returned from querying ChEMBL
		save: Set to True to save dataframe (bool)
	Returns:
		df: The preprocessed dataframe.
	'''
	#Acquire data
	bioactivity_df = query_chembl(target_id)
	#Subset data
	subset = ['molecule_chembl_id', 'canonical_smiles', 'standard_value']
	df = bioactivity_df.copy()[subset]
	df.standard_value = df.standard_value.astype('float64')
	#Add
	df = bioactivity_class(df) #add column
	df = pd.concat([df, lipinski(df.canonical_smiles)], axis=1) #add descriptors
	df = pIC50(df) #add column
	#Subtract
	df = df[df.bioactivity_class != 'intermediate'] #drop middle class
	#take assay with highest potency if multiple assays exist
	df = df.groupby('molecule_chembl_id').min()

	if save:
		#save results of query to csv
		df.to_csv(f'{TARGET_ID}_bioactivity_preprocessed.csv', index=False)

	print("Preprocessed dataframe...\n", df.tail())

	return df


if __name__ == "__main__":
	#Example query using CHEMBL molecule ID number
	target_id = 3199 #acetylcholinesterase, Rattus norvegicus
	preprocess_bioactivity_data(target_id, save=True)
