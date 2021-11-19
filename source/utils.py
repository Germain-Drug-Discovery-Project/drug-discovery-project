import pandas as pd
import numpy as np
from scipy import stats as scs
from rdkit import Chem #conda install -c rdkit rdkit
from rdkit.Chem import Descriptors, Lipinski
from padelpy import padeldescriptor #pip install padelpy


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
			class_names.append('INTERMEDIATE')

	bioactivity_df['bioactivity_class'] = class_names

	return bioactivity_df


def lipinski(bioactivity_df):
	'''Using SMILES notation, returns the four parameters described by
	Lipinski's Rule of Five in dataframe.
	Args:
		bioactivity_df: The dataframe returned from querying ChEMBL
	Returns:
		bioactivity_df: The dataframe with added Lipinski descriptors
	'''
	smiles = bioactivity_df.canonical_smiles
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
		value = round(-np.log10(molar), 5) #uses 6 significant digits
		pIC50.append(value)
	
	bioactivity_df['pIC50'] = pIC50

	return bioactivity_df


def kruskal_wallace(y, df, alpha=.05):
	'''Compare and interpret the active vs. inactive samples
	   using a Mann-Whitney U test.
	   Args:
	   		y: dependent variable (IC50 or Lapinski descriptor)
	   		df: Datframe containing molecules.
	   		alpha: threshold value used to judge significance	
	'''
	print("\n", y)
	active = df[df.bioactivity_class=='ACTIVE'][y]
	intermediate = df[df.bioactivity_class=='INTERMEDIATE'][y]
	inactive = df[df.bioactivity_class=='INACTIVE'][y]
	h, p = scs.kruskal(active, intermediate, inactive)
	print('   H statistic = %.0f, p = %.3f' %(h, p))
	if p > alpha: print('   Same distribution. Fail to reject H0.')
	else: print('   Different distribution. Reject H0.')


def compute_fingerprints(bioactivity_df, output_file, fp='PubchemFingerprinter'):
	'''Computes and outputs binary substructure fingerprint.
	Args:
		bioactivity_df: Dataframe containing SMILES notation
		output file: Where to save the prints (str)
	'''
	#Make the input file
	df_selection = bioactivity_df[['canonical_smiles','molecule_chembl_id']]
	df_selection.to_csv('molecule.smi', sep='\t', index=False, header=False)

	#XML PaDEL files available at github.com/dataprofessor/bioinformatics
	#path to fingerprinter
	descriptortype = f'source/padel_descriptor/{fp}.xml'
	padeldescriptor(mol_dir='molecule.smi',
					d_file=output_file,
					descriptortypes=descriptortype,
					detectaromaticity=True, 
					standardizenitro=True, 
					standardizetautomers=True,
					threads=2, 
					removesalt=True,
					log=True,
					fingerprints=True)


#A hot dog vendor asked a Buddhist monk, "What would you like?"
#The monk said. "Make me one with everything"
#The monk paid with a twenty dollar bill and asked, "Where's my change?"
#The hot dog vendor said, "Change comes from within."