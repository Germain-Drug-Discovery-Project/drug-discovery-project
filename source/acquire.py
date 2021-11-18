
import pandas as pd
from chembl_webresource_client.new_client import new_client #install library with pip


def search_chembl(drug_target):
	'''Return search results for a given drug target
	'''
	target_dicts = []
	results = new_client.target.search(drug_target)
	for r in results:
		target_dicts.append({'organism': r['organism'],
							'pref_name': r['pref_name'],
							'target_chembl_id': r['target_chembl_id']})

	return target_dicts


def query_chembl(TARGET_ID, STANDARD_TYPE = 'IC50', save=False):
	'''Query ChEMBL database for the given target protein and return the bioactivity
	   data that are reported as pChEMBL values.
	Args:
		TARGET_ID: The number part of the molecule ID, CHEMBL... (int)
		STANDARD_TYPE: The measure of drug efficacy, defaults to IC50 (str) 
		save: Set to True to save dataframe (bool)
	Returns:
		Results of query (df)
	'''

	result = new_client.activity.filter(target_chembl_id=TARGET_ID)\
								.filter(standard_type=STANDARD_TYPE)
	#drop row if IC50 is missing
	df = pd.DataFrame(result).dropna(subset=['value']).reset_index(drop=True)

	if save:
		#save results of query to csv
		df.to_csv(f'{TARGET_ID}_bioactivity_data.csv', index=False)

	print(f'\nQuery results retrieved for {TARGET_ID}...')

	return df


if __name__ == "__main__":
	#Example query using CHEMBL molecule ID number
	target_id = 'CHEMBL3199' #acetylcholinesterase, Rattus norvegicus
	df = query_chembl(target_id)
