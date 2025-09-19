import pandas as pd
import numpy as np
from datetime import datetime

def main(): 
    df = pd.read_pickle(r"C:\Users\E.Louwenaar\Documents\Thesis2\cleaned_mincount50_medL4 copy.pkl")
    df.info()
    df_berht = preprocess_to_behrt_format(df)
    df_berht.head()

def preprocess_to_behrt_format(df):
    """
    Transformeer je DataFrame naar BEHRT format
    Input: DataFrame met PATIENTNR, ExecutionDate, ICD10ExternalCodes, Age
    Output: DataFrame met patid, code sequence, age sequence
    """
    
    # Sorteer per patiënt op datum
    df = df.sort_values(['PATIENTNR', 'ExecutionDate'])
    
    # Groepeer per patiënt en datum
    grouped = df.groupby(['PATIENTNR', 'ExecutionDate']).agg({
        'ICD10ExternalCodes': lambda x: list(x) if isinstance(x.iloc[0], str) else x.tolist(),
        'Age': 'first'
    }).reset_index()
    
    # Voeg SEP token toe tussen visits
    patient_sequences = []
    
    for patient_id, patient_data in grouped.groupby('PATIENTNR'):
        code_sequence = []
        age_sequence = []
        
        for idx, row in patient_data.iterrows():
            # Voeg codes van deze visit toe
            codes = row['ICD10ExternalCodes']
            if isinstance(codes, str):
                codes = [codes]
            
            # Converteer ICD10 codes naar kortere format indien nodig
            codes = [code.replace('.', '') for code in codes]  # Verwijder punten
            
            # Voeg codes en leeftijd toe
            for code in codes:
                code_sequence.append(code)
                age_sequence.append(str(int(row['Age'] * 12)))  # Age in maanden
            
            # Voeg SEP toe na elke visit (behalve de laatste)
            code_sequence.append('SEP')
            age_sequence.append(str(int(row['Age'] * 12)))
        
        # Voeg CLS aan begin toe (wordt later in dataloader gedaan)
        patient_sequences.append({
            'patid': patient_id,
            'code': np.array(code_sequence),
            'age': np.array(age_sequence)
        })
    
    return pd.DataFrame(patient_sequences)


def create_vocabularies(behrt_df):
    """Maak token2idx en age2idx dictionaries"""
    
    # Verzamel alle unieke codes
    all_codes = set()
    for codes in behrt_df['code']:
        all_codes.update(codes)
    
    # Verwijder SEP uit codes voor vocabulary
    all_codes.discard('SEP')
    
    # Maak token2idx
    special_tokens = ['PAD', 'UNK', 'CLS', 'SEP', 'MASK']
    token2idx = {token: idx for idx, token in enumerate(special_tokens)}
    
    for idx, code in enumerate(sorted(all_codes)):
        token2idx[code] = len(special_tokens) + idx
    
    idx2token = {idx: token for token, idx in token2idx.items()}
    
    # Maak age vocabulary (0-110 jaar in maanden)
    age2idx = {'PAD': 0, 'UNK': 1}
    for age_months in range(110 * 12):  # 0 tot 110 jaar
        age2idx[str(age_months)] = len(age2idx)
    
    return {'token2idx': token2idx, 'idx2token': idx2token}, age2idx

if __name__ == "__main__":
    main()