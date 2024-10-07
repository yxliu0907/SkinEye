from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from padelpy import from_smiles
from tqdm import tqdm

from utils.utils import load_from_smiles
import glob
from padelpy import padeldescriptor
import numpy as np
import pandas as pd

def smiles_reader(dir):
    df = pd.read_csv(dir)
    SMILES_list = df["smiles"].tolist()
    mols = []

    for smiles in SMILES_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mols.append(mol)
    print("Ignore {} valid SMILES".format(len(SMILES_list) - len(mols)))
    return mols


def fp2arr(fp):
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

descs = [desc_name[0] for desc_name in Descriptors._descList[:]]

del descs[42] 
del descs[13] 
del descs[11] 


def smi2rdkit_descriptor(SMILES):
    mol = Chem.MolFromSmiles(SMILES)

    desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
    rdkit_descriptor = desc_calc.CalcDescriptors(mol)
    return rdkit_descriptor



def smi2rdkit_FP(SMILES, ToBitString=False):
    mol = Chem.MolFromSmiles(SMILES)
    if ToBitString:
        rdkit_FP = Chem.RDKFingerprint(mol).ToBitString()
    else:
        rdkit_FP = Chem.RDKFingerprint(mol)

    return rdkit_FP


def smi2MACCS_FP(SMILES, ToBitString=False):
    mol = Chem.MolFromSmiles(SMILES)
    if ToBitString:
        MACCS_FP = AllChem.GetMACCSKeysFingerprint(mol).ToBitString()
        MACCS_FP = np.array(map(int, MACCS_FP))
    else:
        MACCS_FP = AllChem.GetMACCSKeysFingerprint(mol)

    # maccs = DataStructs.BulkTanimotoSimilarity(MACCS_FP[0], MACCS_FP[1:])
    return MACCS_FP


def smi2Pubchem_FP(SMILES):
    fingerprints = from_smiles(SMILES, fingerprints=True, descriptors=False)
    PubchemFP = ''.join(fingerprints.values())
    return PubchemFP


def smi2ECFP(SMILES, ToBitString=False):
    mol = Chem.MolFromSmiles(SMILES)
    if ToBitString:
        ECFP = AllChem.GetMorganFingerprintAsBitVect(mol, 3, 2048)
        ECFP = np.array(map(int, ECFP))
    else:
        ECFP = AllChem.GetMorganFingerprintAsBitVect(mol, 3, 2048)

    return ECFP


def smi2CDKextended(SMILES):
    fingerprints = load_from_smiles(SMILES, fingerprint="CDKextended")
    CDKextended = ''.join(fingerprints.values())
    CDKextended = np.array(map(int, CDKextended))
    return CDKextended


def smi2KlekotaRoth(SMILES):
    fingerprints = load_from_smiles(SMILES, fingerprint="KlekotaRoth")
    KlekotaRoth = ''.join(fingerprints.values())
    KlekotaRoth = np.array(map(int, KlekotaRoth))
    return KlekotaRoth

def list2Del_FP(smiles_list,fingerprint='CDKextended',np_arr=True):
    xml_files = glob.glob(r"./test\fingerprints_xml/*.xml")
    xml_files.sort()

    FP_list = ['AtomPairs2DCount',
               'AtomPairs2D',
               'EState',
               'CDKextended',
               'CDK',
               'CDKgraphonly',
               'KlekotaRothCount',
               'KlekotaRoth',
               'MACCS',
               'PubChem',
               'SubstructureCount',
               'Substructure']
    if not (fingerprint in FP_list):
        raise ValueError('no such fingerprint')
    fp = dict(zip(FP_list, xml_files))

    df2 = pd.DataFrame({'SMILES': smiles_list})
    df2.to_csv('molecule.smi', sep='\t', index=False, header=False)


    fingerprint_output_file = ''.join([fingerprint,'.csv']) #Substructure.csv 结果文件名
    fingerprint_descriptortypes = fp[fingerprint] #解析文件地址

    padeldescriptor(mol_dir='molecule.smi',
                    d_file=fingerprint_output_file, #'Substructure.csv'
                    #descriptortypes='SubstructureFingerprint.xml',
                    descriptortypes= fingerprint_descriptortypes,
                    detectaromaticity=True,
                    standardizenitro=True,
                    standardizetautomers=True,
                    threads=2,
                    removesalt=True,
                    log=True,
                    fingerprints=True)

    descriptors = pd.read_csv(fingerprint_output_file,index_col=0)
    if np_arr:
        descriptors = np.array(descriptors)
    return descriptors

def smi2rep(smiles, rep_type):
    if rep_type == 'rdkit_descriptor':
        return smi2rdkit_descriptor(smiles)
    elif rep_type == 'rdkit_FP':
        return smi2rdkit_FP(smiles)
    elif rep_type == 'MACCS_FP':
        return smi2MACCS_FP(smiles)
    elif rep_type == 'Pubchem_FP':
        return smi2Pubchem_FP(smiles)
    elif rep_type == 'ECFP':
        return smi2ECFP(smiles)
    elif rep_type == 'CDKextende':
        return smi2CDKextended(smiles)
    elif rep_type == 'KlekotaRoth':
        return smi2KlekotaRoth(smiles)
    else:
        raise ValueError('Invalid rep_type')


def list2rep(smiles_list, rep_type):
    if rep_type == 'rdkit_descriptor':
        fps = [smi2rdkit_descriptor(mol) for mol in tqdm(smiles_list)]
        fps = np.array(fps)
        fps.astype(np.float32)
        fps[np.isnan(fps)] = 0
        #fps = fps[~np.isnan(fps).any(axis=1)]
        return fps
    elif rep_type == 'rdkit_FP':
        fps = [smi2rdkit_FP(mol) for mol in tqdm(smiles_list)]
        fpMtx = np.array([fp2arr(fp) for fp in fps])
        return fpMtx
    elif rep_type == 'MACCS_FP':
        fps = [smi2MACCS_FP(mol) for mol in tqdm(smiles_list)]
        fpMtx = np.array([fp2arr(fp) for fp in fps])
        return fpMtx
    elif rep_type == 'PubChem':
        return list2Del_FP(smiles_list, fingerprint='PubChem', np_arr=True)
    elif rep_type == 'ECFP':
        fps = [smi2ECFP(mol) for mol in tqdm(smiles_list)]
        fpMtx = np.array([fp2arr(fp) for fp in fps])
        return fpMtx
    elif rep_type == 'CDKextended':
        return list2Del_FP(smiles_list, fingerprint='CDKextended', np_arr=True)
    elif rep_type == 'KlekotaRoth':
        return list2Del_FP(smiles_list, fingerprint='KlekotaRoth', np_arr=True)
    else:
        raise ValueError('Invalid rep_type')

if __name__ == '__main__':
    mols = smiles_reader("./dataset/test.csv")


    rdkit_descriptor = smi2rdkit_descriptor("CCC")
    print(rdkit_descriptor)

    exit()
    maccs = smi2MACCS_FP("CCC")
    print(maccs)

    PBC_FP = smi2Pubchem_FP("CCC")
    print(PBC_FP)

    fingerprints = from_smiles('CCC', fingerprints=True, descriptors=False)
    print(''.join(fingerprints.values()))

    CDK = smi2CDKextended('CCC')
    print(len(CDK))

    KR = smi2KlekotaRoth('C1=CC=CC=C1')
    print(len(KR))
