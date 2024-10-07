import glob
import warnings
import random

from collections import OrderedDict
from csv import DictReader
from datetime import datetime
from os import remove
from time import sleep

import numpy as np
from padelpy import padeldescriptor

def load_from_smiles(smiles,
                output_csv: str = None,
                fingerprint: str = "Substructure",
                maxruntime: int = -1,
                ) -> OrderedDict:
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

    fp = dict(zip(FP_list, xml_files))





    fingerprint_output_file = ''.join([fingerprint, '.csv'])  
    fingerprint_descriptortypes = fp[fingerprint]  


    # unit conversion for maximum running time per molecule
    # seconds -> milliseconds
    if maxruntime != -1:
        maxruntime = maxruntime * 1000

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")#[:-3]
    filename = timestamp + str(random.randint(1e8,1e9))

    with open("./test/cache/{}.smi".format(filename), "w") as smi_file:
        if type(smiles) == str:
            smi_file.write(smiles)
        elif type(smiles) == list:
            smi_file.write("\n".join(smiles))
        else:
            raise RuntimeError("Unknown input format for `smiles`: {}".format(
                type(smiles)
            ))
    smi_file.close()

    save_csv = True
    if output_csv is None:
        save_csv = False
        output_csv = "./test/cache/{}.csv".format(timestamp)

    for attempt in range(3):
        try:
            padeldescriptor(mol_dir="../test/cache/{}.smi".format(filename),
                            d_file=output_csv,  # 'Substructure.csv'
                            # descriptortypes='SubstructureFingerprint.xml',
                            descriptortypes=fingerprint_descriptortypes,
                            detectaromaticity=True,
                            standardizenitro=True,
                            standardizetautomers=True,
                            threads=2,
                            removesalt=True,
                            log=True,
                            fingerprints=True)
            break
        except RuntimeError as exception:
            if attempt == 2:
                remove("./test/cache/{}.smi".format(filename))
                if not save_csv:
                    sleep(0.5)
                    try:
                        remove(output_csv)
                    except FileNotFoundError as e:
                        warnings.warn(e, RuntimeWarning)
                raise RuntimeError(exception)
            else:
                continue
        except KeyboardInterrupt as kb_exception:
            remove("./test/cache/{}.smi".format(filename))
            if not save_csv:
                try:
                    remove(output_csv)
                except FileNotFoundError as e:
                    warnings.warn(e, RuntimeWarning)
            raise kb_exception

    with open(output_csv, "r", encoding="utf-8") as desc_file:
        reader = DictReader(desc_file)
        rows = [row for row in reader]
    desc_file.close()

    remove("./test/cache/{}.smi".format(filename))
    if not save_csv:
        remove(output_csv)

    if type(smiles) == list and len(rows) != len(smiles):
        raise RuntimeError("PaDEL-Descriptor failed on one or more mols." +
                           " Ensure the input structures are correct.")
    elif type(smiles) == str and len(rows) == 0:
        raise RuntimeError(
            "PaDEL-Descriptor failed on {}.".format(smiles) +
            " Ensure input structure is correct."
        )

    for idx, r in enumerate(rows):
        if len(r) == 0:
            raise RuntimeError(
                "PaDEL-Descriptor failed on {}.".format(smiles[idx]) +
                " Ensure input structure is correct."
            )

    for idx in range(len(rows)):
        del rows[idx]["Name"]

    if type(smiles) == str:
        return rows[0]
    return rows

def folds_split(X_train,y_train, folds=5, seed=0):
    num = len(X_train) 
    random.seed(seed) 
    all_idx = list(range(num)) 
    random.shuffle(all_idx) 

    idx_list = []
    step = 1/folds

    for i in list(range(folds)):
        #print("up bottom" + str(i) + '=' + str(int(i * step * num_mols)))
        #print("down bottom" + str(i) + '=' + str(int((i + 1) * step * num_mols)))
        sub_idx = all_idx[int(i * step * num):int((i + 1) * step * num)]
        idx_list.append(sub_idx)

    for i in list(range(folds-1)):
        assert len(set(idx_list[i]).intersection(set(idx_list[i+1]))) == 0


    num_idx = 0
    for i in list(range(folds)):
        num_idx += len(idx_list[i])
    assert num_idx == num

    fold_datasets_list = []
    for i in list(range(folds)):
        train_list=list(range(folds))
        train_list.pop(i)
        valid_idx = idx_list[i]
        train_idx = []
        for t in train_list:
            train_idx += idx_list[t]

        X_train_dataset = []
        Y_train_dataset = []
        for i in train_idx:
            X_train_i = X_train[i]
            Y_train_i = y_train[i]
            X_train_dataset.append(X_train_i)
            Y_train_dataset.append(Y_train_i)

        X_valid_dataset = []
        Y_valid_dataset = []
        for i in valid_idx:
            X_valid_i = X_train[i]
            Y_valid_i = y_train[i]
            X_valid_dataset.append(X_valid_i)
            Y_valid_dataset.append(Y_valid_i)


        fold_datasets_list.append((X_train_dataset, Y_train_dataset, X_valid_dataset,Y_valid_dataset))

    return fold_datasets_list

def mean_std(data_list):
    return str(f'{np.mean(data_list):.3f}') + '±' + str(f'{np.std(data_list):.3f}')