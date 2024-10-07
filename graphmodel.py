import datetime
import os
import pickle
import random
import warnings

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.MolStandardize.rdMolStandardize import ChargeParent
from rdkit.Chem.SaltRemover import SaltRemover
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, roc_curve, auc
from torch import nn
from torch.nn.functional import softmax
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, GCN, GIN, GAT, GraphSAGE
from tqdm import tqdm

from utils.utils import mean_std

def figure_draw(loss_list, acc_list, epochs, address, dataset, pretrain=True):
    plt.figure(dpi=300)
    fig, ax = plt.subplots(constrained_layout=True)
    ax_sub = ax.twinx()

    if pretrain:
        l1, = ax.plot(list(range(1, epochs + 1)), loss_list, '-', label='Loss', color='red')
        l2, = ax_sub.plot(list(range(1, epochs + 1)), acc_list, '-', label='Accuracy', color='blue')

        plt.legend(handles=[l1, l2], labels=['Loss', 'Accuracy'], loc=0, fontsize=15)

        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Loss', fontsize=15)

        plt.title("Pretrain")
        ax_sub.set_ylabel('Accuracy', fontsize=15)

        plt.savefig(address + dataset +'LossAcc.jpg')
        plt.show()
    else:
        l1, = ax.plot(list(range(1, epochs + 1)), loss_list, '-', label='Loss', color='red')
        l2, = ax_sub.plot(list(range(1, epochs + 1)), acc_list, '-', label='ROC-AUC', color='blue')

        plt.legend(handles=[l1, l2], labels=['Loss', 'ROC-AUC'], loc=0, fontsize=15)

        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Loss', fontsize=15)

        plt.title("Finetune" + "_" + dataset)
        ax_sub.set_ylabel('ROC-AUC', fontsize=15)

        plt.savefig(address + dataset + 'LossAUC.jpg')
        plt.show()

def lr_scheduler_create(warmup_epochs_rate, epochs, optimizer):
    if warmup_epochs_rate < 0 or warmup_epochs_rate > 0.5:
        raise ValueError("warmup_epochs_rate must be the range of (0,0.5)")
    warmup_epochs = round(epochs * warmup_epochs_rate)

    def lr_function(current_epochs):

        if current_epochs < warmup_epochs:
            return current_epochs / warmup_epochs
        else:
            return (np.cos(np.pi * (current_epochs - warmup_epochs) / (epochs - warmup_epochs)) + 1) / 2

    rate_list = []
    for current_epochs in range(0, epochs + 1):
        rate_list.append(lr_function(current_epochs))

    def fn(epoch):
        return rate_list[epoch]

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn, last_epoch=-1,verbose=True)
    return scheduler
def folds_split(dataset, folds=5, seed=0):
    num_mols = len(dataset) 
    random.seed(seed) 
    all_idx = list(range(num_mols)) 
    random.shuffle(all_idx) 

    idx_list = []
    step = 1/folds

    for i in list(range(folds)):
        sub_idx = all_idx[int(i * step * num_mols):int((i + 1) * step * num_mols)]
        idx_list.append(sub_idx)

    for i in list(range(folds-1)):
        assert len(set(idx_list[i]).intersection(set(idx_list[i+1]))) == 0


    num_idx = 0
    for i in list(range(folds)):
        num_idx += len(idx_list[i])
    assert num_idx == num_mols

    fold_datasets_list = []
    for i in list(range(folds)):
        train_list=list(range(folds))
        train_list.pop(i)
        valid_idx = idx_list[i]
        train_idx = []
        for t in train_list:
            train_idx += idx_list[t]
        train_dataset = dataset[train_idx]
        valid_dataset = dataset[valid_idx]
        fold_list = []
        fold_list.append(train_dataset)
        fold_list.append(valid_dataset)
        fold_datasets_list.append(fold_list)

    return fold_datasets_list


def roc_figure(fpr,tpr,address):
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower right")
    plt.show()

    plt.savefig(address + 'ROC.jpg')

class GNN_standard(torch.nn.Module):
    def __init__(self, GNN_type="gin", num_layer=5,
                 in_channels=20,hidden_channels=52,out_channels=2,
                 jknet_type='lstm', drop_out=0, GNN_layers=5,
                 norm_type='BatchNorm', activation_function='relu',heads=10):
        super(GNN_standard, self).__init__()
        self.GNN_type = GNN_type
        self.num_layer = num_layer
        self.jknet_type = jknet_type
        self.drop_out = drop_out

        if GNN_type == "gin":
            self.GNN = GIN(in_channels=in_channels, hidden_channels=hidden_channels,
                               num_layers=GNN_layers, out_channels=out_channels, dropout=drop_out,
                               jk=jknet_type)
        elif GNN_type == "gcn":
            self.GNN = GCN(in_channels=in_channels, hidden_channels=hidden_channels,
                               num_layers=GNN_layers, out_channels=out_channels, dropout=drop_out,
                                jk=jknet_type)
        elif GNN_type == "gat":
            self.GNN = GAT(in_channels=in_channels, hidden_channels=hidden_channels,
                               num_layers=GNN_layers, out_channels=out_channels, heads=heads, dropout=drop_out,
                                jk=jknet_type)
        elif GNN_type == "sage":
            self.GNN = GraphSAGE(in_channels=in_channels, hidden_channels=hidden_channels,
                                     num_layers=GNN_layers, out_channels=out_channels, dropout=drop_out,
                                      jk=jknet_type)
        else:
            raise ValueError('no such encoder_type')



    def forward(self, data):
        data.x = self.GNN(x=data.x, edge_index=data.edge_index)
        return data
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Invalid SMILES,please check the dataset')

    atoms_features = []
    for atom in mol.GetAtoms():
        atoms_features.append(
            [atom_features_list['possible_atomic_num_list'].index(atom.GetAtomicNum())] +
            [atom_features_list['possible_formal_charge_list'].index(atom.GetFormalCharge())] +
            [atom_features_list['possible_chirality_list'].index(atom.GetChiralTag())] +
            [atom_features_list['possible_hybridization_list'].index(atom.GetHybridization())] +
            [atom_features_list['possible_numH_list'].index(atom.GetTotalNumHs())] +
            [atom_features_list['possible_implicit_valence_list'].index(atom.GetImplicitValence())] +
            [atom_features_list['possible_degree_list'].index(atom.GetDegree())] +
            [atom_features_list['possible_Aromatic'].index(int(atom.GetIsAromatic()))]
        )
    atoms_features = torch.tensor(np.array(atoms_features), dtype=torch.float32)

    bonds_index = []
    bonds_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bonds_index.append((i, j))
        bonds_features.append([bond_features_list['possible_bonds'].index(bond.GetBondType())] +
                              [bond_features_list['possible_bond_dirs'].index(bond.GetBondDir())]
                              )
    if None in bonds_features:
        print(smiles)
    bonds_index = torch.tensor(np.array(bonds_index).T, dtype=torch.long)
    bonds_features = torch.tensor(np.array(bonds_features), dtype=torch.float32)

    return atoms_features, bonds_index, bonds_features

def clean_mol(mol, Cleanup=True, Fragment=True, Desalt=True, Uncharged=True, Hydrogen=False):
    if Cleanup:
        mol = rdMolStandardize.Cleanup(mol)
    if Fragment:
        mol = rdMolStandardize.FragmentParent(mol)
    if Desalt:          remover = SaltRemover()
        mol = remover.StripMol(mol, dontRemoveEverything=True)
    if Uncharged:
        try:
            mol = ChargeParent(mol)
        except:
            return None

    if Hydrogen:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

def mol_check(smiles):
    if smiles is None:
        return True

    mol = Chem.MolFromSmiles(smiles)

    if "*" in smiles:
        return True

    if "C" or "c" in smiles:
        pass
    else:
        return True

    for atom in mol.GetAtoms():
        if not atom.GetAtomicNum() in atom_features_list['possible_atomic_num_list']:
            #print('atom' + str(atom.GetAtomicNum()) + 'not in list')
            return True



    return False

def train(model, device, optimizer, loader):
    criterion = nn.CrossEntropyLoss()

    model.train()

    train_step = tqdm(loader)

    loss_total = 0

    for step, data in enumerate(train_step):  
        data = data.to(device)
        model = model.to(device)

        data = model(data)
        data.prob = global_mean_pool(data.x,data.batch)
        data.prob = softmax(data.prob)
        data.y = torch.reshape(data.y, data.prob.shape)
        loss = criterion(data.prob, data.y.to(torch.float32))  
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        loss_total += float(loss.cpu().item())

    train_loss = loss_total / len(train_step)
    return train_loss


def eval(model, device, loader, test=False):
    model.eval()

    y_true = []  
    y_prob = []  
    y_scores = []  

    for step, data in enumerate(loader):  
        data = data.to(device)
        with torch.no_grad():
            data = model(data)
            data.prob = global_mean_pool(data.x, data.batch)
            data.prob = softmax(data.prob)
            data.pred = torch.round(data.prob)
        data.label = torch.reshape(data.y, (-1, 2))

        '''
        y_true = y_true + data.label[:, 1].tolist()
        y_prob = y_prob + data.prob[:, 1].tolist()
        y_scores = y_scores + data.pred[:, 1].tolist()
        '''

        y_true.append(data.label[:, 1])
        y_prob.append(data.prob[:, 1])
        y_scores.append(data.pred[:, 1])

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_prob = torch.cat(y_prob, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    y_scores = np.around(y_scores, 0).astype(int)  
    if test:
        return y_true, y_prob, y_scores
    else:
        return roc_auc_score(y_true, y_prob)



def main(GNN_dict,GNN_type="gin"):
    warnings.filterwarnings("ignore")  

    time_now = str(datetime.datetime.now())
    time_now = time_now.replace(" ", "_")
    time_now = time_now.replace(":", "-")
    path = './model_save' + '/' + time_now + "Standard_GNN" + '/'

    device = GNN_dict.get('device')
    seed = GNN_dict.get('seed')
    warmup_epochs_rate = GNN_dict.get('warmup_epochs_rate')
    epochs = GNN_dict.get('epochs')
    batch_size = GNN_dict.get('batch_size')
    number_of_multi_tasks = GNN_dict.get('number_of_multi_tasks')
    lr = GNN_dict.get('lr')
    decay = GNN_dict.get('decay')
    graphpred_type = GNN_dict.get('graphpred_type')
    num_layers = GNN_dict.get('num_layers')
    hidden_channels = GNN_dict.get('hidden_channels')
    drop_out = GNN_dict.get('drop_out')
    norm_type = GNN_dict.get('norm_type')
    activation_function = GNN_dict.get('activation_function')



    os.makedirs(os.path.dirname(path))

    print(GNN_dict)

    with open(path + 'fine_tune_parameters.txt', mode='a', encoding='utf-8') as f:
        f.write(str(GNN_dict))

    with open(path + 'encoder_dict.pkl', 'wb') as f:
        pickle.dump(GNN_dict, f)

    np.random.seed(seed)
    torch.cuda.device_count()
    device = torch.device("cuda:0")
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    molecule_dataset = FinetuneMoleculeDataset(dataset_dir="./dataset/skin.csv")

    folds_auc = []
    folds_precision = []
    folds_recall = []
    folds_f1 = []
    folds_epoch = []
    index = 0
    for data_fold in folds_split(molecule_dataset, folds=5, seed=0):

        train_dataset = data_fold[0]
        test_dataset = data_fold[1]
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True,
                                  pin_memory=True, prefetch_factor=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False,
                                 pin_memory=True, prefetch_factor=2)

        model = GNN_standard(GNN_type=GNN_type, num_layer=5,
                 in_channels=8,hidden_channels=50,out_channels=2, drop_out=0, GNN_layers=5,
                 norm_type='BatchNorm', activation_function='relu',heads=10).cuda(device=device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

        scheduler = lr_scheduler_create(warmup_epochs_rate, epochs,optimizer)


        loss_list = []
        test_auc_list = []
        lr_list = []
        last_test_auc = 0

        for epoch in range(1, epochs + 1):
            train_loss = train(model, device, optimizer, train_loader)

            scheduler.step()

            train_auc = eval(model, device, train_loader)
            y_true, y_prob, y_scores = eval(model, device, test_loader, test=True)

            test_auc = roc_auc_score(y_true, y_prob)
            test_precision = precision_score(y_true, y_scores)
            test_recall = recall_score(y_true, y_scores)
            test_f1 = f1_score(y_true, y_scores)

            print('epoch=', epoch, ';', 'loss=', train_loss)
            print('train_auc=', train_auc)
            print('test_auc=', test_auc, ';', 'test_precision=', test_precision, ';', 'test_recall=', test_recall, ';',
                  'test_f1=', test_f1)
            print('lr=', str(optimizer.state_dict()['param_groups'][0]['lr']))

            loss_list.append(train_loss)
            test_auc_list.append(test_auc)
            lr_list.append(str(optimizer.state_dict()['param_groups'][0]['lr']))

            if test_auc > last_test_auc:

                fpr, tpr, thersholds = roc_curve(y_true, y_prob)

                best_auc = test_auc
                print('best_auc=', best_auc)
                best_precision = test_precision
                best_recall = test_recall
                best_f1 = test_f1
                best_epoch = epoch
                last_test_auc = test_auc

        df = pd.DataFrame(
            {'epoch': list(range(1, epochs + 1)), 'loss': loss_list, 'auc': test_auc_list, 'lr': lr_list})
        df.to_csv(path + 'fold' + str(index) + '.csv', index=False)

        folds_auc.append(best_auc)
        folds_precision.append(best_precision)
        folds_recall.append(best_recall)
        folds_f1.append(best_f1)
        folds_epoch.append(best_epoch)

        roc_figure(fpr, tpr, path + 'fold' + str(index) + '_')  
        figure_draw(loss_list, test_auc_list, epochs, path, 'fold_' + str(index),
                    pretrain=False)  
        index += 1

    final_auc = mean_std(folds_auc)
    final_precision = mean_std(folds_precision)
    final_recall = mean_std(folds_recall)
    final_f1 = mean_std(folds_f1)
    final_epoch = mean_std(folds_epoch)

    print('auc=', final_auc, ';', 'precision=', final_precision, ';', 'recall=', final_recall, ';', 'f1=', final_f1,
          ';', 'epoch=', final_epoch)

    with open(path + 'metrics.txt', 'w') as f:  
        f.write('auc_list=' + str(folds_auc) + '\n')
        f.write('auc=' + final_auc + '\n')
        f.write('precision=' + final_precision + '\n')
        f.write('recall=' + final_recall + '\n')
        f.write('f1=' + final_f1 + '\n')
        f.write('epoch=' + final_epoch + '\n')


if __name__ == "__main__":
    warnings.filterwarnings
    GNN_dict = {'device': "cuda",  # type=str, default="cuda", help:"cpu","cuda"
                     'batch_size': 256,  # type=int, default=256, help=input batch size for training
                     'seed': 0,  # type=int, default=0, help:Random Seed
                     'epochs': 30,  # type=int, default=50, help="epoch"
                     'lr': 0.0001,  # type=float, default=0.005, help=learning rate
                     'number_of_multi_tasks': 1,  
                     'warmup_epochs_rate': 0.05,  # type=float, default=0.1,help=warmup_epochs_rate:(0,0.5)
                     'graphpred_type': 'MLP',  # type=str, default=0.1, help="MLP"
                     'num_layers': 2,  # type=int, default=10, help=layers
                     'hidden_channels': 80,  # type=int, default=128, help=hidden_channels
                     'drop_out': 0.25,  # type=float, default=0.1, help=drop_out rate
                     'decay': 0.05,  # type=float, default=0.05, help=weight decay
                     'norm_type': "BatchNorm",
                     # type=str, default='GraphNorm',help="norm_type:'BatchNorm', 'LayerNorm','GraphNorm'"
                     'activation_function': "gelu"
                     # type=str, default='sigmoid',help="relu", "leakyrelu", "prelu", "sigmoid","tanh"
                     }
    main(GNN_dict,GNN_type="gin")