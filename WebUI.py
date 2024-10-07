import json

import gradio as gr

import main
import main2
from plot import tanimoto, PCA_3D, mol_visualize


def RF_Default():
    RF_FP = "rdkit_FP"
    criterion = "gini"
    max_features = "sqrt"
    n_estimators = 100
    spliter = "best"
    bootstrap = True
    oob_score = False
    warm_start = False
    min_samples_leaf = 1
    path = "./"
    return RF_FP, criterion, spliter, max_features, n_estimators, bootstrap, oob_score, warm_start, min_samples_leaf, path

with gr.Blocks(title="tox predictor", theme=gr.themes.Soft()) as app:
    gr.Markdown(value=
                "The software is open source under the MIT protocol, and the author does not have any control over the software, and the user of the software and the voice propagating the software export are solely responsible. <br> If you do not agree with the terms, you may not use or reference any code and files in the software package. For details, see root <b>LICENSE</b>.")

    with gr.Tabs():
        with gr.TabItem("1-Datasets analysis"):
            with gr.TabItem("tanimoto test"):
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            file_name = gr.File(file_count="single",
                                                file_types=[".csv"],
                                                type="filepath",
                                                label="Dataset file",
                                                height=2)
                            gr.Markdown(
                                value="*Please enter the data set in csv format, which includes two columns SMILES and class.")
                        tanimoto_FP = gr.Dropdown(["rdkit_FP", "MACCS_FP", "ECFP"],
                                                  label="Tanimoto representation",
                                                  info="Choose the molecular fingerprint you want to use. ECFP is recommended.",
                                                  scale=1)
                        color = gr.Dropdown(["Blues", "Greens", "Greys", "YlGnBu", "BuGn"],
                                            label="ColorMap",
                                            allow_custom_value=True,
                                            info="Choose the main color you want. Blues is recommended.",
                                            scale=1)
                    with gr.Column():
                        # tanimoto_image = gr.Image(show_download_button=True)
                        tanimoto_image = gr.Plot(label="Plot")
                        generate_butten = gr.Button(value="Generate",
                                                    scale=2,
                                                    variant="primary")
                        generate_butten.click(tanimoto.main, [file_name, tanimoto_FP, color], tanimoto_image)

            with gr.TabItem("split test"):
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            file_name = gr.File(file_count="single",
                                                file_types=[".csv"],
                                                type="filepath",
                                                label="Dataset file",
                                                height=2)
                            gr.Markdown(
                                value="*Please enter the data set in csv format, which includes two columns SMILES and class.")
                        split = gr.Dropdown(["8:1:1", "6:2:2"],
                                            label="Split ratio",
                                            info="Choose the split radio you want to use. 6:2:2 is recommended.",
                                            scale=1)
                        FP = gr.Dropdown(["rdkit_FP", "MACCS_FP", "ECFP"],
                                         label="3D-PCA representation",
                                         info="Choose the molecular fingerprint you want to use. ECFP is recommended.",
                                         scale=1)
                    with gr.Column():
                        # tanimoto_image = gr.Image(show_download_button=True)
                        split_image = gr.Plot(label="Plot")
                        generate_butten1 = gr.Button(value="Generate",
                                                     scale=2,
                                                     variant="primary")
                        generate_butten1.click(PCA_3D.main, [file_name, split, FP], split_image)

        with gr.TabItem("2-Model training"):
            with gr.Tabs():
                with gr.TabItem("Machine Learning"):
                    with gr.Tabs():
                        with gr.TabItem("Random Forest"):
                            with gr.Row():
                                with gr.Column():
                                    with gr.Group():
                                        file_name = gr.File(file_count="single",
                                                            file_types=[".csv"],
                                                            type="filepath",
                                                            label="Dataset file")
                                        gr.Markdown(
                                            value="*Please enter the data set in csv format, which includes two columns SMILES and class.")
                                        RF_FP = gr.Dropdown(
                                            ['rdkit_descriptor', "rdkit_FP", "MACCS_FP", 'PubChem', 'CDKextended',
                                             'KlekotaRoth', "ECFP"],
                                            label="ML representation",
                                            scale=1)
                                with gr.Column():
                                    train1_butten = gr.Button(value="Training",
                                                              scale=3,
                                                              variant="primary")
                                    default_butten = gr.Button(value="Default setting",
                                                               scale=1,
                                                               variant="secondary")
                            with gr.Row():
                                criterion = gr.Dropdown(["gini", "entropy"], label="criterion", info="default:gini")
                                spliter = gr.Dropdown(["best", "random"], label="spliter", info="default:best")
                                max_features = gr.Dropdown(["sqrt", "log2"], label="max_features",
                                                           info="default:sqrt")
                                n_estimators = gr.Slider(minimum=0, maximum=200, label="n_estimators",
                                                         info="recommend:100")
                            with gr.Row():
                                bootstrap = gr.Dropdown([True, False], label="bootstrap", info="default:True")
                                oob_score = gr.Dropdown([True, False], label="oob_score", info="default:False")
                                warm_start = gr.Dropdown([True, False], label="warm_start", info="default:False")
                                min_samples_leaf = gr.Slider(minimum=0, maximum=100, label="min_samples_leaf",
                                                             info="recommend:1")
                            with gr.Column():
                                with gr.Row():
                                    path1 = gr.Text(label="storage path",
                                                    info="Enter the path that your model needs to be stored in.")
                                with gr.Row():
                                    AUC_fold = gr.Text(label="AUC_5Fold")
                                    Accuracy_fold = gr.Text(label="Accuracy_5Fold")
                                    Sensitivity_fold = gr.Text(label="Sensitivity_5Fold")
                                    Specificity_fold = gr.Text(label="Specificity_5Fold")
                                    MCC_fold = gr.Text(label="MCC_5fold")
                                    RE_fold = gr.Text(label="RE_5fold")
                                with gr.Row():
                                    AUC_valid = gr.Text(label="AUC_valid")
                                    Accuracy_valid = gr.Text(label="Accuracy_valid")
                                    Sensitivity_valid = gr.Text(label="Sensitivity_valid")
                                    Specificity_valid = gr.Text(label="Specificity_valid")
                                    MCC_valid = gr.Text(label="MCC_valid")
                                    RE_valid = gr.Text(label="RE_valid")
                                    default_butten.click(RF_Default, [],
                                                         [RF_FP, criterion, spliter, max_features, n_estimators, bootstrap, oob_score, warm_start, min_samples_leaf, path1])
                                    train1_butten.click(main.WebUI_RF, [file_name, RF_FP,
                                                                        criterion, max_features, n_estimators,
                                                                        bootstrap, oob_score,
                                                                        warm_start, min_samples_leaf, path1],
                                                        [AUC_fold, Accuracy_fold, Sensitivity_fold,
                                                         Specificity_fold, MCC_fold, RE_fold, AUC_valid, Accuracy_valid,
                                                         Sensitivity_valid, Specificity_valid, MCC_valid, RE_valid])

                        with gr.TabItem("Support Vector Machine"):
                            gr.Markdown(value="*请输入csv格式的数据集，其中包括SMILES和class两列")
                        with gr.TabItem("eXtreme Gradient Boosting"):
                            gr.Markdown(value="*请输入csv格式的数据集，其中包括SMILES和class两列")
                        with gr.TabItem("Light Gradient Boosting Machine"):
                            gr.Markdown(value="*请输入csv格式的数据集，其中包括SMILES和class两列")
                with gr.TabItem("Graph Neural Network"):
                    gr.Markdown(
                        value="*Please enter the data set in csv format, which includes two columns SMILES and class.")
        with gr.TabItem("3-Interpretability prediction"):
            with gr.TabItem("SHAP of whole dataset"):
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            file_name = gr.File(file_count="single",
                                                file_types=[".csv"],
                                                type="filepath",
                                                label="Dataset file",
                                                height=2)
                            gr.Markdown(
                                value="*Please enter the data set in csv format, which includes two columns SMILES and class.")
                            file_name2 = gr.File(file_count="single",
                                                 file_types=[".pt"],
                                                 type="filepath",
                                                 label="Model file",
                                                 height=2)
                    with gr.Column():
                        generate_butten = gr.Button(value="Generate",
                                                    scale=2,
                                                    variant="primary")
                with gr.Row():
                    figure1 = gr.Plot(label="Plot")
                    figure2 = gr.Plot(label="Plot")
                    generate_butten.click(main2.main_figure1, [file_name], figure1)
                    generate_butten.click(main2.main_figure2, [file_name], figure2)
            with gr.TabItem("SHAP of single SMILES"):
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            input_smiles = gr.Text(label="SMILES input")
                            gr.Markdown(value="*Please enter the SMILES you want to predict.")
                            file_name3 = gr.File(file_count="single",
                                                 file_types=[".pt"],
                                                 type="filepath",
                                                 label="Model file",
                                                 height=2)
                    with gr.Column():
                        generate_butten2 = gr.Button(value="Generate",
                                                     scale=2,
                                                     variant="primary")
                with gr.Row():
                    predict = gr.Text(label="predict", value="Negative")

                with gr.Row():
                    figure3 = gr.Plot(label="Plot")
                    generate_butten2.click(main2.main_figure3, [file_name], figure3)

            with gr.TabItem("Heatmap of single SMILES"):
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            input_smiles2 = gr.Text(label="SMILES input")
                            gr.Markdown(value="*Please enter the SMILES you want to predict.")
                            file_name4 = gr.File(file_count="single",
                                                 file_types=[".pt"],
                                                 type="filepath",
                                                 label="Model file",
                                                 height=2)
                    with gr.Column():
                        generate_butten3 = gr.Button(value="Generate",
                                                     scale=2,
                                                     variant="primary")
                with gr.Row():
                    predict4 = gr.Text(label="predict", value="Negative")
                    figure5 = gr.HTML()
                with gr.Row():
                    figure4 = gr.Plot(label="Plot")
                    generate_butten3.click(mol_visualize.main, [file_name], [figure4, figure5])

app.launch(inbrowser=True, server_name="0.0.0.0", quiet=True, )
