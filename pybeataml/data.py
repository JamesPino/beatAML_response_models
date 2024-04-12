"""
First pass at data centered class.
Ideally a single class instance can handle loading of the entire dataset.

"""
import os
import re

import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split

from pybeataml.data_class import ExperimentalData
from pybeataml.load_data_from_synpase import load_table, load_file, load_excel

# colors for syncing across plots/R/python
cluster_colors = list(sns.color_palette("Dark2", 8)[1:3]) + \
                 list(sns.color_palette("Dark2", 8)[4:6])
data_colors = ["#5495CF", "#F5AF4D", "#DB4743", "#7C873E", "#FEF4D5"]

# current synapse ids, check with Camilo to see if these are the final (
# I know there are some other corrected/v2/uncorrected in the R code)
global_id = 'syn25808020'
phospho_id = 'syn26477193'  # syn25808662
rnaseq_id = 'syn26545877'
drug_response_id = 'syn25830473'
meta_file_id = 'syn26534982'
wes_id = 'syn26428827'
clusters_id = 'syn26642544'
clinical_summary_id = 'syn25796769'
metabolomics_id = 'syn53678273'
lipidomics_id = 'syn52121001'
meta_file_id2 = 'syn25807733'
acetyl_id = 'syn53484994'


def scale_col(my_col, method="z"):
    operations = {
        "median_sub": lambda x: x - x.median(),
        "median_div": lambda x: x / x.median(),
        "z": lambda x: (x - x.mean()) / x.std(),
        "maxMin": lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1
    }

    if method in operations:
        my_col = operations[method](my_col)
    else:
        warnings.warn("column was not scaled. method must be either 'median', 'maxMin', or 'z'")

    return my_col


def load_meta_for_ids():
    f_name = os.path.join(os.path.dirname(__file__), 'data/meta_ids.csv')
    if os.path.exists(f_name):
        return pd.read_csv(f_name)
    metadata_df = load_file(meta_file_id2)
    metadata_df.write_csv(f_name)
    return metadata_df


def prep_rnaseq():
    f_name = os.path.join(os.path.dirname(__file__), 'data/rna.csv')
    if os.path.exists(f_name):
        return pd.read_csv(f_name)

    data = load_table(rnaseq_id)
    subset = data[['display_label', 'labId', 'RNA counts']]
    subset.columns = ['gene_symbol', 'sample_id', 'exp_value']
    subset['source'] = 'rna_seq'
    subset['label'] = subset.gene_symbol + '_rna'
    subset.to_csv(f_name)
    return subset


def prep_metabolomics_HILIC(confidence="high"):
    f_name = os.path.join(os.path.dirname(__file__), 'data/metabolomics_HILIC.csv')
    if os.path.exists(f_name):
        return pd.read_csv(f_name)
    cols = ['display_label', 'sample_id', 'exp_value']

    # import HILIC pos & neg and drop extra rows & columns
    data_pos = load_excel(metabolomics_id, 0)
    data_pos = data_pos.iloc[:-570]  # drop unknowns
    data_pos = data_pos.drop(columns=['Blank_BEAT_AML_01_HILIC_Pos',
                                      'Blank_BEAT_AML_02_HILIC_POS',
                                      'Blank_BEAT_AML_02_HILIC_Pos2',
                                      'Blank_BEAT_AML_03_HILIC_POS',
                                      'Blank_BEAT_AML_04_HILIC_POS',
                                      'Blank_BEAT_AML_05_HILIC_POS'])  # drop blanks
    data_pos = data_pos.drop(columns=['m/z', 'RT [min]', 'Tags',
                                      'Standardized name', 'Super class',
                                      'Main class', ' Sub class', 'Formula',
                                      'Annot. DeltaMass [ppm]',
                                      'Annotation MW', 'Reference Ion'])

    data_neg = load_excel(metabolomics_id, 1)
    data_neg = data_neg.iloc[:-336]  # drop unknowns
    data_neg = data_neg.drop(columns=['Blank_BEAT_AML_01_HILIC_NEG',
                                      'Blank_BEAT_AML_01_HILIC_NEG2',
                                      'Blank_BEAT_AML_02_HILIC_NEG',
                                      'Blank_BEAT_AML_02_HILIC_NEG2',
                                      'Blank_BEAT_AML_02_HILIC_NEG3',
                                      'Blank_BEAT_AML_03_HILIC_NEG',
                                      'Blank_BEAT_AML_04_HILIC_NEG',
                                      'Blank_BEAT_AML_05_HILIC_NEG'])  # drop blanks
    data_neg = data_neg.drop(columns=['m/z', 'RT [min]', 'Tags',
                                      'Standardized name', 'Super class',
                                      'Main class', ' Sub class', 'Formula',
                                      'Annot. DeltaMass [ppm]',
                                      'Annotation MW', 'Reference Ion'])

    # drop rows not meeting confidence threshold
    if confidence == "high":
        data_pos = data_pos.iloc[:-131]
        data_neg = data_neg.iloc[:-75]

    # normalize data
    data_pos = data_pos.T
    data_pos.columns = data_pos.iloc[0]
    data_pos = data_pos[1:]
    data_pos = data_pos.apply(pd.to_numeric, errors='coerce')
    data_pos = data_pos.apply(scale_col)

    data_neg = data_neg.T
    data_neg.columns = data_neg.iloc[0]
    data_neg = data_neg[1:]
    data_neg = data_neg.apply(pd.to_numeric, errors='coerce')
    data_neg = data_neg.apply(scale_col)

    # reformat to long format, normalize, and combine pos & neg data
    data_pos['SampleID.abbrev'] = data_pos.index
    data_pos = pd.melt(data_pos, id_vars=['SampleID.abbrev'],
                       var_name='display_label', value_name='exp_value')

    data_neg['SampleID.abbrev'] = data_neg.index
    data_neg = pd.melt(data_neg, id_vars=['SampleID.abbrev'],
                       var_name='display_label', value_name='exp_value')

    data = pd.concat([data_pos, data_neg])
    # data['exp_value'] = scale_col(data['exp_value'])

    # extract sample IDs from labID column
    string1 = "BEAT_AML_PNL_"
    data['SampleID.abbrev'] = data['SampleID.abbrev'].apply(
        lambda st: st[st.find(string1) + len(string1):st.find("_M")])
    data['SampleID.abbrev'] = pd.to_numeric(data['SampleID.abbrev'], errors='coerce').astype(pd.Int16Dtype())

    # convert sample IDs to barcode IDs
    ex10Metadata_df = load_meta_for_ids()
    ex10Metadata_df = ex10Metadata_df[['SampleID.abbrev', 'Barcode.ID']]
    data = pd.merge(data, ex10Metadata_df)
    data = data.rename(columns={'Barcode.ID': 'sample_id'})

    # reformat data
    subset = data.loc[:, cols]
    subset['source'] = 'metabolomics'
    subset['label'] = subset.display_label + '_met_HILIC'
    subset['label'] = subset['label'].apply(lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    subset.to_csv(f_name)
    return subset


def prep_metabolomics_HILIC(confidence="high"):
    f_name = os.path.join(os.path.dirname(__file__), 'data/metabolomics_HILIC.csv')
    if os.path.exists(f_name):
        return pd.read_csv(f_name)

    data_pos, data_neg = load_and_process_data(confidence)
    data = pd.concat([data_pos, data_neg])
    data = extract_sample_ids(data)
    data = convert_sample_ids_to_barcode_ids(data)
    subset = reformat_data(data)
    subset.to_csv(f_name)

    return subset


def load_and_process_data(confidence):
    data_pos = load_and_process_excel(metabolomics_id, 0, 570, 131 if confidence == "high" else 0)
    data_neg = load_and_process_excel(metabolomics_id, 1, 336, 75 if confidence == "high" else 0)
    return data_pos, data_neg


def load_and_process_excel(id, sheet, drop_rows, additional_drop_rows):
    data = load_excel(id, sheet)
    data = data.iloc[:-(drop_rows + additional_drop_rows)]
    data = data.drop(columns=data.columns[data.columns.str.startswith('Blank_')])
    data = data.drop(columns=data.columns[data.columns.str.contains(
        'm/z|RT \[min\]|Tags|Standardized name|Super class|Main class| Sub class|Formula|Annot. DeltaMass \[ppm\]|Annotation MW|Reference Ion')])
    data = data.T
    data.columns = data.iloc[0]
    data = data[1:].apply(pd.to_numeric, errors='coerce').apply(scale_col)
    data['SampleID.abbrev'] = data.index
    data = pd.melt(data, id_vars=['SampleID.abbrev'], var_name='display_label', value_name='exp_value')
    return data


def extract_sample_ids(data):
    data['SampleID.abbrev'] = data['SampleID.abbrev'].apply(
        lambda st: st[st.find("BEAT_AML_PNL_") + len("BEAT_AML_PNL_"):st.find("_M")])
    data['SampleID.abbrev'] = pd.to_numeric(data['SampleID.abbrev'], errors='coerce').astype(pd.Int16Dtype())
    return data


def convert_sample_ids_to_barcode_ids(data):
    ex10Metadata_df = load_meta_for_ids()[['SampleID.abbrev', 'Barcode.ID']]
    data = pd.merge(data, ex10Metadata_df)
    data = data.rename(columns={'Barcode.ID': 'sample_id'})
    return data


def reformat_data(data):
    subset = data[['display_label', 'sample_id', 'exp_value']]
    subset['source'] = 'metabolomics'
    subset['label'] = subset.display_label + '_met_HILIC'
    subset['label'] = subset['label'].apply(lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    return subset


def prep_metabolomics_RP(confidence="high"):
    f_name = os.path.join(os.path.dirname(__file__), 'data/metabolomics_RP.csv')
    if os.path.exists(f_name):
        return pd.read_csv(f_name)
    cols = ['display_label', 'sample_id',
            'exp_value']

    # import HILIC pos & neg and drop extra rows & columns
    data_pos = load_excel(metabolomics_id, 2)
    data_pos = data_pos.iloc[:-280]  # drop unknowns
    data_pos = data_pos.drop(columns=['Blank_BEAT_AML_01_RP_Pos',
                                      'Blank_BEAT_AML_01_RP_Pos3',
                                      'Blank_BEAT_AML_01_RP_Pos2',
                                      'Blank_BEAT_AML_02_RP_Pos',
                                      'Blank_BEAT_AML_03_RP_Pos',
                                      'Blank_BEAT_AML_04_RP_Pos',
                                      'Blank_BEAT_AML_05_RP_Pos'])  # drop blanks
    data_pos = data_pos.drop(columns=['m/z', 'RT [min]', 'Tags',
                                      'Standardized name', 'Super class',
                                      'Main class', ' Sub class', 'Formula',
                                      'Annot. DeltaMass [ppm]',
                                      'Annotation MW', 'Reference Ion'])

    data_neg = load_excel(metabolomics_id, 3)
    data_neg = data_neg.iloc[:-163]  # drop unknowns
    data_neg = data_neg.drop(columns=['Blank_BEAT_AML_01_RP_Neg',
                                      'Blank_BEAT_AML_02_RP_Neg',
                                      'Blank_BEAT_AML_02_RP_Neg_',
                                      'Blank_BEAT_AML_03_RP_Neg',
                                      'Blank_BEAT_AML_04_RP_Neg',
                                      'Blank_BEAT_AML_05_RP_Neg'])  # drop blanks
    data_neg = data_neg.drop(columns=['m/z', 'RT [min]', 'Tags',
                                      'Standardized name', 'Super class',
                                      'Main class', ' Sub class', 'Formula',
                                      'Annot. DeltaMass [ppm]',
                                      'Annotation MW', 'Reference Ion'])

    # drop rows not meeting confidence threshold
    if confidence == "high":
        data_pos = data_pos.iloc[:-157]
        data_neg = data_neg.iloc[:-57]

    # normalize data
    data_pos = data_pos.T
    data_pos.columns = data_pos.iloc[0]
    data_pos = data_pos[1:]
    data_pos = data_pos.apply(pd.to_numeric, errors='coerce')
    data_pos = data_pos.apply(scale_col)

    data_neg = data_neg.T
    data_neg.columns = data_neg.iloc[0]
    data_neg = data_neg[1:]
    data_neg = data_neg.apply(pd.to_numeric, errors='coerce')
    data_neg = data_neg.apply(scale_col)

    # reformat to long format and combine pos & neg data
    data_pos['SampleID.abbrev'] = data_pos.index
    data_pos = pd.melt(data_pos, id_vars=['SampleID.abbrev'],
                       var_name='display_label', value_name='exp_value')

    data_neg['SampleID.abbrev'] = data_neg.index
    data_neg = pd.melt(data_neg, id_vars=['SampleID.abbrev'],
                       var_name='display_label', value_name='exp_value')

    data = pd.concat([data_pos, data_neg])
    # data['exp_value'] = scale_col(data['exp_value'])

    # extract sample IDs from labID column
    string1 = "BEAT_AML_PNL_"
    data['SampleID.abbrev'] = data['SampleID.abbrev'].apply(
        lambda st: st[st.find(string1) + len(string1):st.find("_M")])
    data['SampleID.abbrev'] = pd.to_numeric(data['SampleID.abbrev'], errors='coerce').astype(pd.Int16Dtype())

    # convert sample IDs to barcode IDs
    ex10Metadata_df = load_meta_for_ids()
    ex10Metadata_df = ex10Metadata_df[['SampleID.abbrev', 'Barcode.ID']]
    data = pd.merge(data, ex10Metadata_df)
    data = data.rename(columns={'Barcode.ID': 'sample_id'})

    # reformat data
    subset = data.loc[:, cols]
    subset['source'] = 'metabolomics'
    subset['label'] = subset.display_label + '_met_RP'
    subset['label'] = subset['label'].apply(lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    subset.to_csv(f_name)
    return subset


def prep_lipidomics():

    f_name = os.path.join(os.path.dirname(__file__), 'data/lipidomics.csv')
    if os.path.exists(f_name):
        return pd.read_csv(f_name)
    cols = ['display_label', 'sample_id', 'exp_value']

    # import pos & neg and drop extra columns
    data_pos = load_excel(lipidomics_id, 1)
    data_pos = data_pos.drop(columns=['CPTAC4_AML_BM_L_QC_01_Lumos_Pos_18Feb23_Crater-WCSH315305',
                                      'CPTAC4_AML_BM_L_QC_02_Lumos_Pos_18Feb23_Crater-WCSH315305',
                                      'CPTAC4_AML_BM_L_QC_03_Lumos_Pos_18Feb23_Crater-WCSH315305',
                                      'CPTAC4_AML_BM_L_QC_04_Lumos_Pos_18Feb23_Crater-WCSH315305',
                                      'CPTAC4_AML_BM_L_QC_05_Lumos_Pos_18Feb23_Crater-WCSH315305',
                                      'CPTAC4_AML_BM_L_QC_06_Lumos_Pos_18Feb23_Crater-WCSH315305',
                                      'CPTAC4_AML_BM_L_QC_07_Lumos_Pos_18Feb23_Crater-WCSH315305',
                                      'CPTAC4_AML_WB_L_QC_01_Lumos_Pos_18Feb23_Crater-WCSH315305',
                                      'CPTAC4_AML_WB_L_QC_02_Lumos_Pos_18Feb23_Crater-WCSH315305',
                                      'CPTAC4_AML_WB_L_QC_03_Lumos_Pos_18Feb23_Crater-WCSH315305',
                                      'CPTAC4_AML_WB_L_QC_04_Lumos_Pos_18Feb23_Crater-WCSH315305',
                                      'CPTAC4_AML_WB_L_QC_05_Lumos_Pos_18Feb23_Crater-WCSH315305',
                                      'CPTAC4_AML_WB_L_QC_06_Lumos_Pos_18Feb23_Crater-WCSH315305',
                                      'CPTAC4_AML_WB_L_QC_07_Lumos_Pos_18Feb23_Crater-WCSH315305'])  # drop CPTAC4
    data_pos = data_pos.drop(columns=['Alignment ID', 'Average Rt(min)',
                                      'Average Mz', 'Adduct type',
                                      'Reference m/z', 'Formula',
                                      'Ontology', 'MS/MS spectrum'])

    data_neg = load_excel(lipidomics_id, 0)
    data_neg = data_neg.drop(columns=['CPTAC4_AML_BM_L_QC_01_Lumos_Neg_22Feb23_Crater-WCSH315305',
                                      'CPTAC4_AML_BM_L_QC_02_Lumos_Neg_22Feb23_Crater-WCSH315305',
                                      'CPTAC4_AML_BM_L_QC_03_Lumos_Neg_22Feb23_Crater-WCSH315305',
                                      'CPTAC4_AML_BM_L_QC_04_Lumos_Neg_22Feb23_Crater-WCSH315305',
                                      'CPTAC4_AML_BM_L_QC_05_Lumos_Neg_22Feb23_Crater-WCSH315305',
                                      'CPTAC4_AML_BM_L_QC_06_Lumos_Neg_22Feb23_Crater-WCSH315305',
                                      'CPTAC4_AML_BM_L_QC_07_Lumos_Neg_22Feb23_Crater-WCSH315305',
                                      'CPTAC4_AML_WB_L_QC_01_Lumos_Neg_22Feb23_Crater-WCSH315305',
                                      'CPTAC4_AML_WB_L_QC_02_Lumos_Neg_22Feb23_Crater-WCSH315305',
                                      'CPTAC4_AML_WB_L_QC_03_Lumos_Neg_22Feb23_Crater-WCSH315305',
                                      'CPTAC4_AML_WB_L_QC_04_Lumos_Neg_22Feb23_Crater-WCSH315305',
                                      'CPTAC4_AML_WB_L_QC_05_Lumos_Neg_22Feb23_Crater-WCSH315305',
                                      'CPTAC4_AML_WB_L_QC_06_Lumos_Neg_22Feb23_Crater-WCSH315305',
                                      'CPTAC4_AML_WB_L_QC_07_Lumos_Neg_22Feb23_Crater-WCSH315305'])  # drop CPTAC4
    data_neg = data_neg.drop(columns=['Alignment ID', 'Average Rt(min)',
                                      'Average Mz', 'Adduct type',
                                      'Reference m/z', 'Formula',
                                      'Ontology', 'MS/MS spectrum'])

    # average across duplicate compound names
    data_pos = data_pos.groupby(['Metabolite name'], as_index=False).mean()
    data_neg = data_neg.groupby(['Metabolite name'], as_index=False).mean()

    # normalize data
    data_pos = data_pos.T
    data_pos.columns = data_pos.iloc[0]
    data_pos = data_pos[1:]
    data_pos = data_pos.apply(pd.to_numeric, errors='coerce')
    data_pos = data_pos.apply(scale_col)

    data_neg = data_neg.T
    data_neg.columns = data_neg.iloc[0]
    data_neg = data_neg[1:]
    data_neg = data_neg.apply(pd.to_numeric, errors='coerce')
    data_neg = data_neg.apply(scale_col)

    # reformat to long format and combine pos & neg data
    data_pos['SampleID.abbrev'] = data_pos.index
    data_pos = pd.melt(data_pos, id_vars=['SampleID.abbrev'],
                       var_name='display_label', value_name='exp_value')

    data_neg['SampleID.abbrev'] = data_neg.index
    data_neg = pd.melt(data_neg, id_vars=['SampleID.abbrev'],
                       var_name='display_label', value_name='exp_value')

    data = pd.concat([data_pos, data_neg])
    # data['exp_value'] = scale_col(data['exp_value'])

    # extract sample IDs from labID column
    string1 = "BEAT_AML_PNL_"
    data['SampleID.abbrev'] = data['SampleID.abbrev'].apply(
        lambda st: st[st.find(string1) + len(string1):st.find("_L")])
    data['SampleID.abbrev'] = pd.to_numeric(data['SampleID.abbrev'], errors='coerce').astype(pd.Int16Dtype())

    # convert sample IDs to barcode IDs
    ex10Metadata_df = load_meta_for_ids()
    ex10Metadata_df = ex10Metadata_df[['SampleID.abbrev', 'Barcode.ID']]
    data = pd.merge(data, ex10Metadata_df)
    data = data.rename(columns={'Barcode.ID': 'sample_id'})

    # reformat data
    subset = data.loc[:, cols]
    subset['source'] = 'lipidomics'
    subset['label'] = subset['display_label'] + '_lip'
    subset['label'] = subset['label'].apply(lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    subset.to_csv(f_name)
    return subset


def prep_phosph():
    file_path = os.path.join(os.path.dirname(__file__), 'data/phospho.csv')

    if os.path.exists(file_path):
        return pd.read_csv(file_path)

    phospho_data = load_table(phospho_id)
    phosph_subset = phospho_data[['Gene', 'SiteID', 'LogRatio', 'SampleID.full', 'Barcode.ID']]
    phosph_subset.rename(columns={
        'Gene': 'gene_symbol',
        'SiteID': 'label',
        'LogRatio': 'exp_value',
        'SampleID.full': 'sample_id_full',
        'Barcode.ID': 'sample_id',
    }, inplace=True)
    phosph_subset['source'] = 'phospho'

    phosph_subset.to_csv(file_path)

    return phosph_subset


def prep_acetyl():
    file_path = os.path.join(os.path.dirname(__file__), 'data/acetyl.csv')

    if os.path.exists(file_path):
        return pd.read_csv(file_path)

    acetyl_data = load_file(acetyl_id)
    acetyl_data['label'] = acetyl_data.index
    acetyl_data = pd.melt(acetyl_data, id_vars=['label'], var_name='SampleID.full', value_name='exp_value')
    acetyl_data['gene_symbol'] = acetyl_data['label'].str.split('-', expand=True)[0]

    ex10Metadata_df = load_meta_for_ids()[['SampleID.full', 'Barcode.ID']]
    acetyl_data = pd.merge(acetyl_data, ex10Metadata_df)
    acetyl_data = acetyl_data.rename(columns={'Barcode.ID': 'sample_id', 'SampleID.full': 'sample_id_full'})

    acetyl_subset = acetyl_data[['gene_symbol', 'label', 'exp_value', 'sample_id_full', 'sample_id']]
    acetyl_subset['source'] = 'acetyl'

    acetyl_subset.to_csv(file_path)

    return acetyl_subset


def prep_proteomics():
    f_name = os.path.join(os.path.dirname(__file__), 'data/global.csv')
    if os.path.exists(f_name):
        return pd.read_csv(f_name)

    proteomics_mapper = {
        'Gene': 'gene_symbol',
        'SiteID': 'label',
        'LogRatio': 'exp_value',
        'SampleID.full': 'sample_id_full',
        'Barcode.ID': 'sample_id',
    }
    global_data = load_table(global_id)
    # remove empty gene columns? Is this safe
    proteomics = global_data.loc[~global_data.Gene.isna(), :].copy()
    proteomics.rename(proteomics_mapper, axis=1, inplace=True)
    pho_cols = ['gene_symbol', 'exp_value',
                'sample_id_full', 'sample_id']
    proteomics = proteomics.loc[:, pho_cols]

    # add source and label column for MAGINE
    proteomics['label'] = proteomics.gene_symbol + '_prot'
    proteomics['source'] = 'proteomics'

    proteomics.to_csv(f_name)
    return proteomics


def load_drug_response():
    f_name = 'data/drug_response.csv'
    f_name = os.path.join(os.path.dirname(__file__), f_name)
    if os.path.exists(f_name):
        return pd.read_csv(f_name)
    response_data = load_table(drug_response_id)
    response_data.auc = response_data.auc.astype(float)
    response_data.aic = response_data.aic.astype(float)
    response_data.deviance = response_data.deviance.astype(float)

    # create this column as filter column
    response_data['new_col'] = response_data.proteomic_lab_id + '_' + response_data.inhibitor
    to_remove = []
    # If multiple measurements, remove if std(auc) >50 (seems to be skewed
    # to remove larger auc)
    for i, j in response_data.groupby(['proteomic_lab_id', 'inhibitor']):
        if j.shape[0] > 1:
            if j['auc'].std() > 50:
                to_remove.append('_'.join(i))

    response_data = response_data.loc[
        ~response_data.new_col.isin(to_remove)].copy()
    response_data = response_data.groupby(
        ['proteomic_lab_id', 'inhibitor']).mean().reset_index()
    response_data = response_data.loc[
        ~(
                (response_data.aic > 12) &
                (response_data.deviance > 2)
        )
    ].copy()
    new_auc = response_data[['proteomic_lab_id', 'inhibitor', 'auc']].copy()
    new_auc.rename({'proteomic_lab_id': 'sample_id'}, axis=1, inplace=True)
    if not os.path.exists(f_name):
        new_auc.to_csv(f_name)
    return new_auc


def load_mutations():
    """
    Loads WES data.

    Processes mutational status into two levels. First one is at the gene level,
    second one gene with amino acid level.


    Returns
    -------

    """
    f_name = os.path.join(os.path.dirname(__file__), 'data/wes.csv')
    if os.path.exists(f_name):
        return pd.read_csv(f_name)
    df = load_table(wes_id)
    df.rename(
        {'symbol': 'gene_symbol', 'labId': 'sample_id'},
        axis=1, inplace=True
    )
    df['exp_value'] = 1
    wes_gene_level = pd.pivot_table(
        df,
        index='sample_id',
        values='exp_value',
        columns='gene_symbol',
        fill_value=0
    )
    wes_gene_level = wes_gene_level.melt(ignore_index=False).reset_index()
    wes_gene_level['label'] = wes_gene_level.gene_symbol + '_mut'
    wes_gene_level['exp_value'] = wes_gene_level['value']
    wes_gene_level['source'] = 'wes'

    wes_aa_level = df.copy()
    wes_aa_level['label'] = wes_aa_level['hgvsp'].str.split(':p.').str.get(1)
    wes_aa_level['label'] = wes_aa_level['gene_symbol'] + '_' + wes_aa_level['label']

    wes_aa_level = pd.pivot_table(
        wes_aa_level,
        index='sample_id',
        values='exp_value',
        columns='label',
        fill_value=0
    )
    wes_aa_level = wes_aa_level.melt(ignore_index=False).reset_index()
    wes_aa_level['gene_symbol'] = wes_aa_level.label.str.split('_').str.get(0)
    wes_aa_level['source'] = 'wes_protein_level'
    wes_aa_level['exp_value'] = wes_aa_level['value']
    merged = pd.concat([wes_gene_level, wes_aa_level])

    merged.to_csv(f_name)
    return merged


def convert_to_matrix(flat_dataframe):
    df = pd.pivot_table(
        flat_dataframe,
        index='sample_id',
        values='exp_value',
        columns='label'
    )
    return df


class AMLData(object):
    def __init__(self):
        self.proteomics = prep_proteomics()
        self.phospho = prep_phosph()
        self.rna_seq = prep_rnaseq()
        self.functional = load_drug_response()
        self.wes = load_mutations()
        self.metabolomics_HILIC = prep_metabolomics_HILIC()
        self.metabolomics_RP = prep_metabolomics_RP()
        self.metabolomics = pd.concat([self.metabolomics_HILIC, self.metabolomics_RP])
        self.lipidomics = prep_lipidomics()
        self.acetyl = prep_acetyl()
        self.flat_data = pd.concat(
            [self.phospho, self.proteomics, self.rna_seq, self.wes,
             self.metabolomics, self.lipidomics, self.acetyl]
        )

        self.meta = add_cluster_plus_meta()
        self.meta = self.meta.join(load_cluster_pred())

        self.feature_names = self.flat_data['label'].unique().tolist()

        self._auc_table = pd.pivot_table(
            self.functional,
            index='sample_id', columns='inhibitor', values='auc'
        )
        self.drug_names = list(self._auc_table.columns.unique())
        self.auc_table = self._auc_table.join(self.meta, on='sample_id')
        # format for magine.ExperimentalData class
        self.flat_data.rename({'gene_symbol': 'identifier'}, axis=1, inplace=True)
        self.flat_data['species_type'] = 'gene'
        self._exp_data = None

    @property
    def exp_data(self):
        if self._exp_data is None:
            self._exp_data = ExperimentalData(self.flat_data)

        return self._exp_data

    def add_meta(self, pivoted_table):
        return self.meta.join(pivoted_table)

    def subset(self, source, with_meta=False):
        _subset = self.subset_flat(source)
        return self.add_meta(convert_to_matrix(_subset)) if with_meta else convert_to_matrix(_subset)

    def subset_flat(self, source):
        source = [source] if isinstance(source, str) else source
        return self.flat_data.loc[self.flat_data.source.isin(source)]

    def get_trainable_data(self, source, drug_name, new_format=False, drop_cols=True,  drop_rows=False):
        mol_data = self.subset(source)
        if isinstance(source, list):
            sample_ids = [set(self.__getattribute__(i).sample_id.values) for i in source]
            mol_data = mol_data.loc[list(set.intersection(*sample_ids).intersection(mol_data.index.values))]

        df_subset = mol_data.join(self.auc_table[drug_name]).dropna(subset=[drug_name])

        # require 50% of the data be present for any given column
        if drop_cols:
            df_subset.dropna(axis=1, thresh=int(df_subset.shape[0] * .50), inplace=True)
        # filter row/sample if missing half the feature values
        if drop_rows:
            df_subset.dropna(axis=0, thresh=int(df_subset.shape[1] * .50), inplace=True)

        return SampleByClusterDataSet(df_subset, drug_name) if new_format else df_subset


class SampleByClusterDataSet(object):
    def __init__(self,
                 df,
                 target_name,
                 ):
        self._df = df
        feat_names = list(set(self._df.columns.values))
        if target_name in feat_names:
            feat_names.remove(target_name)
        self.features = self._df[feat_names].copy()
        self.target = self._df[target_name].values * 1

    def train_test_split(self):

        x_train, x_test, y_train, y_test = train_test_split(
            self.features,
            self.target,
            test_size=0.2,
            shuffle=True,
            random_state=101,
        )

        return x_train, x_test, y_train, y_test

    def remove_features(self, feature_names):
        current_features = set(self.features.columns.values)
        self.features = self._df[
            current_features.difference(set(feature_names))]

    def require_features(self, feature_names):
        current_features = set(self.features.columns.values)
        fn = [i + '_rna' for i in feature_names]
        fn += [i + '_prot' for i in feature_names]
        self.features = self._df[current_features.intersection(set(fn))]

    def require_features_by_label(self, feature_names):
        current_features = set(self.features.columns.values)
        self.features = self._df[
            current_features.intersection(set(feature_names))]


def add_cluster_plus_meta():
    f_name = os.path.join(os.path.dirname(__file__), 'data/meta_labels.csv')
    if os.path.exists(f_name):
        return pd.read_csv(f_name, index_col='sample_id')
    clusters = load_file(clusters_id)
    del clusters['Barcode.ID']
    clusters.reset_index(inplace=True)
    clusters.rename({'index': 'sample_id'}, axis=1, inplace=True)

    summary = load_excel(clinical_summary_id)
    summary['sample_id'] = summary['labId']
    summary.set_index('sample_id', inplace=True)
    summ_cols = [
        'FLT3-ITDcalls',
        'NPM1calls',
        'cumulativeChemo',
        'overallSurvival'
    ]
    summary = summary[summ_cols].copy()
    rename = {
        'positive': True,
        'negative': False,
        'y': True,
        'n': False,
        np.nan: False
    }
    summary.replace(rename, inplace=True)
    merged = summary.join(clusters.set_index('sample_id'))

    merged.to_csv(f_name)
    return merged


def load_cluster_pred():
    f_name = os.path.join(os.path.dirname(__file__), 'data/cluster_pred.csv')
    if os.path.exists(f_name):
        return pd.read_csv(f_name, index_col=0)

    cluster_pred = load_file('syn30030154')
    cluster_pred.rename(columns={'Barcode.ID': 'sample_id'}, inplace=True)
    cluster_pred['Cluster'] = cluster_pred['Cluster'].str.split(' ').str[1].astype(int)
    cluster_pred.set_index('sample_id', inplace=True)

    cluster_pred.to_csv(f_name)
    return cluster_pred


if __name__ == '__main__':
    d = AMLData()
