# This codes are at present not used. This codes where used for cleaning the COVID dataset
import pandas as pd


def filter_columns_with_pattern(datafr, pattern):
    columns_to_ignore = list(filter(lambda x: x.endswith(pattern), list(datafr)))
    upd_datafr = datafr.drop(columns = columns_to_ignore)
    return(upd_datafr)

def empty_fields(datafr):
   #pd.set_option('max_columns', None)
   for row in datafr:
       datafr.fillna(value=0,inplace=True)
       filled_df = datafr.astype(dtype=int,errors='ignore')
   return(filled_df)

def create_indicator_pattern(datafr):
    for row in datafr:
       pd.set_option('max_columns', None)
       pattern_datafr = datafr.assign(pattern = datafr['S1_School closing'].map(str) + datafr['S2_Workplace closing'].map(str) + datafr['S4_Close public transport'].map(str) + datafr['S5_Public information campaigns'].map(str) + datafr['S6_Restrictions on internal movement'].map(str) + datafr['S7_International travel controls'].map(str))     # + datafr['S8_Fiscal measures'].map(str) + datafr['S9_Monetary measures'].map(str) + datafr['S10_Emergency investment in health care'].map(str) + datafr['S11_Investment in Vaccines'].map(str))
    return(pattern_datafr)

class cleaning_oxcovid_data:
    oxcovid_xlsx = pd.read_excel('../xlsx-csv/OxCGRT_Download_latest_data.xlsx')
    cleaned_datafr = filter_columns_with_pattern(oxcovid_xlsx, "_Notes")
    cleaned_datafr = filter_columns_with_pattern(oxcovid_xlsx, "_IsGeneral")
    cleaned_datafr = empty_fields(cleaned_datafr)
    cleaned_datafr = create_indicator_pattern(cleaned_datafr)
    print(cleaned_datafr)
