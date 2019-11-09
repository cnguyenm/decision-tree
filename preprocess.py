import pandas

# df: pandas.DataFrame = pandas.read_csv('./adult.data', header=None)

def pre_process(df: pandas.DataFrame) -> pandas.DataFrame:
    # original columns
    cols = [0, 1, 2, 3, 4, 5, 6, 7]

    # add new cols to df
    # convert all attrib to binary
    for attrib in cols:
        uniques = df[attrib].unique()
        for value in uniques:

            # temp_col = col of T/F
            temp_col = (df[attrib] == value)
            df[value] = temp_col

    # drop old columns
    df.rename(columns={8: 'salaryLevel'}, inplace=True)
    df = df.drop(columns=cols)

    # use index_col for faster search
    label1 = ' >50K'
    index_col = (df['salaryLevel'] == label1)
    df['index_col'] = index_col

    return df

