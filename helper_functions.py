def lump(df, col, limit=10):
    '''
    Lumps values from .value_counts() below `limit` into one category called
    "Other". Inspired by R's fct_lump.
    '''
    s = df[col].value_counts()
    df = s.to_frame()
    df.loc[df[col] <= limit, "LumpedCategory"] = "Other"
    df.loc[df[col] > limit, "LumpedCategory"] = df.loc[df[col] > limit, "LumpedCategory"].index
    lumped_df = df.groupby("LumpedCategory").sum().sort_values(col, ascending=False)
    return lumped_df