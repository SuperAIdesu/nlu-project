import pandas as pd
from config import seed

def sample_by_author(df, seed=None):
    """
    Sample a different question by the same author from the dataframe, excluding the current row. 
    Additionally, retains the category and author of the sampled question.

    Args:
    - df (pd.DataFrame): DataFrame containing (question, answer, response, author, category) columns.

    Returns:
    - df (pd.DataFrame): Modified dataframe with new columns: retain_question, retain_answer, retain_category, retain_author.
    """
    np.random.seed(seed)
    
    def get_sample_by_author(row):
        filtered_df = df[(df['author'] == row['author']) & (df.index != row.name)]
        if not filtered_df.empty:
            sample = filtered_df.sample(1, random_state=seed)
            return (sample['question'].iloc[0], sample['answer'].iloc[0], sample['category'].iloc[0], sample['author'].iloc[0])
        else:
            return ("No other question available", "No answer available", row['category'], row['author'])
    
    results = df.apply(get_sample_by_author, axis=1, result_type='expand')
    df[['retain_question', 'retain_answer', 'retain_category', 'retain_author']] = results
    return df

def sample_by_category(df, seed=None):
    """
    Sample a different question from the same category in the dataframe, excluding the current row.
    Additionally, retains the category and author of the sampled question.

    Args:
    - df (pd.DataFrame): DataFrame containing (question, answer, response, author, category) columns.

    Returns:
    - df (pd.DataFrame): Modified dataframe with new columns: retain_question, retain_answer, retain_category, retain_author.
    """
    np.random.seed(seed)
    
    def get_sample_by_category(row):
        filtered_df = df[(df['category'] == row['category']) & (df.index != row.name)]
        if not filtered_df.empty:
            sample = filtered_df.sample(1, random_state=seed)
            return (sample['question'].iloc[0], sample['answer'].iloc[0], sample['category'].iloc[0], sample['author'].iloc[0])
        else:
            return ("No other question available", "No answer available", row['category'], row['author'])
    
    results = df.apply(get_sample_by_category, axis=1, result_type='expand')
    df[['retain_question', 'retain_answer', 'retain_category', 'retain_author']] = results
    return df

def sample_random_question(df, seed=None):
    """
    Sample a different question from the dataframe randomly, excluding the current row.
    Additionally, retains the category and author of the sampled question.

    Args:
    - df (pd.DataFrame): DataFrame containing (question, answer, response, author, category) columns.

    Returns:
    - df (pd.DataFrame): Modified dataframe with new columns: retain_question, retain_category, retain_author.
    """
    np.random.seed(seed)
    
    def get_random_sample(row):
        filtered_df = df[df.index != row.name]
        if not filtered_df.empty:
            sample = filtered_df.sample(1, random_state=seed)
            return (sample['question'].iloc[0], sample['answer'].iloc[0], sample['category'].iloc[0], sample['author'].iloc[0])
        else:
            return ("No other question available", "No answer available", "No category available", "No author available")
    
    results = df.apply(get_random_sample, axis=1, result_type='expand')
    df[['retain_question', 'retain_answer', 'retain_category', 'retain_author']] = results
    return df