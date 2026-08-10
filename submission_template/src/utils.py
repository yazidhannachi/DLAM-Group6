import hashlib

def hash_config(config_path):
    '''
    Computes hash from config file.
    '''
    with open(config_path, 'rb') as f:
        hash_func = hashlib.new("shake128")
        hash_func.update(f.read())
        file_hash = hash_func.hexdigest(length=4)
    return file_hash

def extract_initial_history(train_df, seq_len):

    history_df = (
        train_df.groupby('series_id', group_keys=False)
        .apply(lambda x: x.tail(seq_len))
        .sort_values(by='series_id')
    )
    return history_df

