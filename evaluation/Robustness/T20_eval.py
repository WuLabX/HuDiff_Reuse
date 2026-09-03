import os

import requests
import sys
import pandas as pd
import time
from tqdm import tqdm
import re
from abnumber import Chain
import concurrent.futures

T20_REGEX = re.compile('<td>T20 Score:</td><td>([0-9.]+)</td>')
def get_t20_online(seq, region=1):
    if region == 1:
        chain = Chain(seq, scheme='imgt')
        chain_type = 'vh' if chain.chain_type == 'H' else ('vl' if chain.chain_type == 'L' else 'vk')
    elif region == 2:
        chain_type = 'vh'
    else:
        raise ValueError('Region type do not appropriate.')

    html = None
    for retry in range(5):
        url = f'https://sam.curiaglobal.com/t20/cgi-bin/blast.py?chain={chain_type}&region={region}&output=3&seqs={seq}'
        try:
            request = requests.get(url, timeout=30)
            if request.ok:
                html = request.text
                break
        except Exception as e:
            print(e)
        except:
            continue
        time.sleep(0.5 + retry * 5)
        print('Retry', retry+1)
    if not html:
        sys.exit(1)
    # print(html)
    matches = T20_REGEX.findall(html)
    time.sleep(1)
    if not matches:
        print(html)
        # raise ValueError(f'Error calling url {url}')
        return None, None
    return float(matches[0]), chain_type

def get_pair_data_t20(h_seq, l_seq, region=1):
    h_score, h_type = get_t20_online(h_seq, region)
    l_score, l_type = get_t20_online(l_seq, region)
    # print(h_score, l_score)
    return [h_score, h_type, l_score, l_type, h_seq, l_seq]


def get_one_chain_framework_t20(h_seq, region=2):
    h_score, h_type = get_t20_online(h_seq, region)
    return [h_score, h_type, h_seq]


def process_line(line):
    h_seq = line[1]['hseq']
    l_seq = line[1]['lseq']
    name = [line[1]['name']]
    data = []
    for retry in range(3):
        try:
            data = get_pair_data_t20(h_seq, l_seq)
            if len(data) > 2:
                break
        except:
            time.sleep(5)
            # continue
    # if data is not None:
    print(data)
    if len(data) > 2:
        new_data = name + data
        new_line_df = pd.DataFrame([new_data], columns=['Raw_name', 'h_score', 'h_gene', 'l_score', 'l_gene', 'h_seq', 'l_seq'])
        return new_line_df
    else:
        return None

def process_one_seq_and_frame_line(line):
    h_seq = line[1]['hseq']
    name = [line[1]['name']]
    # name = ['vhhseq' + str(line[0])]
    data = []
    for retry in range(3):
        try:
            data = get_one_chain_framework_t20(h_seq, region=2)
            if len(data) > 2:
                break
        except:
            time.sleep(5)
            continue
    # if data is not None:
    print(data)
    if len(data) > 2:
        new_data = name + data
        new_line_df = pd.DataFrame([new_data], columns=['Raw_name', 'h_score', 'h_gene', 'h_seq'])
        return new_line_df
    else:
        return None


def make_t20_raw_names_unique(sample_human_df):
    """T20 checkpoints use Raw_name as a key, so duplicate sample names need row ids."""
    sample_human_df = sample_human_df.copy()
    raw_names = sample_human_df['name'].astype(str)
    if raw_names.duplicated(keep=False).any():
        sample_human_df['name'] = [
            f'{name}__row{idx:04d}' for idx, name in enumerate(raw_names)
        ]
    return sample_human_df


def frame_main(sample_fpath=None):
    if sample_fpath is None:
        sample_fpath = '/sample_humanization_result.csv'

    
    print(sample_fpath)
    save_fpath = os.path.join(os.path.dirname(sample_fpath), 'sample_frame_t20_score.csv')
    if os.path.exists(save_fpath):
        return save_fpath
    progress_fpath = save_fpath + '.progress.csv'

    sample_df = pd.read_csv(sample_fpath)

    sample_human_df = sample_df[sample_df['Specific'] == 'humanization'].reset_index(drop=True)
    sample_human_df = make_t20_raw_names_unique(sample_human_df)
    valid_mask = ~sample_human_df['hseq'].astype(str).str.contains(
        r'[^ACDEFGHIKLMNPQRSTVWY]', regex=True
    )
    invalid_names = sample_human_df.loc[~valid_mask, 'name'].astype(str).tolist()
    if invalid_names:
        print(f'Skipping {len(invalid_names)} T20-incompatible sequences: {invalid_names}')
    valid_human_df = sample_human_df[valid_mask].copy()
    if os.path.exists(progress_fpath):
        progress_df = pd.read_csv(progress_fpath)
    else:
        progress_df = pd.DataFrame(columns=['Raw_name', 'h_score', 'h_gene', 'h_seq'])

    # Reuse scores for identical generated sequences across seeds, modes and
    # temperatures.  HuAb348 has substantial cross-run sequence duplication.
    cache_fpath = os.getenv('T20_CACHE_FPATH')
    if cache_fpath and os.path.exists(cache_fpath):
        cache_df = pd.read_csv(cache_fpath).drop_duplicates('h_seq', keep='last')
    else:
        cache_df = pd.DataFrame(columns=['h_seq', 'h_score', 'h_gene'])

    completed_names = set(progress_df['Raw_name'].astype(str))
    if not cache_df.empty:
        cached_scores = cache_df.set_index('h_seq')[['h_score', 'h_gene']].to_dict('index')
        cached_rows = []
        for _, row in valid_human_df.iterrows():
            if str(row['name']) in completed_names or row['hseq'] not in cached_scores:
                continue
            score = cached_scores[row['hseq']]
            cached_rows.append({
                'Raw_name': row['name'],
                'h_score': score['h_score'],
                'h_gene': score['h_gene'],
                'h_seq': row['hseq'],
            })
        if cached_rows:
            progress_df = pd.concat([progress_df, pd.DataFrame(cached_rows)], ignore_index=True)
            progress_df.to_csv(progress_fpath, index=False)

    completed_names = set(progress_df['Raw_name'].astype(str))
    pending_df = valid_human_df[~valid_human_df['name'].astype(str).isin(completed_names)]

    # The public T20 endpoint throttles larger request bursts.  Keep the
    # concurrency conservative so long HuAb348 batches complete reliably.
    max_workers = int(os.getenv('T20_MAX_WORKERS', '2'))
    # Threads avoid forking active TLS state, which can otherwise leave
    # requests to the public endpoint hanging for several minutes.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_one_seq_and_frame_line, pending_df.iterrows())
        failed = []
        for (idx, row), result in zip(pending_df.iterrows(), tqdm(results, total=len(pending_df))):
            if result is None:
                failed.append(idx)
                continue
            progress_df = pd.concat([progress_df, result], ignore_index=True)
            # Durable per-sequence checkpoint: an interrupted online run can
            # resume without repeating scores already returned by the server.
            progress_df.to_csv(progress_fpath, index=False)
            if cache_fpath:
                cache_df = pd.concat(
                    [cache_df, result[['h_seq', 'h_score', 'h_gene']]],
                    ignore_index=True,
                ).drop_duplicates('h_seq', keep='last')
                cache_df.to_csv(cache_fpath, index=False)

    print(failed)
    if failed or len(progress_df) != len(valid_human_df):
        raise RuntimeError(
            f'T20 scoring incomplete: {len(progress_df)}/{len(valid_human_df)} valid sequences; '
            f'failed indices: {failed}'
        )
    save_frame_t20_df = progress_df[['Raw_name', 'h_score', 'h_gene', 'h_seq']]
    save_frame_t20_df.to_csv(save_fpath, index=False)
    os.remove(progress_fpath)
    return save_fpath


def main(sample_fpath=None):
    """
    Gather the T20 score from the website.
    :return:
    """
    if sample_fpath is None:
        sample_fpath = '/humanization_pair_data_filter.csv'
                   
    save_fpath = os.path.join(os.path.dirname(sample_fpath), 'sample_t20_score.csv')
    if os.path.exists(save_fpath):
        return save_fpath
    progress_fpath = save_fpath + '.progress.csv'

    sample_df = pd.read_csv(sample_fpath)

    sample_human_df = sample_df[sample_df['Specific'] == 'humanization'].reset_index(drop=True)
    sample_human_df = make_t20_raw_names_unique(sample_human_df)
    valid_h = ~sample_human_df['hseq'].astype(str).str.contains(
        r'[^ACDEFGHIKLMNPQRSTVWY]', regex=True
    )
    valid_l = ~sample_human_df['lseq'].astype(str).str.contains(
        r'[^ACDEFGHIKLMNPQRSTVWY]', regex=True
    )
    valid_human_df = sample_human_df[valid_h & valid_l].copy()
    invalid_names = sample_human_df.loc[~(valid_h & valid_l), 'name'].astype(str).tolist()
    if invalid_names:
        print(f'Skipping {len(invalid_names)} T20-incompatible pairs: {invalid_names}')

    columns = ['Raw_name', 'h_score', 'h_gene', 'l_score', 'l_gene', 'h_seq', 'l_seq']
    if os.path.exists(progress_fpath) and os.path.getsize(progress_fpath) > 0:
        progress_df = pd.read_csv(progress_fpath)
    else:
        progress_df = pd.DataFrame(columns=columns)

    cache_fpath = os.getenv('T20_PAIR_CACHE_FPATH') or os.getenv('T20_CACHE_FPATH')
    cache_columns = ['seq', 'score', 'gene']
    if cache_fpath and os.path.exists(cache_fpath):
        cache_df = pd.read_csv(cache_fpath).drop_duplicates('seq', keep='last')
    else:
        cache_df = pd.DataFrame(columns=cache_columns)

    completed_names = set(progress_df['Raw_name'].astype(str))
    if not cache_df.empty:
        cached_scores = cache_df.set_index('seq')[['score', 'gene']].to_dict('index')
        cached_rows = []
        for _, row in valid_human_df.iterrows():
            raw_name = str(row['name'])
            h_seq = row['hseq']
            l_seq = row['lseq']
            if raw_name in completed_names:
                continue
            if h_seq not in cached_scores or l_seq not in cached_scores:
                continue
            h_score = cached_scores[h_seq]
            l_score = cached_scores[l_seq]
            cached_rows.append({
                'Raw_name': row['name'],
                'h_score': h_score['score'],
                'h_gene': h_score['gene'],
                'l_score': l_score['score'],
                'l_gene': l_score['gene'],
                'h_seq': h_seq,
                'l_seq': l_seq,
            })
        if cached_rows:
            progress_df = pd.concat([progress_df, pd.DataFrame(cached_rows)], ignore_index=True)
            progress_df.to_csv(progress_fpath, index=False)

    completed_names = set(progress_df['Raw_name'].astype(str))
    pending_df = valid_human_df[~valid_human_df['name'].astype(str).isin(completed_names)]

    max_workers = int(os.getenv('T20_PAIR_MAX_WORKERS', '2'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_line, pending_df.iterrows())
        failed = []
        for (idx, row), result in zip(pending_df.iterrows(), tqdm(results, total=len(pending_df))):
            if result is None:
                failed.append(idx)
                continue
            progress_df = pd.concat([progress_df, result], ignore_index=True)
            progress_df.to_csv(progress_fpath, index=False)
            if cache_fpath:
                cache_rows = [
                    {
                        'seq': result.iloc[0]['h_seq'],
                        'score': result.iloc[0]['h_score'],
                        'gene': result.iloc[0]['h_gene'],
                    },
                    {
                        'seq': result.iloc[0]['l_seq'],
                        'score': result.iloc[0]['l_score'],
                        'gene': result.iloc[0]['l_gene'],
                    },
                ]
                cache_df = pd.concat([cache_df, pd.DataFrame(cache_rows)], ignore_index=True)
                cache_df = cache_df.drop_duplicates('seq', keep='last')
                cache_df.to_csv(cache_fpath, index=False)

    print(failed)
    if failed or len(progress_df) != len(valid_human_df):
        raise RuntimeError(
            f'T20 scoring incomplete: {len(progress_df)}/{len(valid_human_df)} valid pairs; '
            f'failed indices: {failed}'
        )
    save_t20_df = progress_df[columns]
    save_t20_df.to_csv(save_fpath, index=False)
    if os.path.exists(progress_fpath):
        os.remove(progress_fpath)
    return save_fpath


if __name__ == '__main__':
    main()
