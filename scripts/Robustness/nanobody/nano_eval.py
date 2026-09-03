import os
import subprocess
import sys
import argparse
from tqdm import tqdm
from abnumber import Chain
import numpy as np
import shutil
import pandas as pd

current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, current_dir)

from evaluation.Robustness.T20_eval import frame_main as tframemain
from utils.Robustness.misc import get_logger


def cal_fr_preservation(chain1, chain2):
    identity = 0
    fr_sum = 0
    align = chain1.align(chain2)
    for pos in align.positions:
        if not pos.is_in_cdr():
            a1, a2 = align[pos]
            if a1 == a2:
                identity += 1
            fr_sum += 1
    return identity / fr_sum


def cal_group_fr_germline_identity(df, scheme='imgt'):
    """计算Germline identity，返回带index的字典"""
    identity_results = []
    for idx in tqdm(df.index, desc="Calculating Germline Identity"):
        try:
            # 兼容不同的列名，优先读取 h_seq (T20结果), 如果没有则读取 hseq (原始数据)
            if 'h_seq' in df.columns:
                h_seq = df.iloc[idx]['h_seq']
            else:
                h_seq = df.iloc[idx]['hseq']
                
            if 'Raw_name' in df.columns:
                name = df.iloc[idx]['Raw_name']
            else:
                name = df.iloc[idx]['name']

            h_chain = Chain(h_seq, scheme=scheme)
            h_chain_graft = h_chain.graft_cdrs_onto_human_germline()
            fr_h_ratio = cal_fr_preservation(h_chain, h_chain_graft)
            identity_results.append({
                'Raw_name': name,
                'h_seq': h_seq,
                'germline_identity': fr_h_ratio
            })
        except Exception as e:
            # print(f"Error at index {idx}: {e}") # 减少刷屏，需要调试可打开
            continue
    return identity_results


def cal_mean(vh_dir, vhh_dir):
    vh_fpath = os.path.join(vh_dir, 'sample_nano_vh_abnativ_seq_scores.csv')
    vhh_fpath = os.path.join(vhh_dir, 'sample_nano_vhh_abnativ_seq_scores.csv')

    if not os.path.isfile(vh_fpath):
        raise FileNotFoundError(vh_fpath)
    if not os.path.isfile(vhh_fpath):
        raise FileNotFoundError(vhh_fpath)

    sample_vh_df = pd.read_csv(vh_fpath)
    sample_vhh_df = pd.read_csv(vhh_fpath)

    sample_vh_score = sample_vh_df['AbNatiV VH Score']
    sample_vhh_score = sample_vhh_df['AbNatiV VHH Score']

    ref_vh_score = 0.7378085839359757
    ref_vhh_score = 0.9143594023426274

    dev_vh_score = sample_vh_score.mean() - ref_vh_score
    
    return dev_vh_score, sample_vh_score.mean(), sample_vhh_score.mean()


def run_abnativ(exec_path, nat_type, input_fa, out_dir, oid, extra_args=None):
    cmd = [
        exec_path, 'score',
        '-nat', nat_type,
        '-i', input_fa,
        '-odir', out_dir,
        '-oid', oid,
        '-align'
    ]
    if extra_args:
        cmd.extend(extra_args)

    print('RUN:', ' '.join(cmd))

    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    out, err = p.communicate()

    if p.returncode != 0:
        raise RuntimeError(f'abnativ {nat_type} failed: {err}')


def main(root_path):
    root_path = os.path.abspath(root_path)
    base_dir = os.path.dirname(root_path)

    logger = get_logger('sample', base_dir, log_name='eval_log.txt')

    exec_path = shutil.which("abnativ")
    if exec_path is None:
        raise RuntimeError("abnativ not found in PATH")

    input_fa_path = os.path.join(base_dir, 'sample_identity.fa')

    # ============ VH/VHH 目录结构 ============
    vh_base_dir = os.path.join(base_dir, 'VH')
    vhh_base_dir = os.path.join(base_dir, 'VHH')
    os.makedirs(vh_base_dir, exist_ok=True)
    os.makedirs(vhh_base_dir, exist_ok=True)

    # Sample目录
    output_vh_dir = os.path.join(vh_base_dir, 'sample_nano_vh')
    output_vhh_dir = os.path.join(vhh_base_dir, 'sample_nano_vhh')
    os.makedirs(output_vh_dir, exist_ok=True)
    os.makedirs(output_vhh_dir, exist_ok=True)

    # Raw目录
    raw_vh_dir = os.path.join(vh_base_dir, 'raw_nano_vh')
    raw_vhh_dir = os.path.join(vhh_base_dir, 'raw_nano_vhh')
    os.makedirs(raw_vh_dir, exist_ok=True)
    os.makedirs(raw_vhh_dir, exist_ok=True)

    # ============ 读取数据 ============
    sample_result_df = pd.read_csv(root_path)
    sample_nano_df = sample_result_df[sample_result_df['Specific'] == 'nano'].reset_index(drop=True)
    sample_human_df = sample_result_df[sample_result_df['Specific'] == 'humanization'].reset_index(drop=True)

    # ============ 准备Raw的FASTA文件 ============
    raw_nano_fasta_path = os.path.join(base_dir, 'raw_nano_identity.fa')
    if not os.path.exists(raw_nano_fasta_path):
        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord
        from Bio import SeqIO
        
        seq_records = []
        for i, row in sample_nano_df.iterrows():
            # 优先读 hseq (原始csv列名)，兼容 h_seq
            seq_val = row.get('hseq', row.get('h_seq')) 
            seq_record = SeqRecord(
                Seq(seq_val),
                id=f"v_nano_{i}",
                description='VH'
            )
            seq_records.append(seq_record)
        
        with open(raw_nano_fasta_path, 'w') as f:
            SeqIO.write(seq_records, f, 'fasta')

    # ============ VH/VHH 评分 - Sample ============
    vh_csv = os.path.join(output_vh_dir, 'sample_nano_vh_abnativ_seq_scores.csv')
    vhh_csv = os.path.join(output_vhh_dir, 'sample_nano_vhh_abnativ_seq_scores.csv')

    if not os.path.isfile(vh_csv):
        run_abnativ(exec_path, 'VH', input_fa_path, output_vh_dir, 'sample_nano_vh')

    if not os.path.isfile(vhh_csv):
        run_abnativ(
            exec_path, 'VHH',
            input_fa_path,
            output_vhh_dir,
            'sample_nano_vhh',
            extra_args=['-isVHH']
        )

    # ============ VH/VHH 评分 - Raw ============
    raw_vh_csv = os.path.join(raw_vh_dir, 'raw_nano_vh_abnativ_seq_scores.csv')
    raw_vhh_csv = os.path.join(raw_vhh_dir, 'raw_nano_vhh_abnativ_seq_scores.csv')

    if not os.path.isfile(raw_vh_csv):
        run_abnativ(exec_path, 'VH', raw_nano_fasta_path, raw_vh_dir, 'raw_nano_vh')

    if not os.path.isfile(raw_vhh_csv):
        run_abnativ(
            exec_path, 'VHH',
            raw_nano_fasta_path,
            raw_vhh_dir,
            'raw_nano_vhh',
            extra_args=['-isVHH']
        )

    # ============ 计算VH/VHH平均分 ============
    sample_vh_df = pd.read_csv(vh_csv)
    sample_vhh_df = pd.read_csv(vhh_csv)
    sample_vh_mean = sample_vh_df['AbNatiV VH Score'].mean()
    sample_vhh_mean = sample_vhh_df['AbNatiV VHH Score'].mean()
    
    raw_vh_df = pd.read_csv(raw_vh_csv)
    raw_vhh_df = pd.read_csv(raw_vhh_csv)
    raw_vh_mean = raw_vh_df['AbNatiV VH Score'].mean()
    raw_vhh_mean = raw_vhh_df['AbNatiV VHH Score'].mean()

    # ============ T20 评分 (修正版：不改列名) ============
    t20_base_dir = os.path.join(base_dir, 'T20')
    os.makedirs(t20_base_dir, exist_ok=True)

    # --- 1. Raw nano T20 ---
    raw_frame_t20_output = os.path.join(t20_base_dir, 'raw_frame_t20_score.csv')
    raw_nano_csv_path = os.path.join(base_dir, 'temp_raw_nano_for_t20.csv')
    if os.path.isfile(raw_frame_t20_output):
        raw_frame_t20_df = pd.read_csv(raw_frame_t20_output)
    else:
        # 构造 Raw 输入
        sample_nano_df_for_t20 = sample_nano_df.copy()

        # T20_eval.py 会筛选 Specific == 'humanization'，所以这里必须改 Specific
        sample_nano_df_for_t20['Specific'] = 'humanization'
        sample_nano_df_for_t20.to_csv(raw_nano_csv_path, index=False)

        raw_t20_generated_fpath = tframemain(raw_nano_csv_path)
        raw_frame_t20_df = pd.read_csv(raw_t20_generated_fpath)
        raw_frame_t20_df.to_csv(raw_frame_t20_output, index=False)

        # 清理 Raw 中间文件
        if os.path.exists(raw_t20_generated_fpath):
            os.remove(raw_t20_generated_fpath)
        if os.path.exists(raw_nano_csv_path):
            os.remove(raw_nano_csv_path)
    raw_frame_t20 = raw_frame_t20_df['h_score'].mean()

    # --- 2. Humanized Sample T20 ---
    sample_nano_csv_path = os.path.join(base_dir, 'temp_sample_nano_for_t20.csv')
    
    # 构造 Sample 输入
    sample_human_df_for_t20 = sample_human_df.copy()
    
    # 【修正点】：不执行 rename，保持 'hseq' 和 'name' 
    # 只要确保 Specific 是 humanization 即可（你的原始数据里本身就是，这里保险起见再赋一次值也可以）
    sample_human_df_for_t20['Specific'] = 'humanization'
    
    sample_frame_t20_output = os.path.join(t20_base_dir, 'sample_frame_t20_score.csv')
    if os.path.isfile(sample_frame_t20_output):
        sample_t20_frame_df = pd.read_csv(sample_frame_t20_output)
    else:
        # 保存真正的人源化序列临时文件并运行 T20
        sample_human_df_for_t20.to_csv(sample_nano_csv_path, index=False)
        sample_t20_generated_fpath = tframemain(sample_nano_csv_path)
        sample_t20_frame_df = pd.read_csv(sample_t20_generated_fpath)
        sample_t20_frame_df.to_csv(sample_frame_t20_output, index=False)

        # 清理 Sample 中间文件
        if os.path.exists(sample_t20_generated_fpath):
            os.remove(sample_t20_generated_fpath)
        if os.path.exists(sample_nano_csv_path):
            os.remove(sample_nano_csv_path)
    sample_frame_t20 = sample_t20_frame_df['h_score'].mean()

    # ============ Germline Identity ============
    germline_base_dir = os.path.join(base_dir, 'Germline_identity')
    os.makedirs(germline_base_dir, exist_ok=True)

    # 1. Raw数据的Germline identity
    # raw_frame_t20_df 是 T20 输出的结果，列名固定为 ['Raw_name', 'h_score', 'h_gene', 'h_seq']
    # 所以可以直接用
    raw_germline_output = os.path.join(germline_base_dir, 'raw_germline_identity.csv')
    if os.path.isfile(raw_germline_output):
        raw_germline_df = pd.read_csv(raw_germline_output)
    else:
        raw_identity_list = cal_group_fr_germline_identity(raw_frame_t20_df)
        raw_germline_df = pd.DataFrame(raw_identity_list)
        raw_germline_df.to_csv(raw_germline_output, index=False)
    
    # 2. Sample数据的Germline identity
    # sample_human_df_for_t20 是我们的原始 Sample 输入（列名 hseq, name）
    # cal_group_fr_germline_identity 里面做了兼容判断，所以这里也能直接跑
    sample_germline_output = os.path.join(germline_base_dir, 'sample_germline_identity.csv')
    if os.path.isfile(sample_germline_output):
        sample_germline_df = pd.read_csv(sample_germline_output)
    else:
        sample_identity_list = cal_group_fr_germline_identity(sample_human_df_for_t20)
        sample_germline_df = pd.DataFrame(sample_identity_list)
        sample_germline_df.to_csv(sample_germline_output, index=False)

    # ============ 计算平均值 ============
    raw_germline_mean = raw_germline_df['germline_identity'].mean()
    sample_germline_mean = sample_germline_df['germline_identity'].mean()

    # ============ 日志输出 ============
    logger.info('=' * 60)
    logger.info('T20 Framework Score:')
    logger.info(f'  Raw Frame t20 score: {raw_frame_t20:.4f}')
    logger.info(f'  Sample Frame t20 score: {sample_frame_t20:.4f}')
    logger.info(f'  Improve Frame t20 score: {sample_frame_t20 - raw_frame_t20:.4f}')
    
    logger.info('=' * 60)
    logger.info('VH Score:')
    logger.info(f'  Raw VH score: {raw_vh_mean:.4f}')
    logger.info(f'  Sample VH score: {sample_vh_mean:.4f}')
    logger.info(f'  Improve VH score: {sample_vh_mean - raw_vh_mean:.4f}')
    
    logger.info('=' * 60)
    logger.info('VHH Score:')
    logger.info(f'  Raw VHH score: {raw_vhh_mean:.4f}')
    logger.info(f'  Sample VHH score: {sample_vhh_mean:.4f}')
    logger.info(f'  Improve VHH score: {sample_vhh_mean - raw_vhh_mean:.4f}')
    
    logger.info('=' * 60)
    logger.info('Germline Identity:')
    logger.info(f'  Raw Germline identity: {raw_germline_mean:.4f}')
    logger.info(f'  Sample Germline identity: {sample_germline_mean:.4f}')
    logger.info(f'  Improve Germline identity: {sample_germline_mean - raw_germline_mean:.4f}')
    
    logger.info('=' * 60)
    logger.info('File Locations:')
    logger.info(f'  VH: {vh_base_dir}')
    logger.info(f'  VHH: {vhh_base_dir}')
    logger.info(f'  T20: {t20_base_dir}')
    logger.info(f'  Germline_identity: {germline_base_dir}')
    logger.info('=' * 60)


if __name__ == '__main__':
    current_ld_library_path = os.getenv("LD_LIBRARY_PATH", "")
    os.environ['LD_LIBRARY_PATH'] = '/mnt/wucy/miniconda3/envs/abnativ/lib:' + current_ld_library_path

    parser = argparse.ArgumentParser(description='Evaluate HuDiff nanobody/heavy-chain robustness outputs.')
    parser.add_argument('sample_result', help='Path to sample_humanization_result.csv')
    args = parser.parse_args()
    main(args.sample_result)
