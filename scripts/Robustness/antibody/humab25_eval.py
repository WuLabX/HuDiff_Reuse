"""
抗体人源化评估脚本 (Antibody Humanization Evaluation for Humab25)

评估以下指标：
1. AbNatiV VH Score (重链人源性)
2. AbNatiV VL Score (轻链人源性)
3. T20 Score (免疫原性)
4. OASis Score (BioPhi工具)
5. Germline Identity (FR区域与人类Germline一致性)
6. Preservation (序列保持率)
7. Mutation Precision (突变精确度)
8. ABLSTM Score (可开发性)

对比三个数据源：
- Mouse: 原始鼠源序列
- Exp: 实验人源化序列 (experimental_humanized.csv)
- Sample: 你的人源化序列 (sample_humanization_result.csv)
"""
import os, subprocess, shutil, tempfile
from tqdm import tqdm
from abnumber import Chain
import numpy as np
import pandas as pd
import sys
import pickle

# 添加 ABLSTM 目录到 Python 路径（用于导入 ablstm 模块）
ABLSTM_DIR = '/mnt/wucy/WUCHUYA/ABLSTM'
if ABLSTM_DIR not in sys.path:
    sys.path.insert(0, ABLSTM_DIR)

current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, current_dir)
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from evaluation.Robustness.T20_eval import main as t20_main
from utils.Robustness.misc import get_logger

BIOPHI_DIR = '/mnt/wucy/WUCHUYA/BioPhi'
OASIS_DB_PATH = os.path.join(BIOPHI_DIR, 'OASis_9mers_v1.db')
ABNATIV_DIR = '/mnt/wucy/WUCHUYA/AbNatiV'

# ============ 辅助函数 ============

def seqs_to_fasta(seqs, names, save_path):
    seq_records = [SeqRecord(Seq(seq), id=name, description='') for seq, name in zip(seqs, names)]
    with open(save_path, 'w') as f:
        SeqIO.write(seq_records, f, 'fasta')

def seqs_to_paired_fasta(h_seqs, l_seqs, names, save_path):
    seq_records = []
    for h_seq, l_seq, name in zip(h_seqs, l_seqs, names):
        seq_records.append(SeqRecord(Seq(h_seq), id=f"{name}_VH", description=''))
        seq_records.append(SeqRecord(Seq(l_seq), id=f"{name}_VL", description=''))
    with open(save_path, 'w') as f:
        SeqIO.write(seq_records, f, 'fasta')

def run_abnativ(exec_path, nat_type, input_fa, out_dir, oid):
    cmd = [exec_path, 'score', '-nat', nat_type, '-i', input_fa, '-odir', out_dir, '-oid', oid, '-align']
    print('RUN:', ' '.join(cmd))
    env = os.environ.copy()
    env['PYTHONPATH'] = ABNATIV_DIR + os.pathsep + env.get('PYTHONPATH', '')
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    out, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError(f'abnativ {nat_type} failed: {err}')

def run_oasis(biophi_exec, input_fa, oasis_db, output_xlsx):
    cmd = [biophi_exec, 'oasis', input_fa, '--oasis-db', oasis_db, '--output', output_xlsx]
    print('RUN:', ' '.join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError(f'BioPhi OASis failed: {err}')

def detect_light_chain_type(seq):
    try:
        chain = Chain(seq, scheme='imgt')
        return 'VKappa' if chain.chain_type == 'K' else 'VLambda'
    except:
        return 'VKappa'

def cal_fr_preservation(chain1, chain2):
    identity = fr_sum = 0
    try:
        align = chain1.align(chain2)
        for pos in align.positions:
            if not pos.is_in_cdr():
                a1, a2 = align[pos]
                if a1 == a2:
                    identity += 1
                fr_sum += 1
        return identity / fr_sum if fr_sum > 0 else 0
    except:
        return None

def cal_all_preservation(chain1, chain2):
    identity = total = 0
    try:
        align = chain1.align(chain2)
        for pos in align.positions:
            a1, a2 = align[pos]
            if a1 == a2:
                identity += 1
            total += 1
        return identity / total if total > 0 else 0
    except:
        return None

def cal_vernier_preservation(chain1, chain2):
    identity = vernier_sum = 0
    try:
        align = chain1.align(chain2)
        for pos in align.positions:
            if pos.is_in_vernier():
                a1, a2 = align[pos]
                if a1 == a2:
                    identity += 1
                vernier_sum += 1
        return identity / vernier_sum if vernier_sum > 0 else 0
    except:
        return None

def cal_fr_mutation_precision(expchain, parental, test):
    share = only = 0
    align = expchain.align(parental, test)
    for pos in align.positions:
        if not pos.is_in_cdr():
            exp, mou, aa = align[pos]
            if exp != mou or aa != mou:
                if exp == aa:
                    share += 1
                elif aa != mou:
                    only += 1
    return share / (share + only) if share + only > 0 else None

def cal_vernier_mutation_precision(expchain, parental, test):
    share = only = 0
    align = expchain.align(parental, test)
    for pos in align.positions:
        if pos.is_in_vernier():
            exp, mou, aa = align[pos]
            if exp != mou or aa != mou:
                if exp == aa:
                    share += 1
                elif aa != mou:
                    only += 1
    return share / (share + only) if share + only > 0 else None

def cal_germline_identity_single(seq, scheme='imgt'):
    try:
        chain = Chain(seq, scheme=scheme)
        chain_graft = chain.graft_cdrs_onto_human_germline()
        return cal_fr_preservation(chain, chain_graft)
    except:
        return None

def cal_group_fr_germline_identity(df, h_col='hseq', l_col='lseq', name_col='name'):
    results = []
    for idx in tqdm(df.index, desc="Germline Identity"):
        try:
            h_seq, l_seq = df.iloc[idx][h_col], df.iloc[idx][l_col]
            name = df.iloc[idx][name_col] if name_col in df.columns else f"sample_{idx}"
            h_gi = cal_germline_identity_single(h_seq)
            l_gi = cal_germline_identity_single(l_seq)
            results.append({'name': name, 'h_germline_identity': h_gi, 'l_germline_identity': l_gi})
        except:
            continue
    return results

def cal_group_fr_precision(exp_df, mou_df, sample_df, scheme='kabat'):
    h_list, l_list = [], []
    for idx in tqdm(exp_df.index):
        exp_h, exp_l = exp_df.iloc[idx]['h_seq'], exp_df.iloc[idx]['l_seq']
        mou_h, mou_l = mou_df.iloc[idx]['h_seq'], mou_df.iloc[idx]['l_seq']
        sap_h, sap_l = sample_df.iloc[idx]['hseq'], sample_df.iloc[idx]['lseq']
        exp_h_c, exp_l_c = Chain(exp_h, scheme=scheme), Chain(exp_l, scheme=scheme)
        mou_h_c, mou_l_c = Chain(mou_h, scheme=scheme), Chain(mou_l, scheme=scheme)
        sap_h_c, sap_l_c = Chain(sap_h, scheme=scheme), Chain(sap_l, scheme=scheme)
        h_r = cal_fr_mutation_precision(exp_h_c, mou_h_c, sap_h_c)
        l_r = cal_fr_mutation_precision(exp_l_c, mou_l_c, sap_l_c)
        if h_r is not None:
            h_list.append(h_r)
        if l_r is not None:
            l_list.append(l_r)
    return h_list, l_list

def cal_group_vernier_precision(exp_df, mou_df, sample_df, scheme='kabat'):
    h_list, l_list = [], []
    for idx in tqdm(exp_df.index):
        exp_h, exp_l = exp_df.iloc[idx]['h_seq'], exp_df.iloc[idx]['l_seq']
        mou_h, mou_l = mou_df.iloc[idx]['h_seq'], mou_df.iloc[idx]['l_seq']
        sap_h, sap_l = sample_df.iloc[idx]['hseq'], sample_df.iloc[idx]['lseq']
        exp_h_c, exp_l_c = Chain(exp_h, scheme=scheme), Chain(exp_l, scheme=scheme)
        mou_h_c, mou_l_c = Chain(mou_h, scheme=scheme), Chain(mou_l, scheme=scheme)
        sap_h_c, sap_l_c = Chain(sap_h, scheme=scheme), Chain(sap_l, scheme=scheme)
        h_r = cal_vernier_mutation_precision(exp_h_c, mou_h_c, sap_h_c)
        l_r = cal_vernier_mutation_precision(exp_l_c, mou_l_c, sap_l_c)
        if h_r is not None:
            h_list.append(h_r)
        if l_r is not None:
            l_list.append(l_r)
    return h_list, l_list

def cal_group_all_perservation(human_df, mouse_df, scheme='imgt', idx_type='lab'):
    """计算序列保持率（组）"""
    if idx_type == 'lab':
        h_idx, l_idx = 'h_seq', 'l_seq'
        m_h_idx, m_l_idx = 'h_seq', 'l_seq'
    else:
        h_idx, l_idx = 'hseq', 'lseq'
        m_h_idx, m_l_idx = 'hseq', 'lseq'
    
    all_list, vernier_list = [], []
    for idx in tqdm(human_df.index):
        h_h, h_l = human_df.iloc[idx][h_idx], human_df.iloc[idx][l_idx]
        m_h, m_l = mouse_df.iloc[idx][m_h_idx], mouse_df.iloc[idx][m_l_idx]
        h_h_c, h_l_c = Chain(h_h, scheme=scheme), Chain(h_l, scheme=scheme)
        m_h_c, m_l_c = Chain(m_h, scheme=scheme), Chain(m_l, scheme=scheme)
        all_list.append([cal_all_preservation(h_h_c, m_h_c), cal_all_preservation(h_l_c, m_l_c)])
        vernier_list.append([cal_vernier_preservation(h_h_c, m_h_c), cal_vernier_preservation(h_l_c, m_l_c)])
    return all_list, vernier_list

def seq_trans_to_aho(sequences):
    """将序列转换为AHO格式用于ABLSTM评估"""
    from anarci import anarci
    import re
    data = [(f'{i}', seq) for i, seq in enumerate(sequences)]
    h_results = anarci(data, scheme='aho', output=False)
    h_seq_results = h_results[0]
    aho_seq_list = []
    for seq_list in h_seq_results:
        re_seq = seq_list[0][0]
        str_re_seq = str(re_seq)
        matches = re.findall(r"'([A-Z\-])'", str_re_seq)
        aho_seq = '-' + ''.join(matches)
        if len(aho_seq) != 150:
            pad_count = 150 - len(aho_seq)
            aho_seq = aho_seq + '-' * pad_count
        aho_seq_list.append(aho_seq)
    return aho_seq_list

def run_ablstm_eval(sequences):
    """运行ABLSTM评估"""
    from ablstm import ModelLSTM
    aho_list = seq_trans_to_aho(sequences)
    
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w') as tmp:
        tmp_fpath = tmp.name
        for seq in aho_list:
            tmp.write(seq + '\n')
    
    try:
        model_data_path = '/mnt/wucy/WUCHUYA/ABLSTM/saved_models/tmp/model_tmp.npy'
        pred_model = ModelLSTM(embedding_dim=64, hidden_dim=64, device='cpu', gapped=True, fixed_len=True)
        pred_model.load(fn=model_data_path)
        h_score = pred_model.eval(tmp_fpath)
        os.remove(tmp_fpath)
        return h_score
    except Exception as e:
        os.remove(tmp_fpath)
        raise e

# ============ 主函数 ============

def main(root_path):
    root_path = os.path.abspath(root_path)
    base_dir = os.path.dirname(root_path)
    logger = get_logger('humab25_eval', base_dir, log_name='eval_log.txt')
    logger.info('=' * 60)
    logger.info('Humab25 Antibody Humanization Evaluation')
    logger.info('=' * 60)
    
    abnativ_conda = os.path.expanduser('/mnt/wucy/miniconda3/envs/abnativ/bin/abnativ')
    abnativ_exec = abnativ_conda if os.path.isfile(abnativ_conda) else shutil.which("abnativ")
    abnativ_local = os.path.join(ABNATIV_DIR, 'bin', 'abnativ')
    if abnativ_exec is None and os.path.isfile(abnativ_local):
        abnativ_exec = abnativ_local
    biophi_exec = shutil.which("biophi")
    oasis_db = OASIS_DB_PATH if os.path.exists(OASIS_DB_PATH) else None
    
    # 读取 Sample 数据
    sample_df = pd.read_csv(root_path)
    mouse_df = sample_df[sample_df['Specific'] == 'mouse'].reset_index(drop=True)
    human_df = sample_df[sample_df['Specific'] == 'humanization'].reset_index(drop=True)
    logger.info(f'Mouse: {len(mouse_df)}, Humanized: {len(human_df)}')
    
    if len(human_df) == 0:
        logger.error('No humanization sequences!')
        return
    
    # 读取 Exp (实验人源化) 数据
    lab_human = os.path.join(current_dir, 'data/antibody_eval_data/Humab25_data/experimental_humanized.csv')
    lab_mouse = os.path.join(current_dir, 'data/antibody_eval_data/Humab25_data/parental_mouse.csv')
    exp_df = None
    if os.path.isfile(lab_human) and os.path.isfile(lab_mouse):
        exp_df = pd.read_csv(lab_human)
        logger.info(f'Experimental humanized: {len(exp_df)}')
    
    light_type = detect_light_chain_type(human_df.iloc[0]['lseq'])
    logger.info(f'Light chain: {light_type}')
    
    dirs = ['VH', 'VL', 'T20', 'Germline_identity', 'OASis', 'Preservation', 'Mutation_precision', 'ABLSTM']
    dir_paths = {d: os.path.join(base_dir, d) for d in dirs}
    for d in dir_paths.values():
        os.makedirs(d, exist_ok=True)
    
    # ============ 准备 FASTA 文件 ============
    # Mouse FASTA
    mouse_h_fa = os.path.join(base_dir, 'mouse_heavy.fa')
    mouse_l_fa = os.path.join(base_dir, 'mouse_light.fa')
    mouse_paired_fa = os.path.join(base_dir, 'mouse_paired.fa')
    
    if len(mouse_df) > 0:
        seqs_to_fasta(mouse_df['hseq'].tolist(), [f"mouse_{i}" for i in range(len(mouse_df))], mouse_h_fa)
        seqs_to_fasta(mouse_df['lseq'].tolist(), [f"mouse_{i}" for i in range(len(mouse_df))], mouse_l_fa)
        seqs_to_paired_fasta(mouse_df['hseq'].tolist(), mouse_df['lseq'].tolist(), [f"mouse_{i}" for i in range(len(mouse_df))], mouse_paired_fa)
    
    # Sample (Humanization) FASTA
    human_h_fa = os.path.join(base_dir, 'sample_heavy.fa')
    human_l_fa = os.path.join(base_dir, 'sample_light.fa')
    human_paired_fa = os.path.join(base_dir, 'sample_identity.fa')
    
    seqs_to_fasta(human_df['hseq'].tolist(), [f"human_{i}" for i in range(len(human_df))], human_h_fa)
    seqs_to_fasta(human_df['lseq'].tolist(), [f"human_{i}" for i in range(len(human_df))], human_l_fa)
    seqs_to_paired_fasta(human_df['hseq'].tolist(), human_df['lseq'].tolist(), [f"human_{i}" for i in range(len(human_df))], human_paired_fa)
    
    # Exp FASTA (如果存在)
    exp_h_fa = os.path.join(base_dir, 'exp_heavy.fa')
    exp_l_fa = os.path.join(base_dir, 'exp_light.fa')
    exp_paired_fa = os.path.join(base_dir, 'exp_paired.fa')
    if exp_df is not None:
        seqs_to_fasta(exp_df['h_seq'].tolist(), [f"exp_{i}" for i in range(len(exp_df))], exp_h_fa)
        seqs_to_fasta(exp_df['l_seq'].tolist(), [f"exp_{i}" for i in range(len(exp_df))], exp_l_fa)
        seqs_to_paired_fasta(exp_df['h_seq'].tolist(), exp_df['l_seq'].tolist(), [f"exp_{i}" for i in range(len(exp_df))], exp_paired_fa)
    
    # ============ 1. AbNatiV VH 评分 ============
    logger.info('=' * 60)
    logger.info('1. AbNatiV VH Scoring...')
    
    # 创建子目录
    mouse_vh_dir = os.path.join(dir_paths['VH'], 'mouse_vh')
    exp_vh_dir = os.path.join(dir_paths['VH'], 'exp_vh')
    sample_vh_dir = os.path.join(dir_paths['VH'], 'sample_vh')
    os.makedirs(mouse_vh_dir, exist_ok=True)
    os.makedirs(exp_vh_dir, exist_ok=True)
    os.makedirs(sample_vh_dir, exist_ok=True)
    
    mouse_vh_csv = os.path.join(mouse_vh_dir, 'mouse_vh_abnativ_seq_scores.csv')
    exp_vh_csv = os.path.join(exp_vh_dir, 'exp_vh_abnativ_seq_scores.csv')
    sample_vh_csv = os.path.join(sample_vh_dir, 'sample_vh_abnativ_seq_scores.csv')
    
    # Mouse VH
    if abnativ_exec and len(mouse_df) > 0 and not os.path.isfile(mouse_vh_csv):
        try:
            run_abnativ(abnativ_exec, 'VH', mouse_h_fa, mouse_vh_dir, 'mouse_vh')
        except Exception as e:
            logger.error(f'Mouse VH failed: {e}')
    
    # Exp VH
    if abnativ_exec and exp_df is not None and not os.path.isfile(exp_vh_csv):
        try:
            run_abnativ(abnativ_exec, 'VH', exp_h_fa, exp_vh_dir, 'exp_vh')
        except Exception as e:
            logger.error(f'Exp VH failed: {e}')
    
    # Sample VH
    if abnativ_exec and not os.path.isfile(sample_vh_csv):
        try:
            run_abnativ(abnativ_exec, 'VH', human_h_fa, sample_vh_dir, 'sample_vh')
        except Exception as e:
            logger.error(f'Sample VH failed: {e}')
    
    # 读取结果
    mouse_vh_mean = exp_vh_mean = sample_vh_mean = None
    if os.path.isfile(mouse_vh_csv):
        df = pd.read_csv(mouse_vh_csv)
        mouse_vh_mean = df['AbNatiV VH Score'].mean()
        logger.info(f'  Mouse VH: {mouse_vh_mean:.4f}')
    if os.path.isfile(exp_vh_csv):
        df = pd.read_csv(exp_vh_csv)
        exp_vh_mean = df['AbNatiV VH Score'].mean()
        logger.info(f'  Exp VH: {exp_vh_mean:.4f}')
        if mouse_vh_mean:
            logger.info(f'  Exp VH Improvement vs Mouse: {exp_vh_mean - mouse_vh_mean:.4f}')
    if os.path.isfile(sample_vh_csv):
        df = pd.read_csv(sample_vh_csv)
        sample_vh_mean = df['AbNatiV VH Score'].mean()
        logger.info(f'  Sample VH: {sample_vh_mean:.4f}')
        if mouse_vh_mean:
            logger.info(f'  Sample VH Improvement vs Mouse: {sample_vh_mean - mouse_vh_mean:.4f}')
    
    # 保存 summary
    pd.DataFrame({
        'Type': ['Mouse', 'Exp', 'Sample'],
        'VH_Score_Mean': [mouse_vh_mean, exp_vh_mean, sample_vh_mean],
        'VH_Improvement_vs_Mouse': [None, exp_vh_mean - mouse_vh_mean if (mouse_vh_mean and exp_vh_mean) else None, 
                                     sample_vh_mean - mouse_vh_mean if (mouse_vh_mean and sample_vh_mean) else None]
    }).to_csv(os.path.join(dir_paths['VH'], 'vh_summary.csv'), index=False)
    
    # ============ 2. AbNatiV VL 评分 ============
    logger.info('=' * 60)
    logger.info(f'2. AbNatiV {light_type} Scoring...')
    
    # 创建子目录
    mouse_vl_dir = os.path.join(dir_paths['VL'], 'mouse_vl')
    exp_vl_dir = os.path.join(dir_paths['VL'], 'exp_vl')
    sample_vl_dir = os.path.join(dir_paths['VL'], 'sample_vl')
    os.makedirs(mouse_vl_dir, exist_ok=True)
    os.makedirs(exp_vl_dir, exist_ok=True)
    os.makedirs(sample_vl_dir, exist_ok=True)
    
    vl_col = f'AbNatiV {light_type} Score'
    mouse_vl_csv = os.path.join(mouse_vl_dir, 'mouse_vl_abnativ_seq_scores.csv')
    exp_vl_csv = os.path.join(exp_vl_dir, 'exp_vl_abnativ_seq_scores.csv')
    sample_vl_csv = os.path.join(sample_vl_dir, 'sample_vl_abnativ_seq_scores.csv')
    
    # Mouse VL
    if abnativ_exec and len(mouse_df) > 0 and not os.path.isfile(mouse_vl_csv):
        try:
            run_abnativ(abnativ_exec, light_type, mouse_l_fa, mouse_vl_dir, 'mouse_vl')
        except Exception as e:
            logger.error(f'Mouse VL failed: {e}')
    
    # Exp VL
    if abnativ_exec and exp_df is not None and not os.path.isfile(exp_vl_csv):
        try:
            run_abnativ(abnativ_exec, light_type, exp_l_fa, exp_vl_dir, 'exp_vl')
        except Exception as e:
            logger.error(f'Exp VL failed: {e}')
    
    # Sample VL
    if abnativ_exec and not os.path.isfile(sample_vl_csv):
        try:
            run_abnativ(abnativ_exec, light_type, human_l_fa, sample_vl_dir, 'sample_vl')
        except Exception as e:
            logger.error(f'Sample VL failed: {e}')
    
    # 读取结果
    mouse_vl_mean = exp_vl_mean = sample_vl_mean = None
    if os.path.isfile(mouse_vl_csv):
        df = pd.read_csv(mouse_vl_csv)
        mouse_vl_mean = df[vl_col].mean()
        logger.info(f'  Mouse {light_type}: {mouse_vl_mean:.4f}')
    if os.path.isfile(exp_vl_csv):
        df = pd.read_csv(exp_vl_csv)
        exp_vl_mean = df[vl_col].mean()
        logger.info(f'  Exp {light_type}: {exp_vl_mean:.4f}')
        if mouse_vl_mean:
            logger.info(f'  Exp {light_type} Improvement vs Mouse: {exp_vl_mean - mouse_vl_mean:.4f}')
    if os.path.isfile(sample_vl_csv):
        df = pd.read_csv(sample_vl_csv)
        sample_vl_mean = df[vl_col].mean()
        logger.info(f'  Sample {light_type}: {sample_vl_mean:.4f}')
        if mouse_vl_mean:
            logger.info(f'  Sample {light_type} Improvement vs Mouse: {sample_vl_mean - mouse_vl_mean:.4f}')
    
    # 保存 summary
    pd.DataFrame({
        'Type': ['Mouse', 'Exp', 'Sample'],
        'VL_Score_Mean': [mouse_vl_mean, exp_vl_mean, sample_vl_mean],
        'VL_Improvement_vs_Mouse': [None, exp_vl_mean - mouse_vl_mean if (mouse_vl_mean and exp_vl_mean) else None,
                                     sample_vl_mean - mouse_vl_mean if (mouse_vl_mean and sample_vl_mean) else None]
    }).to_csv(os.path.join(dir_paths['VL'], 'vl_summary.csv'), index=False)
    
    # ============ 3. T20 评分 ============
    logger.info('=' * 60)
    logger.info('3. T20 Scoring...')
    
    mouse_t20_csv = os.path.join(dir_paths['T20'], 'mouse_t20_score.csv')
    exp_t20_csv = os.path.join(dir_paths['T20'], 'exp_t20_score.csv')
    sample_t20_csv = os.path.join(dir_paths['T20'], 'sample_t20_score.csv')
    
    def run_t20_eval(df, output_csv, prefix):
        """运行 T20 评估"""
        try:
            # 临时修改 DataFrame
            df_copy = df.copy()
            
            # 处理列名映射（T20 期望 hseq/lseq，但 Exp 数据是 h_seq/l_seq）
            if 'h_seq' in df_copy.columns and 'hseq' not in df_copy.columns:
                df_copy = df_copy.rename(columns={'h_seq': 'hseq', 'l_seq': 'lseq'})
            if 'Raw_name' in df_copy.columns and 'name' not in df_copy.columns:
                df_copy = df_copy.rename(columns={'Raw_name': 'name'})
            
            # T20 只处理 humanization 序列，添加 Specific 列
            df_copy['Specific'] = 'humanization'
            # 添加 name 列（如果不存在）
            if 'name' not in df_copy.columns:
                df_copy['name'] = [f'{prefix}_{i}' for i in range(len(df_copy))]
            
            # 临时输入文件
            tmp_input = os.path.join(dir_paths['T20'], f'tmp_{prefix}_input.csv')
            df_copy.to_csv(tmp_input, index=False)
            
            # 预期的输出文件路径 (t20_main 保存的位置)
            t20_output_dir = dir_paths['T20']
            t20_output = os.path.join(t20_output_dir, 'sample_t20_score.csv')
            
            # 如果输出已存在，先备份
            if os.path.isfile(t20_output):
                backup = os.path.join(t20_output_dir, f'sample_t20_score_backup_{prefix}.csv')
                shutil.move(t20_output, backup)
            
            # 调用 T20
            t20_main(tmp_input)
            
            # 移动结果到目标位置
            if os.path.isfile(t20_output):
                shutil.move(t20_output, output_csv)
                logger.info(f'  {prefix} T20 saved to: {output_csv}')
                
                # 恢复备份（如果有）
                backup = os.path.join(t20_output_dir, f'sample_t20_score_backup_{prefix}.csv')
                if os.path.isfile(backup):
                    shutil.move(backup, t20_output)
            else:
                logger.warning(f'  {prefix} T20 no output file generated')
            
            # 清理临时输入
            if os.path.isfile(tmp_input):
                os.remove(tmp_input)
                
            return output_csv if os.path.isfile(output_csv) else None
        except Exception as e:
            logger.error(f'{prefix} T20 failed: {e}')
            # 清理临时文件
            for f in os.listdir(t20_output_dir):
                if f.startswith('sample_t20_score_backup'):
                    os.remove(os.path.join(t20_output_dir, f))
            return None
    
    # Mouse T20
    if len(mouse_df) > 0:
        run_t20_eval(mouse_df, mouse_t20_csv, 'mouse')
    
    # Exp T20
    if exp_df is not None:
        run_t20_eval(exp_df, exp_t20_csv, 'exp')
    
    # Sample T20
    run_t20_eval(human_df, sample_t20_csv, 'sample')
    
    # 读取结果
    mouse_h_t20 = mouse_l_t20 = None
    exp_h_t20 = exp_l_t20 = None
    sample_h_t20 = sample_l_t20 = None
    
    if os.path.isfile(mouse_t20_csv):
        df = pd.read_csv(mouse_t20_csv)
        mouse_h_t20, mouse_l_t20 = df['h_score'].mean(), df['l_score'].mean()
        logger.info(f'  Mouse T20 H: {mouse_h_t20:.2f}, L: {mouse_l_t20:.2f}')
    
    if os.path.isfile(exp_t20_csv):
        df = pd.read_csv(exp_t20_csv)
        exp_h_t20, exp_l_t20 = df['h_score'].mean(), df['l_score'].mean()
        logger.info(f'  Exp T20 H: {exp_h_t20:.2f}, L: {exp_l_t20:.2f}')
        if mouse_h_t20:
            logger.info(f'  Exp T20 Improvement vs Mouse H: {exp_h_t20 - mouse_h_t20:.2f}, L: {exp_l_t20 - mouse_l_t20:.2f}')
    
    if os.path.isfile(sample_t20_csv):
        df = pd.read_csv(sample_t20_csv)
        sample_h_t20, sample_l_t20 = df['h_score'].mean(), df['l_score'].mean()
        logger.info(f'  Sample T20 H: {sample_h_t20:.2f}, L: {sample_l_t20:.2f}')
        if mouse_h_t20:
            logger.info(f'  Sample T20 Improvement vs Mouse H: {sample_h_t20 - mouse_h_t20:.2f}, L: {sample_l_t20 - mouse_l_t20:.2f}')
    
    # 保存 summary
    pd.DataFrame({
        'Type': ['Mouse', 'Exp', 'Sample'],
        'T20_H_Mean': [mouse_h_t20, exp_h_t20, sample_h_t20],
        'T20_L_Mean': [mouse_l_t20, exp_l_t20, sample_l_t20],
        'T20_H_Improve_vs_Mouse': [None, exp_h_t20 - mouse_h_t20 if (mouse_h_t20 and exp_h_t20) else None,
                                    sample_h_t20 - mouse_h_t20 if (mouse_h_t20 and sample_h_t20) else None],
        'T20_L_Improve_vs_Mouse': [None, exp_l_t20 - mouse_l_t20 if (mouse_l_t20 and exp_l_t20) else None,
                                    sample_l_t20 - mouse_l_t20 if (mouse_l_t20 and sample_l_t20) else None]
    }).to_csv(os.path.join(dir_paths['T20'], 't20_summary.csv'), index=False)
    
    # ============ 4. OASis 评分 ============
    logger.info('=' * 60)
    logger.info('4. OASis Scoring...')
    
    if biophi_exec and oasis_db:
        mouse_oasis_csv = os.path.join(dir_paths['OASis'], 'mouse_oasis.csv')
        exp_oasis_csv = os.path.join(dir_paths['OASis'], 'exp_oasis.csv')
        sample_oasis_csv = os.path.join(dir_paths['OASis'], 'sample_oasis.csv')
        
        # Mouse OASis
        if len(mouse_df) > 0 and not os.path.isfile(mouse_oasis_csv):
            try:
                xlsx = mouse_oasis_csv.replace('.csv', '.xlsx')
                run_oasis(biophi_exec, mouse_paired_fa, oasis_db, xlsx)
                if os.path.isfile(xlsx):
                    pd.read_excel(xlsx, sheet_name='OASis Curves', index_col=0).to_csv(mouse_oasis_csv)
                    os.remove(xlsx)
            except Exception as e:
                logger.error(f'Mouse OASis failed: {e}')
        
        # Exp OASis
        if exp_df is not None and not os.path.isfile(exp_oasis_csv):
            try:
                xlsx = exp_oasis_csv.replace('.csv', '.xlsx')
                run_oasis(biophi_exec, exp_paired_fa, oasis_db, xlsx)
                if os.path.isfile(xlsx):
                    pd.read_excel(xlsx, sheet_name='OASis Curves', index_col=0).to_csv(exp_oasis_csv)
                    os.remove(xlsx)
            except Exception as e:
                logger.error(f'Exp OASis failed: {e}')
        
        # Sample OASis
        if not os.path.isfile(sample_oasis_csv):
            try:
                xlsx = sample_oasis_csv.replace('.csv', '.xlsx')
                run_oasis(biophi_exec, human_paired_fa, oasis_db, xlsx)
                if os.path.isfile(xlsx):
                    pd.read_excel(xlsx, sheet_name='OASis Curves', index_col=0).to_csv(sample_oasis_csv)
                    os.remove(xlsx)
            except Exception as e:
                logger.error(f'Sample OASis failed: {e}')
        
        # 读取结果
        mouse_oasis = exp_oasis = sample_oasis = None
        if os.path.isfile(mouse_oasis_csv):
            df = pd.read_csv(mouse_oasis_csv, index_col=0)
            mouse_oasis = df['50%'].mean()
            logger.info(f'  Mouse OASis 50%: {mouse_oasis:.4f}')
        if os.path.isfile(exp_oasis_csv):
            df = pd.read_csv(exp_oasis_csv, index_col=0)
            exp_oasis = df['50%'].mean()
            logger.info(f'  Exp OASis 50%: {exp_oasis:.4f}')
            if mouse_oasis:
                logger.info(f'  Exp OASis Improvement vs Mouse: {exp_oasis - mouse_oasis:.4f}')
        if os.path.isfile(sample_oasis_csv):
            df = pd.read_csv(sample_oasis_csv, index_col=0)
            sample_oasis = df['50%'].mean()
            logger.info(f'  Sample OASis 50%: {sample_oasis:.4f}')
            if mouse_oasis:
                logger.info(f'  Sample OASis Improvement vs Mouse: {sample_oasis - mouse_oasis:.4f}')
        
        # 保存 summary
        pd.DataFrame({
            'Type': ['Mouse', 'Exp', 'Sample'],
            'OASis_50pct_Mean': [mouse_oasis, exp_oasis, sample_oasis],
            'OASis_Improve_vs_Mouse': [None, exp_oasis - mouse_oasis if (mouse_oasis and exp_oasis) else None,
                                        sample_oasis - mouse_oasis if (mouse_oasis and sample_oasis) else None]
        }).to_csv(os.path.join(dir_paths['OASis'], 'oasis_summary.csv'), index=False)
    else:
        logger.warning('OASis skipped (BioPhi or DB not available)')
    
    # ============ 5. Germline Identity 评分 ============
    logger.info('=' * 60)
    logger.info('5. Germline Identity Scoring...')
    
    mouse_gi_csv = os.path.join(dir_paths['Germline_identity'], 'mouse_germline_identity.csv')
    exp_gi_csv = os.path.join(dir_paths['Germline_identity'], 'exp_germline_identity.csv')
    sample_gi_csv = os.path.join(dir_paths['Germline_identity'], 'sample_germline_identity.csv')
    
    # Mouse GI
    if len(mouse_df) > 0 and not os.path.isfile(mouse_gi_csv):
        pd.DataFrame(cal_group_fr_germline_identity(mouse_df)).to_csv(mouse_gi_csv, index=False)
    
    # Exp GI
    if exp_df is not None and not os.path.isfile(exp_gi_csv):
        pd.DataFrame(cal_group_fr_germline_identity(exp_df, h_col='h_seq', l_col='l_seq', name_col='Raw_name')).to_csv(exp_gi_csv, index=False)
    
    # Sample GI
    if not os.path.isfile(sample_gi_csv):
        pd.DataFrame(cal_group_fr_germline_identity(human_df)).to_csv(sample_gi_csv, index=False)
    
    # 读取结果
    mouse_h_gi = mouse_l_gi = None
    exp_h_gi = exp_l_gi = None
    sample_h_gi = sample_l_gi = None
    
    if os.path.isfile(mouse_gi_csv):
        df = pd.read_csv(mouse_gi_csv)
        mouse_h_gi, mouse_l_gi = df['h_germline_identity'].mean(), df['l_germline_identity'].mean()
        logger.info(f'  Mouse GI H: {mouse_h_gi:.4f}, L: {mouse_l_gi:.4f}')
    
    if os.path.isfile(exp_gi_csv):
        df = pd.read_csv(exp_gi_csv)
        exp_h_gi, exp_l_gi = df['h_germline_identity'].mean(), df['l_germline_identity'].mean()
        logger.info(f'  Exp GI H: {exp_h_gi:.4f}, L: {exp_l_gi:.4f}')
        if mouse_h_gi:
            logger.info(f'  Exp GI Improvement vs Mouse H: {exp_h_gi - mouse_h_gi:.4f}, L: {exp_l_gi - mouse_l_gi:.4f}')
    
    if os.path.isfile(sample_gi_csv):
        df = pd.read_csv(sample_gi_csv)
        sample_h_gi, sample_l_gi = df['h_germline_identity'].mean(), df['l_germline_identity'].mean()
        logger.info(f'  Sample GI H: {sample_h_gi:.4f}, L: {sample_l_gi:.4f}')
        if mouse_h_gi:
            logger.info(f'  Sample GI Improvement vs Mouse H: {sample_h_gi - mouse_h_gi:.4f}, L: {sample_l_gi - mouse_l_gi:.4f}')
    
    # 保存 summary
    pd.DataFrame({
        'Type': ['Mouse', 'Exp', 'Sample'],
        'Heavy_GI_Mean': [mouse_h_gi, exp_h_gi, sample_h_gi],
        'Light_GI_Mean': [mouse_l_gi, exp_l_gi, sample_l_gi],
        'Heavy_GI_Improve_vs_Mouse': [None, exp_h_gi - mouse_h_gi if (mouse_h_gi and exp_h_gi) else None,
                                       sample_h_gi - mouse_h_gi if (mouse_h_gi and sample_h_gi) else None],
        'Light_GI_Improve_vs_Mouse': [None, exp_l_gi - mouse_l_gi if (mouse_l_gi and exp_l_gi) else None,
                                       sample_l_gi - mouse_l_gi if (mouse_l_gi and sample_l_gi) else None]
    }).to_csv(os.path.join(dir_paths['Germline_identity'], 'germline_identity_summary.csv'), index=False)
    
    # ============ 6. Preservation 评分 ============
    logger.info('=' * 60)
    logger.info('6. Preservation Scoring...')
    
    # Exp Preservation (Exp vs Mouse)
    if exp_df is not None and os.path.isfile(lab_mouse):
        lab_h_df = exp_df
        lab_m_df = pd.read_csv(lab_mouse)
        exp_all, exp_vernier = cal_group_all_perservation(lab_h_df, lab_m_df, scheme='kabat', idx_type='lab')
        pd.DataFrame({
            'Sample': range(len(exp_all)), 
            'H_all': [x[0] for x in exp_all], 
            'L_all': [x[1] for x in exp_all],
            'H_vernier': [x[0] for x in exp_vernier], 
            'L_vernier': [x[1] for x in exp_vernier]
        }).to_csv(os.path.join(dir_paths['Preservation'], 'exp_preservation.csv'), index=False)
        logger.info(f'  Exp all: H={np.array(exp_all)[:,0].mean():.4f}, L={np.array(exp_all)[:,1].mean():.4f}')
    
    # Sample Preservation (Sample vs Mouse)
    sample_all, sample_vernier = cal_group_all_perservation(human_df, mouse_df, scheme='kabat', idx_type='sap')
    pd.DataFrame({
        'Sample': range(len(sample_all)), 
        'H_all': [x[0] for x in sample_all], 
        'L_all': [x[1] for x in sample_all],
        'H_vernier': [x[0] for x in sample_vernier], 
        'L_vernier': [x[1] for x in sample_vernier]
    }).to_csv(os.path.join(dir_paths['Preservation'], 'sample_preservation.csv'), index=False)
    logger.info(f'  Sample all: H={np.array(sample_all)[:,0].mean():.4f}, L={np.array(sample_all)[:,1].mean():.4f}')
    
    # ============ 7. Mutation Precision 评分 ============
    logger.info('=' * 60)
    logger.info('7. Vernier Mutation Precision Scoring...')
    
    if exp_df is not None and os.path.isfile(lab_mouse):
        lab_h_df = exp_df
        lab_m_df = pd.read_csv(lab_mouse)
        exp_vernier_h, exp_vernier_l = cal_group_vernier_precision(lab_h_df, lab_m_df, human_df, scheme='kabat')
        
        # 确保 H 和 L 长度一致
        min_len = min(len(exp_vernier_h), len(exp_vernier_l))
        if len(exp_vernier_h) > min_len:
            exp_vernier_h = exp_vernier_h[:min_len]
        if len(exp_vernier_l) > min_len:
            exp_vernier_l = exp_vernier_l[:min_len]
        
        if len(exp_vernier_h) > 0:
            pd.DataFrame({
                'Sample': range(len(exp_vernier_h)), 
                'H_precision': exp_vernier_h,
                'L_precision': exp_vernier_l
            }).to_csv(os.path.join(dir_paths['Mutation_precision'], 'vernier_precision.csv'), index=False)
            logger.info(f'  Vernier precision: H={np.array(exp_vernier_h).mean():.4f}, L={np.array(exp_vernier_l).mean():.4f}')
    
    logger.info('=' * 60)
    logger.info('8. FR Mutation Precision Scoring...')
    
    if exp_df is not None and os.path.isfile(lab_mouse):
        lab_h_df = exp_df
        lab_m_df = pd.read_csv(lab_mouse)
        exp_fr_h, exp_fr_l = cal_group_fr_precision(lab_h_df, lab_m_df, human_df, scheme='kabat')
        
        # 确保 H 和 L 长度一致
        min_len = min(len(exp_fr_h), len(exp_fr_l))
        if len(exp_fr_h) > min_len:
            exp_fr_h = exp_fr_h[:min_len]
        if len(exp_fr_l) > min_len:
            exp_fr_l = exp_fr_l[:min_len]
        
        if len(exp_fr_h) > 0:
            pd.DataFrame({
                'Sample': range(len(exp_fr_h)), 
                'H_precision': exp_fr_h,
                'L_precision': exp_fr_l
            }).to_csv(os.path.join(dir_paths['Mutation_precision'], 'fr_precision.csv'), index=False)
            logger.info(f'  FR precision: H={np.array(exp_fr_h).mean():.4f}, L={np.array(exp_fr_l).mean():.4f}')
    
    # ============ 9. ABLSTM 评分 ============
    logger.info('=' * 60)
    logger.info('9. ABLSTM Scoring...')
    
    mouse_ablstm_csv = os.path.join(dir_paths['ABLSTM'], 'mouse_ablstm_score.csv')
    exp_ablstm_csv = os.path.join(dir_paths['ABLSTM'], 'exp_ablstm_score.csv')
    sample_ablstm_csv = os.path.join(dir_paths['ABLSTM'], 'sample_ablstm_score.csv')
    
    # Mouse ABLSTM
    if len(mouse_df) > 0 and not os.path.isfile(mouse_ablstm_csv):
        try:
            mouse_h_sequences = mouse_df['hseq'].tolist()
            mouse_h_scores = run_ablstm_eval(mouse_h_sequences)
            pd.DataFrame({'Sample': range(len(mouse_h_scores)), 'H_ABLSTM_Score': mouse_h_scores}).to_csv(mouse_ablstm_csv, index=False)
            logger.info(f'  Mouse ABLSTM H-score: {np.mean(mouse_h_scores):.4f}')
        except Exception as e:
            logger.error(f'Mouse ABLSTM failed: {e}')
    
    # Exp ABLSTM
    if exp_df is not None and not os.path.isfile(exp_ablstm_csv):
        try:
            exp_h_sequences = exp_df['h_seq'].tolist()
            exp_h_scores = run_ablstm_eval(exp_h_sequences)
            pd.DataFrame({'Sample': range(len(exp_h_scores)), 'H_ABLSTM_Score': exp_h_scores}).to_csv(exp_ablstm_csv, index=False)
            logger.info(f'  Exp ABLSTM H-score: {np.mean(exp_h_scores):.4f}')
        except Exception as e:
            logger.error(f'Exp ABLSTM failed: {e}')
    
    # Sample ABLSTM
    if not os.path.isfile(sample_ablstm_csv):
        try:
            human_h_sequences = human_df['hseq'].tolist()
            human_h_scores = run_ablstm_eval(human_h_sequences)
            pd.DataFrame({'Sample': range(len(human_h_scores)), 'H_ABLSTM_Score': human_h_scores}).to_csv(sample_ablstm_csv, index=False)
            logger.info(f'  Sample ABLSTM H-score: {np.mean(human_h_scores):.4f}')
        except Exception as e:
            logger.error(f'Sample ABLSTM failed: {e}')
    
    # 读取结果
    mouse_ablstm_mean = exp_ablstm_mean = sample_ablstm_mean = None
    if os.path.isfile(mouse_ablstm_csv):
        df = pd.read_csv(mouse_ablstm_csv)
        mouse_ablstm_mean = df['H_ABLSTM_Score'].mean()
        logger.info(f'  Mouse ABLSTM Mean: {mouse_ablstm_mean:.4f}')
    
    if os.path.isfile(exp_ablstm_csv):
        df = pd.read_csv(exp_ablstm_csv)
        exp_ablstm_mean = df['H_ABLSTM_Score'].mean()
        logger.info(f'  Exp ABLSTM Mean: {exp_ablstm_mean:.4f}')
        if mouse_ablstm_mean:
            logger.info(f'  Exp ABLSTM Improvement vs Mouse: {exp_ablstm_mean - mouse_ablstm_mean:.4f}')
    
    if os.path.isfile(sample_ablstm_csv):
        df = pd.read_csv(sample_ablstm_csv)
        sample_ablstm_mean = df['H_ABLSTM_Score'].mean()
        logger.info(f'  Sample ABLSTM Mean: {sample_ablstm_mean:.4f}')
        if mouse_ablstm_mean:
            logger.info(f'  Sample ABLSTM Improvement vs Mouse: {sample_ablstm_mean - mouse_ablstm_mean:.4f}')
    
    # 保存 summary
    pd.DataFrame({
        'Type': ['Mouse', 'Exp', 'Sample'],
        'ABLSTM_H_Score_Mean': [mouse_ablstm_mean, exp_ablstm_mean, sample_ablstm_mean],
        'ABLSTM_Improve_vs_Mouse': [None, exp_ablstm_mean - mouse_ablstm_mean if (mouse_ablstm_mean and exp_ablstm_mean) else None,
                                     sample_ablstm_mean - mouse_ablstm_mean if (mouse_ablstm_mean and sample_ablstm_mean) else None]
    }).to_csv(os.path.join(dir_paths['ABLSTM'], 'ablstm_summary.csv'), index=False)
    
    # ============ Summary ============
    logger.info('=' * 60)
    logger.info('SUMMARY - Output directories:')
    for d, p in dir_paths.items():
        logger.info(f'  {d}: {p}')
    logger.info('=' * 60)
    logger.info('Evaluation completed!')

if __name__ == '__main__':
    # 设置环境变量
    current_path = os.getenv("PATH", "")
    current_ld_path = os.getenv("LD_LIBRARY_PATH", "")

    # AbNatiV 环境
    abnativ_lib = os.path.expanduser('/mnt/wucy/miniconda3/envs/abnativ/lib')
    abnativ_bin = os.path.expanduser('/mnt/wucy/miniconda3/envs/abnativ/bin')
    if os.path.exists(abnativ_lib):
        os.environ['LD_LIBRARY_PATH'] = abnativ_lib + ':' + current_ld_path
    if os.path.exists(abnativ_bin):
        os.environ['PATH'] = abnativ_bin + ':' + current_path
        print(f"AbNatiV env configured: {abnativ_lib}")

    # BioPhi 环境
    biophi_bin = os.path.expanduser('/mnt/wucy/miniconda3/envs/biophi/bin')
    if os.path.exists(biophi_bin):
        os.environ['PATH'] = biophi_bin + ':' + current_path
        print(f"BioPhi added to PATH: {biophi_bin}")

    import argparse
    parser = argparse.ArgumentParser(description='Humab25 antibody humanization evaluation')
    parser.add_argument('sample_path', type=str, help='Path to sample_humanization_result.csv')
    args = parser.parse_args()
    main(args.sample_path)
