"""
抗体人源化评估脚本 (Antibody Humanization Evaluation for HuAb348)
"""

import os
import subprocess
import shutil
import tempfile
from tqdm import tqdm
from abnumber import Chain
import numpy as np
import pandas as pd
import sys
import re # ABLSTM 需要用到正则
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

# ============ 路径配置 ============
# 添加 ABLSTM 目录到 Python 路径
ABLSTM_DIR = '/mnt/wucy/WUCHUYA/ABLSTM'
if ABLSTM_DIR not in sys.path:
    sys.path.insert(0, ABLSTM_DIR)

current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, current_dir)

from evaluation.Robustness.T20_eval import main as t20_main
from utils.Robustness.misc import get_logger

# 工具路径
BIOPHI_DIR = '/mnt/wucy/WUCHUYA/BioPhi'
OASIS_DB_PATH = os.path.join(BIOPHI_DIR, 'OASis_9mers_v1.db')
ABNATIV_DIR = '/mnt/wucy/WUCHUYA/AbNatiV'

# 数据路径
LAB_MOUSE_FPATH = 'data/antibody_eval_data/HuAb348_data/sample_t20_mouse_score.csv'
LAB_HUMAN_FPATH = 'data/antibody_eval_data/HuAb348_data/sample_t20_exp_score.csv'
# 这里的静态文件路径仅作备用，现在的脚本会优先尝试主动计算
LAB_MOUSE_T20 = 'data/antibody_eval_data/HuAb348_data/sample_t20_mouse_score.csv'
LAB_EXP_T20 = 'data/antibody_eval_data/HuAb348_data/sample_t20_exp_score.csv'

# ============ 辅助函数 ============

def seqs_to_fasta(seqs, names, save_path):
    """将序列列表保存为FASTA格式"""
    seq_records = [SeqRecord(Seq(seq), id=name, description='') for seq, name in zip(seqs, names)]
    with open(save_path, 'w') as f:
        SeqIO.write(seq_records, f, 'fasta')

def seqs_to_paired_fasta(h_seqs, l_seqs, names, save_path):
    """将配对的重链和轻链序列保存为FASTA格式"""
    seq_records = []
    for h_seq, l_seq, name in zip(h_seqs, l_seqs, names):
        seq_records.append(SeqRecord(Seq(h_seq), id=f"{name}_VH", description=''))
        seq_records.append(SeqRecord(Seq(l_seq), id=f"{name}_VL", description=''))
    with open(save_path, 'w') as f:
        SeqIO.write(seq_records, f, 'fasta')

def run_abnativ(exec_path, nat_type, input_fa, out_dir, oid):
    """运行AbNatiV工具评估人源性"""
    cmd = [exec_path, 'score', '-nat', nat_type, '-i', input_fa, '-odir', out_dir, '-oid', oid, '-align']
    # print('RUN:', ' '.join(cmd))
    env = os.environ.copy()
    env['PYTHONPATH'] = ABNATIV_DIR + os.pathsep + env.get('PYTHONPATH', '')
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    out, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError(f'abnativ {nat_type} failed: {err}')

def run_oasis(biophi_exec, input_fa, oasis_db, output_xlsx):
    """运行BioPhi OASis工具"""
    cmd = [biophi_exec, 'oasis', input_fa, '--oasis-db', oasis_db, '--output', output_xlsx]
    # print('RUN:', ' '.join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError(f'OASis failed: {err}')

def detect_light_chain_type(seq):
    """检测轻链类型 (Kappa或Lambda)"""
    try:
        chain = Chain(seq, scheme='imgt')
        return 'VKappa' if chain.chain_type == 'K' else 'VLambda'
    except:
        return 'VKappa'  # 默认为Kappa

# ============ ABLSTM 相关函数 (修复部分) ============

def seq_trans_to_aho(sequences):
    """将序列转换为AHO格式用于ABLSTM评估 (移植自 humab25_eval.py)"""
    from anarci import anarci
    data = [(f'{i}', seq) for i, seq in enumerate(sequences)]
    h_results = anarci(data, scheme='aho', output=False)
    h_seq_results = h_results[0]
    
    aho_seq_list = []
    for seq_list in h_seq_results:
        if seq_list and seq_list[0]:
            re_seq = seq_list[0][0]
            # ANARCI returns entries as ``((aho_position, insertion_code), aa)``.
            # Do not parse the string representation: insertion codes such as
            # "B" can otherwise be mistaken for amino acids and crash ABLSTM.
            aho_seq = '-' + ''.join(pos_aa[1] for pos_aa in re_seq)
            if len(aho_seq) != 150:
                if len(aho_seq) < 150:
                    pad_count = 150 - len(aho_seq)
                    aho_seq = aho_seq + '-' * pad_count
                else:
                    aho_seq = aho_seq[:150]
            aho_seq_list.append(aho_seq)
        else:
            aho_seq_list.append('-' * 150) # Fallback
    return aho_seq_list

def run_ablstm_eval(sequences):
    """运行ABLSTM评估 (修复为直接调用模型)"""
    try:
        if ABLSTM_DIR not in sys.path:
            sys.path.insert(0, ABLSTM_DIR)
        from ablstm import ModelLSTM
        
        # 1. 转换序列为 AHO 格式
        aho_list = seq_trans_to_aho(sequences)
        
        # 2. 写入临时文件
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w') as tmp:
            tmp_fpath = tmp.name
            for seq in aho_list:
                tmp.write(seq + '\n')
        
        try:
            # 3. 加载模型 (使用 humab25_eval.py 中的路径)
            model_data_path = os.path.join(ABLSTM_DIR, 'saved_models/tmp/model_tmp.npy')
            # model_data_path = '/mnt/wucy/WUCHUYA/ABLSTM/saved_models/tmp/model_tmp.npy'
            
            pred_model = ModelLSTM(embedding_dim=64, hidden_dim=64, device='cpu', gapped=True, fixed_len=True)
            
            if not os.path.exists(model_data_path):
                 # 尝试另一个常见路径
                model_data_path = os.path.join(ABLSTM_DIR, 'models', 'best_model.npy')
                if not os.path.exists(model_data_path):
                    # 如果找不到模型文件，抛出明确错误
                     raise FileNotFoundError(f"ABLSTM model not found at {model_data_path}")

            pred_model.load(fn=model_data_path)
            
            # 4. 预测
            h_score = pred_model.eval(tmp_fpath)
            
            os.remove(tmp_fpath)
            return h_score
            
        except Exception as e:
            if os.path.exists(tmp_fpath):
                os.remove(tmp_fpath)
            raise e
            
    except Exception as e:
        print(f"Error in ABLSTM evaluation: {e}")
        # 返回全 None 列表以防止程序崩溃
        return [None] * len(sequences)

# ============ 计算函数 (保持原样) ============

def cal_fr_preservation(chain1, chain2):
    """计算FR区域保持率"""
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
    """计算全序列保持率"""
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
    """计算Vernier区域保持率"""
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
    """计算FR区域突变精确度"""
    share = only = 0
    try:
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
    except:
        return None

def cal_vernier_mutation_precision(expchain, parental, test):
    """计算Vernier区域突变精确度"""
    share = only = 0
    try:
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
    except:
        return None

def cal_germline_identity_single(seq, scheme='imgt'):
    """计算单条序列的Germline Identity"""
    try:
        chain = Chain(seq, scheme=scheme)
        chain_graft = chain.graft_cdrs_onto_human_germline()
        return cal_fr_preservation(chain, chain_graft)
    except:
        return None

def cal_group_fr_germline_identity(df, h_col='hseq', l_col='lseq', name_col='name'):
    """批量计算Germline Identity"""
    results = []
    for idx in tqdm(df.index, desc="Germline Identity"):
        try:
            h_seq = df.iloc[idx][h_col]
            l_seq = df.iloc[idx][l_col]
            name = df.iloc[idx][name_col] if name_col in df.columns else f"sample_{idx}"
            h_gi = cal_germline_identity_single(h_seq)
            l_gi = cal_germline_identity_single(l_seq)
            results.append({
                'name': name, 
                'h_germline_identity': h_gi, 
                'l_germline_identity': l_gi
            })
        except Exception as e:
            # print(f"Warning: Germline identity calculation failed for index {idx}: {e}")
            continue
    return results

def cal_group_all_perservation(human_df, mouse_df, scheme='kabat', idx_type='lab'):
    """批量计算全序列和Vernier区域保持率"""
    if idx_type == 'lab':
        h_idx, l_idx = 'h_seq', 'l_seq'
    else:
        h_idx, l_idx = 'hseq', 'lseq'
    
    preservation_all_ratio_list = []
    preservation_vernier_ratio_list = []
    
    for idx in tqdm(human_df.index, desc="Preservation"):
        try:
            human_h_seq = human_df.iloc[idx][h_idx]
            human_l_seq = human_df.iloc[idx][l_idx]
            human_h_chain = Chain(human_h_seq, scheme=scheme)
            human_l_chain = Chain(human_l_seq, scheme=scheme)

            mouse_h_seq = mouse_df.iloc[idx]['h_seq']
            mouse_l_seq = mouse_df.iloc[idx]['l_seq']
            mouse_h_chain = Chain(mouse_h_seq, scheme=scheme)
            mouse_l_chain = Chain(mouse_l_seq, scheme=scheme)

            h_per_all_ratio = cal_all_preservation(human_h_chain, mouse_h_chain)
            l_per_all_ratio = cal_all_preservation(human_l_chain, mouse_l_chain)

            h_per_vernier_ratio = cal_vernier_preservation(human_h_chain, mouse_h_chain)
            l_per_vernier_ratio = cal_vernier_preservation(human_l_chain, mouse_l_chain)

            preservation_all_ratio_list.append([h_per_all_ratio, l_per_all_ratio])
            preservation_vernier_ratio_list.append([h_per_vernier_ratio, l_per_vernier_ratio])
        except Exception as e:
            # print(f"Warning: Preservation calculation failed for index {idx}: {e}")
            continue

    return preservation_all_ratio_list, preservation_vernier_ratio_list

def cal_group_fr_precision(exp_df, mou_df, sample_df, scheme='kabat'):
    """批量计算FR区域突变精确度"""
    h_list, l_list = [], []
    for idx in tqdm(exp_df.index, desc="FR Precision"):
        try:
            exp_h = exp_df.iloc[idx]['h_seq']
            exp_l = exp_df.iloc[idx]['l_seq']
            mou_h = mou_df.iloc[idx]['h_seq']
            mou_l = mou_df.iloc[idx]['l_seq']
            sap_h = sample_df.iloc[idx]['hseq']
            sap_l = sample_df.iloc[idx]['lseq']
            
            exp_h_c = Chain(exp_h, scheme=scheme)
            exp_l_c = Chain(exp_l, scheme=scheme)
            mou_h_c = Chain(mou_h, scheme=scheme)
            mou_l_c = Chain(mou_l, scheme=scheme)
            sap_h_c = Chain(sap_h, scheme=scheme)
            sap_l_c = Chain(sap_l, scheme=scheme)
            
            h_r = cal_fr_mutation_precision(exp_h_c, mou_h_c, sap_h_c)
            l_r = cal_fr_mutation_precision(exp_l_c, mou_l_c, sap_l_c)
            
            if h_r is not None:
                h_list.append(h_r)
            if l_r is not None:
                l_list.append(l_r)
        except Exception as e:
            # print(f"Warning: FR precision calculation failed for index {idx}: {e}")
            continue
    
    return h_list, l_list

def cal_group_vernier_precision(exp_df, mou_df, sample_df, scheme='kabat'):
    """批量计算Vernier区域突变精确度"""
    h_list, l_list = [], []
    for idx in tqdm(exp_df.index, desc="Vernier Precision"):
        try:
            exp_h = exp_df.iloc[idx]['h_seq']
            exp_l = exp_df.iloc[idx]['l_seq']
            mou_h = mou_df.iloc[idx]['h_seq']
            mou_l = mou_df.iloc[idx]['l_seq']
            sap_h = sample_df.iloc[idx]['hseq']
            sap_l = sample_df.iloc[idx]['lseq']
            
            exp_h_c = Chain(exp_h, scheme=scheme)
            exp_l_c = Chain(exp_l, scheme=scheme)
            mou_h_c = Chain(mou_h, scheme=scheme)
            mou_l_c = Chain(mou_l, scheme=scheme)
            sap_h_c = Chain(sap_h, scheme=scheme)
            sap_l_c = Chain(sap_l, scheme=scheme)
            
            h_r = cal_vernier_mutation_precision(exp_h_c, mou_h_c, sap_h_c)
            l_r = cal_vernier_mutation_precision(exp_l_c, mou_l_c, sap_l_c)
            
            if h_r is not None:
                h_list.append(h_r)
            if l_r is not None:
                l_list.append(l_r)
        except Exception as e:
            # print(f"Warning: Vernier precision calculation failed for index {idx}: {e}")
            continue
    
    return h_list, l_list

# ============ 主评估函数 ============

def main(sample_path):
    """
    主评估函数
    Args:
        sample_path: sample_humanization_result.csv 的路径
    """
    # 创建输出目录结构
    base_dir = os.path.dirname(sample_path)
    eval_dir = os.path.join(base_dir, 'evaluation_results')
    
    dir_paths = {
        'AbNatiV': os.path.join(eval_dir, '1_AbNatiV'),
        'T20': os.path.join(eval_dir, '2_T20'),
        'OASis': os.path.join(eval_dir, '3_OASis'),
        'Germline_identity': os.path.join(eval_dir, '4_Germline_Identity'),
        'Preservation': os.path.join(eval_dir, '5_Preservation'),
        'Mutation_precision': os.path.join(eval_dir, '6_Mutation_Precision'),
        'ABLSTM': os.path.join(eval_dir, '7_ABLSTM')
    }
    
    for dir_path in dir_paths.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # 初始化logger
    logger = get_logger('HuAb348_eval', eval_dir, log_name='eval_log.txt')
    logger.info('=' * 80)
    logger.info('HuAb348 Antibody Humanization Evaluation')
    logger.info('=' * 80)
    logger.info(f'Sample file: {sample_path}')
    logger.info(f'Output directory: {eval_dir}')
    logger.info('=' * 80)
    
    # 读取数据
    logger.info('Loading data...')
    sample_df = pd.read_csv(sample_path)
    human_df = sample_df[sample_df['Specific'] == 'humanization'].reset_index(drop=True)
    
    # 读取实验数据
    exp_df = None
    mouse_df = None

    # 直接读取指定的 Exp 数据文件
    if os.path.isfile(LAB_HUMAN_FPATH):
        logger.info(f'Loading Exp data from: {LAB_HUMAN_FPATH}')
        exp_df = pd.read_csv(LAB_HUMAN_FPATH)
        logger.info(f'Loaded {len(exp_df)} experimental humanized sequences')
    
    # 直接读取指定的 Mouse 数据文件
    if os.path.isfile(LAB_MOUSE_FPATH):
        logger.info(f'Loading Mouse data from: {LAB_MOUSE_FPATH}')
        mouse_df = pd.read_csv(LAB_MOUSE_FPATH)
        logger.info(f'Loaded {len(mouse_df)} mouse sequences')
    
    logger.info(f'Loaded {len(human_df)} sample humanized sequences')
    
    # ============ 1. AbNatiV 评分 ============
    logger.info('=' * 80)
    logger.info('1. AbNatiV Scoring...')
    
    # 创建子目录以保持整洁
    abnativ_mouse_dir = os.path.join(dir_paths['AbNatiV'], 'mouse')
    abnativ_exp_dir = os.path.join(dir_paths['AbNatiV'], 'exp')
    abnativ_sample_dir = os.path.join(dir_paths['AbNatiV'], 'sample')
    os.makedirs(abnativ_mouse_dir, exist_ok=True)
    os.makedirs(abnativ_exp_dir, exist_ok=True)
    os.makedirs(abnativ_sample_dir, exist_ok=True)

    # 查找 AbNatiV 可执行文件
    abnativ_conda = os.path.expanduser('/mnt/wucy/miniconda3/envs/abnativ/bin/abnativ')
    abnativ_exec = None
    if os.path.isfile(abnativ_conda):
        abnativ_exec = abnativ_conda
        logger.info(f"Using Conda AbNatiV: {abnativ_exec}")
    else:
        abnativ_exec = shutil.which("abnativ")
        if abnativ_exec is None:
            abnativ_local = os.path.join(ABNATIV_DIR, 'bin', 'abnativ')
            if os.path.isfile(abnativ_local):
                abnativ_exec = abnativ_local
            else:
                abnativ_exec = os.path.join(ABNATIV_DIR, 'abnativ')
    
    # AbNatiV: 初始化变量
    mouse_vh_mean = mouse_vl_mean = None
    exp_vh_mean = exp_vl_mean = None
    sample_vh_mean = sample_vl_mean = None

    # --- Mouse AbNatiV ---
    if mouse_df is not None and len(mouse_df) > 0:
        mouse_vh_oid = 'mouse_vh'
        mouse_vh_score_file = os.path.join(abnativ_mouse_dir, f'{mouse_vh_oid}_abnativ_seq_scores.csv')
        
        # VH
        if not os.path.isfile(mouse_vh_score_file):
            mouse_h_fa = os.path.join(abnativ_mouse_dir, 'mouse_vh.fasta')
            seqs_to_fasta(mouse_df['h_seq' if 'h_seq' in mouse_df.columns else 'hseq'].tolist(),
                         [f'mouse_{i}' for i in range(len(mouse_df))], mouse_h_fa)
            try:
                run_abnativ(abnativ_exec, 'VH', mouse_h_fa, abnativ_mouse_dir, mouse_vh_oid)
            except Exception as e:
                logger.error(f'  Mouse VH AbNatiV failed: {e}')
        
        if os.path.isfile(mouse_vh_score_file):
            df = pd.read_csv(mouse_vh_score_file)
            mouse_vh_mean = df['AbNatiV VH Score'].mean()
            logger.info(f'  Mouse VH Mean Score: {mouse_vh_mean:.4f}')

        # VL
        mouse_l_seqs = mouse_df['l_seq' if 'l_seq' in mouse_df.columns else 'lseq'].tolist()
        light_types = [detect_light_chain_type(seq) for seq in mouse_l_seqs]
        vl_scores = []
        for lt in ['VKappa', 'VLambda']:
            indices = [i for i, t in enumerate(light_types) if t == lt]
            if not indices: continue
            oid = f'mouse_{lt.lower()}'
            score_file = os.path.join(abnativ_mouse_dir, f'{oid}_abnativ_seq_scores.csv')
            if not os.path.isfile(score_file):
                l_fa = os.path.join(abnativ_mouse_dir, f'mouse_{lt.lower()}.fasta')
                seqs_to_fasta([mouse_l_seqs[i] for i in indices], [f'mouse_{i}' for i in indices], l_fa)
                try: run_abnativ(abnativ_exec, lt, l_fa, abnativ_mouse_dir, oid)
                except Exception as e: logger.error(f'  Mouse {lt} AbNatiV failed: {e}')
            if os.path.isfile(score_file):
                df = pd.read_csv(score_file)
                col_name = [c for c in df.columns if 'Score' in c and 'AbNatiV' in c][0]
                vl_scores.extend(df[col_name].tolist())
        if vl_scores:
            mouse_vl_mean = np.mean(vl_scores)
            logger.info(f'  Mouse VL Mean Score: {mouse_vl_mean:.4f}')

    # --- Exp AbNatiV ---
    if exp_df is not None and len(exp_df) > 0:
        exp_vh_oid = 'exp_vh'
        exp_vh_score_file = os.path.join(abnativ_exp_dir, f'{exp_vh_oid}_abnativ_seq_scores.csv')
        
        # VH
        if not os.path.isfile(exp_vh_score_file):
            exp_h_fa = os.path.join(abnativ_exp_dir, 'exp_vh.fasta')
            seqs_to_fasta(exp_df['h_seq'].tolist(), [f'exp_{i}' for i in range(len(exp_df))], exp_h_fa)
            try: run_abnativ(abnativ_exec, 'VH', exp_h_fa, abnativ_exp_dir, exp_vh_oid)
            except Exception as e: logger.error(f'  Exp VH AbNatiV failed: {e}')
        
        if os.path.isfile(exp_vh_score_file):
            df = pd.read_csv(exp_vh_score_file)
            exp_vh_mean = df['AbNatiV VH Score'].mean()
            logger.info(f'  Exp VH Mean Score: {exp_vh_mean:.4f}')

        # VL
        exp_l_seqs = exp_df['l_seq'].tolist()
        light_types = [detect_light_chain_type(seq) for seq in exp_l_seqs]
        vl_scores = []
        for lt in ['VKappa', 'VLambda']:
            indices = [i for i, t in enumerate(light_types) if t == lt]
            if not indices: continue
            oid = f'exp_{lt.lower()}'
            score_file = os.path.join(abnativ_exp_dir, f'{oid}_abnativ_seq_scores.csv')
            if not os.path.isfile(score_file):
                l_fa = os.path.join(abnativ_exp_dir, f'exp_{lt.lower()}.fasta')
                seqs_to_fasta([exp_l_seqs[i] for i in indices], [f'exp_{i}' for i in indices], l_fa)
                try: run_abnativ(abnativ_exec, lt, l_fa, abnativ_exp_dir, oid)
                except Exception as e: logger.error(f'  Exp {lt} AbNatiV failed: {e}')
            if os.path.isfile(score_file):
                df = pd.read_csv(score_file)
                col_name = [c for c in df.columns if 'Score' in c and 'AbNatiV' in c][0]
                vl_scores.extend(df[col_name].tolist())
        if vl_scores:
            exp_vl_mean = np.mean(vl_scores)
            logger.info(f'  Exp VL Mean Score: {exp_vl_mean:.4f}')

    # --- Sample AbNatiV ---
    sample_vh_oid = 'sample_vh'
    sample_vh_score_file = os.path.join(abnativ_sample_dir, f'{sample_vh_oid}_abnativ_seq_scores.csv')
    
    if not os.path.isfile(sample_vh_score_file):
        sample_h_fa = os.path.join(abnativ_sample_dir, 'sample_vh.fasta')
        seqs_to_fasta(human_df['hseq'].tolist(), [f'sample_{i}' for i in range(len(human_df))], sample_h_fa)
        try: run_abnativ(abnativ_exec, 'VH', sample_h_fa, abnativ_sample_dir, sample_vh_oid)
        except Exception as e: logger.error(f'  Sample VH AbNatiV failed: {e}')

    if os.path.isfile(sample_vh_score_file):
        df = pd.read_csv(sample_vh_score_file)
        sample_vh_mean = df['AbNatiV VH Score'].mean()
        logger.info(f'  Sample VH Mean Score: {sample_vh_mean:.4f}')

    sample_l_seqs = human_df['lseq'].tolist()
    light_types = [detect_light_chain_type(seq) for seq in sample_l_seqs]
    vl_scores = []
    for lt in ['VKappa', 'VLambda']:
        indices = [i for i, t in enumerate(light_types) if t == lt]
        if not indices: continue
        oid = f'sample_{lt.lower()}'
        score_file = os.path.join(abnativ_sample_dir, f'{oid}_abnativ_seq_scores.csv')
        if not os.path.isfile(score_file):
            l_fa = os.path.join(abnativ_sample_dir, f'sample_{lt.lower()}.fasta')
            seqs_to_fasta([sample_l_seqs[i] for i in indices], [f'sample_{i}' for i in indices], l_fa)
            try: run_abnativ(abnativ_exec, lt, l_fa, abnativ_sample_dir, oid)
            except Exception as e: logger.error(f'  Sample {lt} AbNatiV failed: {e}')
        if os.path.isfile(score_file):
            df = pd.read_csv(score_file)
            col_name = [c for c in df.columns if 'Score' in c and 'AbNatiV' in c][0]
            vl_scores.extend(df[col_name].tolist())
    if vl_scores:
        sample_vl_mean = np.mean(vl_scores)
        logger.info(f'  Sample VL Mean Score: {sample_vl_mean:.4f}')
    
    # 保存 AbNatiV Summary
    pd.DataFrame({
        'Type': ['Mouse', 'Exp', 'Sample'],
        'VH_Mean': [mouse_vh_mean, exp_vh_mean, sample_vh_mean],
        'VL_Mean': [mouse_vl_mean, exp_vl_mean, sample_vl_mean],
        'VH_Improve_vs_Mouse': [None, 
                                exp_vh_mean - mouse_vh_mean if (mouse_vh_mean and exp_vh_mean) else None,
                                sample_vh_mean - mouse_vh_mean if (mouse_vh_mean and sample_vh_mean) else None],
        'VL_Improve_vs_Mouse': [None,
                                exp_vl_mean - mouse_vl_mean if (mouse_vl_mean and exp_vl_mean) else None,
                                sample_vl_mean - mouse_vl_mean if (mouse_vl_mean and sample_vl_mean) else None]
    }).to_csv(os.path.join(dir_paths['AbNatiV'], 'abnativ_summary.csv'), index=False)
    
    # ============ 2. T20 评分 ============
    logger.info('=' * 80)
    logger.info('2. T20 Scoring...')
    
    # Mouse T20
    mouse_t20_h = mouse_t20_l = None
    if os.path.isfile(LAB_MOUSE_T20):
        df = pd.read_csv(LAB_MOUSE_T20)
        mouse_t20_h = df['h_score'].mean()
        mouse_t20_l = df['l_score'].mean()
        logger.info(f'  Mouse T20: H={mouse_t20_h:.4f}, L={mouse_t20_l:.4f}')
        shutil.copy(LAB_MOUSE_T20, os.path.join(dir_paths['T20'], 'mouse_t20_score.csv'))
    
    # Exp T20
    exp_t20_h = exp_t20_l = None
    if os.path.isfile(LAB_EXP_T20):
        df = pd.read_csv(LAB_EXP_T20)
        exp_t20_h = df['h_score'].mean()
        exp_t20_l = df['l_score'].mean()
        logger.info(f'  Exp T20: H={exp_t20_h:.4f}, L={exp_t20_l:.4f}')
        if mouse_t20_h:
            logger.info(f'  Exp T20 Improvement vs Mouse: H={exp_t20_h - mouse_t20_h:.4f}, L={exp_t20_l - mouse_t20_l:.4f}')
        shutil.copy(LAB_EXP_T20, os.path.join(dir_paths['T20'], 'exp_t20_score.csv'))
    
    # Sample T20
    sample_t20_path = t20_main(sample_path)
    if os.path.isfile(sample_t20_path):
        shutil.copy(sample_t20_path, os.path.join(dir_paths['T20'], 'sample_t20_score.csv'))
        df = pd.read_csv(sample_t20_path)
        sample_t20_h = df['h_score'].mean()
        sample_t20_l = df['l_score'].mean()
        logger.info(f'  Sample T20: H={sample_t20_h:.4f}, L={sample_t20_l:.4f}')
        if mouse_t20_h:
            logger.info(f'  Sample T20 Improvement vs Mouse: H={sample_t20_h - mouse_t20_h:.4f}, L={sample_t20_l - mouse_t20_l:.4f}')
    
    pd.DataFrame({
        'Type': ['Mouse', 'Exp', 'Sample'],
        'H_T20_Mean': [mouse_t20_h, exp_t20_h, sample_t20_h],
        'L_T20_Mean': [mouse_t20_l, exp_t20_l, sample_t20_l],
        'H_T20_Improve_vs_Mouse': [None,
                                   exp_t20_h - mouse_t20_h if (mouse_t20_h and exp_t20_h) else None,
                                   sample_t20_h - mouse_t20_h if (mouse_t20_h and sample_t20_h) else None],
        'L_T20_Improve_vs_Mouse': [None,
                                   exp_t20_l - mouse_t20_l if (mouse_t20_l and exp_t20_l) else None,
                                   sample_t20_l - mouse_t20_l if (mouse_t20_l and sample_t20_l) else None]
    }).to_csv(os.path.join(dir_paths['T20'], 't20_summary.csv'), index=False)
    
    # ============ 3. OASis 评分 (修复: 补充 Mouse/Exp 计算逻辑) ============
    logger.info('=' * 80)
    logger.info('3. OASis Scoring...')
    
    biophi_exec = shutil.which("biophi")
    if biophi_exec is None:
        biophi_exec = os.path.join(os.path.expanduser('/mnt/wucy/miniconda3/envs/biophi/bin'), 'biophi')
    
    # 定义通用的 OASis 处理函数
    def process_oasis(name, df, h_col, l_col, out_dir):
        xlsx_path = os.path.join(out_dir, f'{name}_oasis.xlsx')
        mean_50 = None
        
        # 1. 运行计算 (如果文件不存在)
        if not os.path.isfile(xlsx_path):
            if df is None or len(df) == 0:
                logger.warning(f'  Skipping {name} OASis: No data')
                return None
            
            fa_path = os.path.join(out_dir, f'{name}_sequences.fasta')
            seqs_to_paired_fasta(df[h_col].tolist(), df[l_col].tolist(), 
                                [f'{name}_{i}' for i in range(len(df))], fa_path)
            try:
                run_oasis(biophi_exec, fa_path, OASIS_DB_PATH, xlsx_path)
            except Exception as e:
                logger.error(f'  {name} OASis run failed: {e}')
                return None
        
        # 2. 读取结果并打印 50% Mean
        if os.path.isfile(xlsx_path):
            try:
                # OASis 结果通常在 'OASis Curves' sheet 中
                res_df = pd.read_excel(xlsx_path, sheet_name='OASis Curves', index_col=0)
                if '50%' in res_df.columns:
                    mean_50 = res_df['50%'].mean()
                    logger.info(f'  {name.capitalize()} OASis 50% Mean: {mean_50:.4f}')
                else:
                    logger.warning(f'  {name} OASis: Column "50%" not found in Excel')
            except Exception as e:
                logger.error(f'  {name} OASis read excel failed: {e}')
        return mean_50

    # 依次处理 Mouse, Exp, Sample
    mouse_oasis_50 = process_oasis('mouse', mouse_df, 'h_seq', 'l_seq', dir_paths['OASis'])
    exp_oasis_50 = process_oasis('exp', exp_df, 'h_seq', 'l_seq', dir_paths['OASis'])
    sample_oasis_50 = process_oasis('sample', human_df, 'hseq', 'lseq', dir_paths['OASis'])
    
    pd.DataFrame({
        'Type': ['Mouse', 'Exp', 'Sample'],
        'OASis_50%_Mean': [mouse_oasis_50, exp_oasis_50, sample_oasis_50],
        'OASis_Improve_vs_Mouse': [None,
                                   exp_oasis_50 - mouse_oasis_50 if (mouse_oasis_50 and exp_oasis_50) else None,
                                   sample_oasis_50 - mouse_oasis_50 if (mouse_oasis_50 and sample_oasis_50) else None]
    }).to_csv(os.path.join(dir_paths['OASis'], 'oasis_summary.csv'), index=False)
    
    # ============ 4. Germline Identity 评分 ============
    logger.info('=' * 80)
    logger.info('4. Germline Identity Scoring...')
    
    # Mouse Germline Identity
    mouse_h_gi = mouse_l_gi = None
    if mouse_df is not None and len(mouse_df) > 0:
        mouse_gi_csv = os.path.join(dir_paths['Germline_identity'], 'mouse_germline_identity.csv')
        if os.path.isfile(mouse_gi_csv):
            mouse_gi_results = pd.read_csv(mouse_gi_csv).to_dict('records')
        else:
            mouse_gi_results = cal_group_fr_germline_identity(
                mouse_df,
                h_col='h_seq' if 'h_seq' in mouse_df.columns else 'hseq',
                l_col='l_seq' if 'l_seq' in mouse_df.columns else 'lseq',
                name_col='name' if 'name' in mouse_df.columns else None
            )
        if mouse_gi_results:
            if not os.path.isfile(mouse_gi_csv):
                pd.DataFrame(mouse_gi_results).to_csv(mouse_gi_csv, index=False)
            mouse_h_gi = np.array([r['h_germline_identity'] for r in mouse_gi_results if r['h_germline_identity'] is not None]).mean()
            mouse_l_gi = np.array([r['l_germline_identity'] for r in mouse_gi_results if r['l_germline_identity'] is not None]).mean()
            logger.info(f'  Mouse Germline Identity: H={mouse_h_gi:.4f}, L={mouse_l_gi:.4f}')
    
    # Exp Germline Identity
    exp_h_gi = exp_l_gi = None
    if exp_df is not None and len(exp_df) > 0:
        exp_gi_csv = os.path.join(dir_paths['Germline_identity'], 'exp_germline_identity.csv')
        if os.path.isfile(exp_gi_csv):
            exp_gi_results = pd.read_csv(exp_gi_csv).to_dict('records')
        else:
            exp_gi_results = cal_group_fr_germline_identity(exp_df, h_col='h_seq', l_col='l_seq')
        if exp_gi_results:
            if not os.path.isfile(exp_gi_csv):
                pd.DataFrame(exp_gi_results).to_csv(exp_gi_csv, index=False)
            exp_h_gi = np.array([r['h_germline_identity'] for r in exp_gi_results if r['h_germline_identity'] is not None]).mean()
            exp_l_gi = np.array([r['l_germline_identity'] for r in exp_gi_results if r['l_germline_identity'] is not None]).mean()
            logger.info(f'  Exp Germline Identity: H={exp_h_gi:.4f}, L={exp_l_gi:.4f}')
            if mouse_h_gi:
                logger.info(f'  Exp GI Improvement vs Mouse: H={exp_h_gi - mouse_h_gi:.4f}, L={exp_l_gi - mouse_l_gi:.4f}')
    
    # Sample Germline Identity
    sample_gi_csv = os.path.join(dir_paths['Germline_identity'], 'sample_germline_identity.csv')
    if os.path.isfile(sample_gi_csv):
        sample_gi_results = pd.read_csv(sample_gi_csv).to_dict('records')
    else:
        sample_gi_results = cal_group_fr_germline_identity(human_df, h_col='hseq', l_col='lseq')
    if sample_gi_results:
        if not os.path.isfile(sample_gi_csv):
            pd.DataFrame(sample_gi_results).to_csv(sample_gi_csv, index=False)
        sample_h_gi = np.array([r['h_germline_identity'] for r in sample_gi_results if r['h_germline_identity'] is not None]).mean()
        sample_l_gi = np.array([r['l_germline_identity'] for r in sample_gi_results if r['l_germline_identity'] is not None]).mean()
        logger.info(f'  Sample Germline Identity: H={sample_h_gi:.4f}, L={sample_l_gi:.4f}')
        if mouse_h_gi:
            logger.info(f'  Sample GI Improvement vs Mouse: H={sample_h_gi - mouse_h_gi:.4f}, L={sample_l_gi - mouse_l_gi:.4f}')
    
    pd.DataFrame({
        'Type': ['Mouse', 'Exp', 'Sample'],
        'H_GI_Mean': [mouse_h_gi, exp_h_gi, sample_h_gi],
        'L_GI_Mean': [mouse_l_gi, exp_l_gi, sample_l_gi],
        'H_GI_Improve_vs_Mouse': [None,
                                  exp_h_gi - mouse_h_gi if (mouse_h_gi and exp_h_gi) else None,
                                  sample_h_gi - mouse_h_gi if (mouse_h_gi and sample_h_gi) else None],
        'L_GI_Improve_vs_Mouse': [None,
                                  exp_l_gi - mouse_l_gi if (mouse_l_gi and exp_l_gi) else None,
                                  sample_l_gi - mouse_l_gi if (mouse_l_gi and sample_l_gi) else None]
    }).to_csv(os.path.join(dir_paths['Germline_identity'], 'germline_identity_summary.csv'), index=False)
    
    # ============ 5. Preservation 评分 ============
    logger.info('=' * 80)
    logger.info('5. Preservation Scoring...')
     
    # Exp Preservation (vs Mouse)
    if exp_df is not None and mouse_df is not None:
        exp_preservation_csv = os.path.join(dir_paths['Preservation'], 'exp_preservation.csv')
        if os.path.isfile(exp_preservation_csv):
            exp_preservation_df = pd.read_csv(exp_preservation_csv)
            logger.info(f'  Exp All Preservation: H={exp_preservation_df["H_all"].mean():.4f}, L={exp_preservation_df["L_all"].mean():.4f}')
            logger.info(f'  Exp Vernier Preservation: H={exp_preservation_df["H_vernier"].mean():.4f}, L={exp_preservation_df["L_vernier"].mean():.4f}')
        else:
            exp_all, exp_vernier = cal_group_all_perservation(exp_df, mouse_df, scheme='kabat', idx_type='lab')
            if exp_all:
                pd.DataFrame({
                    'Sample': range(len(exp_all)),
                    'H_all': [x[0] for x in exp_all],
                    'L_all': [x[1] for x in exp_all],
                    'H_vernier': [x[0] for x in exp_vernier],
                    'L_vernier': [x[1] for x in exp_vernier]
                }).to_csv(exp_preservation_csv, index=False)
                logger.info(f'  Exp All Preservation: H={np.array(exp_all)[:,0].mean():.4f}, L={np.array(exp_all)[:,1].mean():.4f}')
                logger.info(f'  Exp Vernier Preservation: H={np.array(exp_vernier)[:,0].mean():.4f}, L={np.array(exp_vernier)[:,1].mean():.4f}')
    
    # Sample Preservation (vs Mouse)
    if mouse_df is not None:
        sample_preservation_csv = os.path.join(dir_paths['Preservation'], 'sample_preservation.csv')
        if os.path.isfile(sample_preservation_csv):
            sample_preservation_df = pd.read_csv(sample_preservation_csv)
            logger.info(f'  Sample All Preservation: H={sample_preservation_df["H_all"].mean():.4f}, L={sample_preservation_df["L_all"].mean():.4f}')
            logger.info(f'  Sample Vernier Preservation: H={sample_preservation_df["H_vernier"].mean():.4f}, L={sample_preservation_df["L_vernier"].mean():.4f}')
        else:
            sample_all, sample_vernier = cal_group_all_perservation(human_df, mouse_df, scheme='kabat', idx_type='sap')
            if sample_all:
                pd.DataFrame({
                    'Sample': range(len(sample_all)),
                    'H_all': [x[0] for x in sample_all],
                    'L_all': [x[1] for x in sample_all],
                    'H_vernier': [x[0] for x in sample_vernier],
                    'L_vernier': [x[1] for x in sample_vernier]
                }).to_csv(sample_preservation_csv, index=False)
                logger.info(f'  Sample All Preservation: H={np.array(sample_all)[:,0].mean():.4f}, L={np.array(sample_all)[:,1].mean():.4f}')
                logger.info(f'  Sample Vernier Preservation: H={np.array(sample_vernier)[:,0].mean():.4f}, L={np.array(sample_vernier)[:,1].mean():.4f}')
    
    # ============ 6. Mutation Precision 评分 ============
    logger.info('=' * 80)
    logger.info('6. Mutation Precision Scoring...')
    
    if exp_df is not None and mouse_df is not None:
        # Vernier Mutation Precision
        logger.info('  Calculating Vernier Mutation Precision...')
        vernier_precision_csv = os.path.join(dir_paths['Mutation_precision'], 'vernier_precision.csv')
        if os.path.isfile(vernier_precision_csv):
            vernier_precision_df = pd.read_csv(vernier_precision_csv)
            vernier_h = vernier_precision_df['H_precision'].dropna().tolist()
            vernier_l = vernier_precision_df['L_precision'].dropna().tolist()
        else:
            vernier_h, vernier_l = cal_group_vernier_precision(exp_df, mouse_df, human_df, scheme='kabat')
        
        if vernier_h and vernier_l:
            min_len = min(len(vernier_h), len(vernier_l))
            if not os.path.isfile(vernier_precision_csv):
                pd.DataFrame({
                    'Sample': range(min_len),
                    'H_precision': vernier_h[:min_len],
                    'L_precision': vernier_l[:min_len]
                }).to_csv(vernier_precision_csv, index=False)
            logger.info(f'  Vernier Precision: H={np.array(vernier_h).mean():.4f}, L={np.array(vernier_l).mean():.4f}')
            logger.info(f'  Vernier Precision Mean: {(np.array(vernier_h).mean() + np.array(vernier_l).mean()) / 2:.4f}')
        
        # FR Mutation Precision
        logger.info('  Calculating FR Mutation Precision...')
        fr_precision_csv = os.path.join(dir_paths['Mutation_precision'], 'fr_precision.csv')
        if os.path.isfile(fr_precision_csv):
            fr_precision_df = pd.read_csv(fr_precision_csv)
            fr_h = fr_precision_df['H_precision'].dropna().tolist()
            fr_l = fr_precision_df['L_precision'].dropna().tolist()
        else:
            fr_h, fr_l = cal_group_fr_precision(exp_df, mouse_df, human_df, scheme='kabat')
        
        if fr_h and fr_l:
            min_len = min(len(fr_h), len(fr_l))
            if not os.path.isfile(fr_precision_csv):
                pd.DataFrame({
                    'Sample': range(min_len),
                    'H_precision': fr_h[:min_len],
                    'L_precision': fr_l[:min_len]
                }).to_csv(fr_precision_csv, index=False)
            logger.info(f'  FR Precision: H={np.array(fr_h).mean():.4f}, L={np.array(fr_l).mean():.4f}')
            logger.info(f'  FR Precision Mean: {(np.array(fr_h).mean() + np.array(fr_l).mean()) / 2:.4f}')
    
    # ============ 7. ABLSTM 评分 (修复: 使用新函数) ============
    logger.info('=' * 80)
    logger.info('7. ABLSTM Scoring...')
    
    # Mouse ABLSTM
    mouse_ablstm_mean = None
    if mouse_df is not None and len(mouse_df) > 0:
        csv_path = os.path.join(dir_paths['ABLSTM'], 'mouse_ablstm_score.csv')
        if not os.path.isfile(csv_path):
            scores = run_ablstm_eval(mouse_df['h_seq' if 'h_seq' in mouse_df.columns else 'hseq'].tolist())
            pd.DataFrame({'Sample': range(len(scores)), 'H_ABLSTM_Score': scores}).to_csv(csv_path, index=False)
        
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            mouse_ablstm_mean = df['H_ABLSTM_Score'].mean()
            logger.info(f'  Mouse ABLSTM Mean: {mouse_ablstm_mean:.4f}')
            
    # Exp ABLSTM
    exp_ablstm_mean = None
    if exp_df is not None and len(exp_df) > 0:
        csv_path = os.path.join(dir_paths['ABLSTM'], 'exp_ablstm_score.csv')
        if not os.path.isfile(csv_path):
            scores = run_ablstm_eval(exp_df['h_seq'].tolist())
            pd.DataFrame({'Sample': range(len(scores)), 'H_ABLSTM_Score': scores}).to_csv(csv_path, index=False)
            
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            exp_ablstm_mean = df['H_ABLSTM_Score'].mean()
            logger.info(f'  Exp ABLSTM Mean: {exp_ablstm_mean:.4f}')
            if mouse_ablstm_mean:
                logger.info(f'    > Improvement: {exp_ablstm_mean - mouse_ablstm_mean:.4f}')

    # Sample ABLSTM
    sample_ablstm_mean = None
    csv_path = os.path.join(dir_paths['ABLSTM'], 'sample_ablstm_score.csv')
    if not os.path.isfile(csv_path):
        scores = run_ablstm_eval(human_df['hseq'].tolist())
        pd.DataFrame({'Sample': range(len(scores)), 'H_ABLSTM_Score': scores}).to_csv(csv_path, index=False)
        
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        sample_ablstm_mean = df['H_ABLSTM_Score'].mean()
        logger.info(f'  Sample ABLSTM Mean: {sample_ablstm_mean:.4f}')
        if mouse_ablstm_mean:
            logger.info(f'    > Improvement: {sample_ablstm_mean - mouse_ablstm_mean:.4f}')
    
    pd.DataFrame({
        'Type': ['Mouse', 'Exp', 'Sample'],
        'ABLSTM_H_Mean': [mouse_ablstm_mean, exp_ablstm_mean, sample_ablstm_mean],
        'ABLSTM_Improve_vs_Mouse': [None,
                                    exp_ablstm_mean - mouse_ablstm_mean if (mouse_ablstm_mean and exp_ablstm_mean) else None,
                                    sample_ablstm_mean - mouse_ablstm_mean if (mouse_ablstm_mean and sample_ablstm_mean) else None]
    }).to_csv(os.path.join(dir_paths['ABLSTM'], 'ablstm_summary.csv'), index=False)
    
    # ============ 最终总结 ============
    logger.info('=' * 80)
    logger.info('EVALUATION SUMMARY')
    logger.info('=' * 80)
    logger.info('Output directories:')
    for name, path in dir_paths.items():
        logger.info(f'  {name}: {path}')
    logger.info('=' * 80)
    logger.info('Evaluation completed successfully!')
    logger.info('=' * 80)


if __name__ == '__main__':
    # 设置环境变量
    current_path = os.getenv("PATH", "")
    current_ld_path = os.getenv("LD_LIBRARY_PATH", "")
    
    # AbNatiV 环境配置
    abnativ_lib = os.path.expanduser('/mnt/wucy/miniconda3/envs/abnativ/lib')
    abnativ_bin = os.path.expanduser('/mnt/wucy/miniconda3/envs/abnativ/bin')
    if os.path.exists(abnativ_lib):
        os.environ['LD_LIBRARY_PATH'] = abnativ_lib + ':' + current_ld_path
    if os.path.exists(abnativ_bin):
        os.environ['PATH'] = abnativ_bin + ':' + current_path
        print(f"AbNatiV env configured: {abnativ_lib}")
    
    # BioPhi 环境配置
    biophi_bin = os.path.expanduser('/mnt/wucy/miniconda3/envs/biophi/bin')
    if os.path.exists(biophi_bin):
        os.environ['PATH'] = biophi_bin + ':' + current_path
        print(f"BioPhi added to PATH: {biophi_bin}")
    
    import argparse
    parser = argparse.ArgumentParser(description='HuAb348 antibody humanization evaluation')
    parser.add_argument('sample_path', type=str, help='Path to sample_humanization_result.csv')
    args = parser.parse_args()
    main(args.sample_path)
