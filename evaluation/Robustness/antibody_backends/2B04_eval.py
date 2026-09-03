"""
抗体人源化评估脚本 (Antibody Humanization Evaluation)

评估指标：
1. T20 分数（重链和轻链）
2. AbNatiV VH 分数（重链人源性）
3. AbNatiV VL 分数（轻链人源性 - VKappa/VLambda）
4. Germline Identity 分数（FR区域与人类Germline一致性）
5. OASis 分数（BioPhi工具）

输入：sample_humanization_result.csv
输出：存储在同目录下的子文件夹中
"""
import os
import subprocess
from tqdm import tqdm
from abnumber import Chain
import numpy as np
import shutil
import pandas as pd
import sys

current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, current_dir)

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from evaluation.Robustness.T20_eval import main as t20_main
from utils.Robustness.misc import get_logger


# ============ 路径配置 ============
BIOPHI_DIR = '/mnt/wucy/WUCHUYA/BioPhi'
OASIS_DB_PATH = os.path.join(BIOPHI_DIR, 'OASis_9mers_v1.db')


# ============ 辅助函数 ============

def cal_fr_preservation(chain1, chain2):
    """计算两条链的Framework区域保留率"""
    identity = 0
    fr_sum = 0
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


def cal_germline_identity_single(seq, scheme='imgt'):
    """计算单条序列的Germline Identity"""
    try:
        chain = Chain(seq, scheme=scheme)
        chain_graft = chain.graft_cdrs_onto_human_germline()
        return cal_fr_preservation(chain, chain_graft)
    except:
        return None


def cal_group_germline_identity(df, h_col='hseq', l_col='lseq', name_col='name', scheme='imgt'):
    """计算一组序列的Germline Identity"""
    results = []
    for idx in tqdm(df.index, desc="Calculating Germline Identity"):
        try:
            h_seq = df.iloc[idx][h_col]
            l_seq = df.iloc[idx][l_col]
            name = df.iloc[idx][name_col]
            
            h_gi = cal_germline_identity_single(h_seq, scheme)
            l_gi = cal_germline_identity_single(l_seq, scheme)
            
            results.append({
                'name': name,
                'h_seq': h_seq,
                'l_seq': l_seq,
                'h_germline_identity': h_gi,
                'l_germline_identity': l_gi
            })
        except Exception as e:
            continue
    return results


def detect_light_chain_type(seq):
    """检测轻链类型 (kappa 或 lambda)"""
    try:
        chain = Chain(seq, scheme='imgt')
        if chain.chain_type == 'K':
            return 'VKappa'
        elif chain.chain_type == 'L':
            return 'VLambda'
        else:
            return 'VKappa'
    except:
        return 'VKappa'


def seqs_to_fasta(seqs, names, save_path):
    """将序列列表保存为FASTA文件"""
    seq_records = [SeqRecord(Seq(seq), id=name, description='') for seq, name in zip(seqs, names)]
    with open(save_path, 'w') as f:
        SeqIO.write(seq_records, f, 'fasta')


def seqs_to_paired_fasta(h_seqs, l_seqs, names, save_path):
    """将配对的重链轻链保存为FASTA文件（用于OASis）"""
    seq_records = []
    for h_seq, l_seq, name in zip(h_seqs, l_seqs, names):
        # 重链
        seq_records.append(SeqRecord(Seq(h_seq), id=f"{name}_VH", description=''))
        # 轻链
        seq_records.append(SeqRecord(Seq(l_seq), id=f"{name}_VL", description=''))
    with open(save_path, 'w') as f:
        SeqIO.write(seq_records, f, 'fasta')


def run_abnativ(exec_path, nat_type, input_fa, out_dir, oid, extra_args=None):
    """运行AbNatiV命令行工具"""
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


def run_oasis(biophi_exec, input_fa, oasis_db, output_xlsx):
    """运行BioPhi OASis评分"""
    cmd = [
        biophi_exec, 'oasis',
        input_fa,
        '--oasis-db', oasis_db,
        '--output', output_xlsx
    ]
    
    print('RUN:', ' '.join(cmd))
    
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    
    out, err = p.communicate()
    print(out)
    if err:
        print('STDERR:', err)
    
    if p.returncode != 0:
        raise RuntimeError(f'BioPhi OASis failed: {err}')


# ============ 主函数 ============

def main(root_path):
    """
    抗体人源化评估主函数
    
    Args:
        root_path: sample_humanization_result.csv 的路径
    """
    root_path = os.path.abspath(root_path)
    base_dir = os.path.dirname(root_path)

    logger = get_logger('ab_eval', base_dir, log_name='eval_log.txt')
    logger.info(f'Evaluating: {root_path}')

    # ============ 检查工具可用性 ============
    abnativ_exec = shutil.which("abnativ")
    if abnativ_exec is None:
        logger.warning("abnativ not found in PATH")
        abnativ_exec = None
    else:
        logger.info(f"AbNatiV found: {abnativ_exec}")

    # BioPhi 可执行文件
    biophi_exec = shutil.which("biophi")
    if biophi_exec is None:
        logger.warning("biophi not found in PATH. Try: conda activate biophi")
    else:
        logger.info(f"BioPhi found: {biophi_exec}")
    
    # OASis 数据库
    oasis_db = OASIS_DB_PATH
    if os.path.exists(oasis_db):
        logger.info(f"OASis DB found: {oasis_db}")
    else:
        logger.warning(f"OASis DB not found at: {oasis_db}")
        logger.warning("Download from: https://zenodo.org/record/5164685")
        oasis_db = None

    # ============ 读取数据 ============
    sample_df = pd.read_csv(root_path)
    
    # 分离原始序列和人源化序列
    mouse_df = sample_df[sample_df['sample_humanization_result.csvific'] == 'mouse'].reset_index(drop=True)
    human_df = sample_df[sample_df['Specific'] == 'humanization'].reset_index(drop=True)

    logger.info(f'Mouse (original) sequences: {len(mouse_df)}')
    logger.info(f'Humanized sequences: {len(human_df)}')

    if len(human_df) == 0:
        logger.error('No humanization sequences found!')
        return

    # 检测轻链类型
    light_type = detect_light_chain_type(human_df.iloc[0]['lseq'])
    logger.info(f'Detected light chain type: {light_type}')

    # ============ 创建目录结构 ============
    vh_base_dir = os.path.join(base_dir, 'VH')
    vl_base_dir = os.path.join(base_dir, 'VL')  # 轻链结果统一命名为VL
    t20_base_dir = os.path.join(base_dir, 'T20')
    germline_base_dir = os.path.join(base_dir, 'Germline_identity')
    oasis_base_dir = os.path.join(base_dir, 'OASis')
    
    for d in [vh_base_dir, vl_base_dir, t20_base_dir, germline_base_dir, oasis_base_dir]:
        os.makedirs(d, exist_ok=True)

    # ============ 准备FASTA文件 ============
    # Mouse (原始序列)
    mouse_h_fa = os.path.join(base_dir, 'mouse_heavy.fa')
    mouse_l_fa = os.path.join(base_dir, 'mouse_light.fa')
    mouse_paired_fa = os.path.join(base_dir, 'mouse_paired.fa')
    
    if len(mouse_df) > 0:
        seqs_to_fasta(mouse_df['hseq'].tolist(), 
                      [f"mouse_{i}" for i in range(len(mouse_df))], 
                      mouse_h_fa)
        seqs_to_fasta(mouse_df['lseq'].tolist(), 
                      [f"mouse_{i}" for i in range(len(mouse_df))], 
                      mouse_l_fa)
        seqs_to_paired_fasta(mouse_df['hseq'].tolist(),
                             mouse_df['lseq'].tolist(),
                             [f"mouse_{i}" for i in range(len(mouse_df))],
                             mouse_paired_fa)

    # Human (人源化序列)
    human_h_fa = os.path.join(base_dir, 'sample_heavy.fa')
    human_l_fa = os.path.join(base_dir, 'sample_light.fa')
    human_paired_fa = os.path.join(base_dir, 'sample_identity.fa')
    
    seqs_to_fasta(human_df['hseq'].tolist(), 
                  [f"human_{i}" for i in range(len(human_df))], 
                  human_h_fa)
    seqs_to_fasta(human_df['lseq'].tolist(), 
                  [f"human_{i}" for i in range(len(human_df))], 
                  human_l_fa)
    seqs_to_paired_fasta(human_df['hseq'].tolist(),
                         human_df['lseq'].tolist(),
                         [f"human_{i}" for i in range(len(human_df))],
                         human_paired_fa)

    # ============ 1. AbNatiV VH 评分 ============
    logger.info('=' * 60)
    logger.info('1. AbNatiV VH Scoring...')
    
    mouse_vh_dir = os.path.join(vh_base_dir, 'mouse_vh')
    sample_vh_dir = os.path.join(vh_base_dir, 'sample_vh')
    os.makedirs(mouse_vh_dir, exist_ok=True)
    os.makedirs(sample_vh_dir, exist_ok=True)

    mouse_vh_csv = os.path.join(mouse_vh_dir, 'mouse_vh_abnativ_seq_scores.csv')
    sample_vh_csv = os.path.join(sample_vh_dir, 'sample_vh_abnativ_seq_scores.csv')

    if abnativ_exec:
        # Mouse VH
        if len(mouse_df) > 0 and not os.path.isfile(mouse_vh_csv):
            try:
                run_abnativ(abnativ_exec, 'VH', mouse_h_fa, mouse_vh_dir, 'mouse_vh')
            except Exception as e:
                logger.error(f'Mouse VH scoring failed: {e}')
        
        # Sample VH
        if not os.path.isfile(sample_vh_csv):
            try:
                run_abnativ(abnativ_exec, 'VH', human_h_fa, sample_vh_dir, 'sample_vh')
            except Exception as e:
                logger.error(f'Sample VH scoring failed: {e}')

    # 读取VH分数
    mouse_vh_mean, sample_vh_mean = None, None
    if os.path.isfile(mouse_vh_csv):
        mouse_vh_df = pd.read_csv(mouse_vh_csv)
        mouse_vh_mean = mouse_vh_df['AbNatiV VH Score'].mean()
        logger.info(f'  Mouse VH Score: {mouse_vh_mean:.4f}')
    if os.path.isfile(sample_vh_csv):
        sample_vh_df = pd.read_csv(sample_vh_csv)
        sample_vh_mean = sample_vh_df['AbNatiV VH Score'].mean()
        logger.info(f'  Sample VH Score: {sample_vh_mean:.4f}')
    if mouse_vh_mean and sample_vh_mean:
        logger.info(f'  VH Improvement: {sample_vh_mean - mouse_vh_mean:.4f}')

    # ============ 2. AbNatiV VL 评分 ============
    logger.info('=' * 60)
    logger.info(f'2. AbNatiV {light_type} Scoring...')
    
    mouse_vl_dir = os.path.join(vl_base_dir, 'mouse_vl')
    sample_vl_dir = os.path.join(vl_base_dir, 'sample_vl')
    os.makedirs(mouse_vl_dir, exist_ok=True)
    os.makedirs(sample_vl_dir, exist_ok=True)

    mouse_vl_csv = os.path.join(mouse_vl_dir, 'mouse_vl_abnativ_seq_scores.csv')
    sample_vl_csv = os.path.join(sample_vl_dir, 'sample_vl_abnativ_seq_scores.csv')

    if abnativ_exec:
        # Mouse VL
        if len(mouse_df) > 0 and not os.path.isfile(mouse_vl_csv):
            try:
                run_abnativ(abnativ_exec, light_type, mouse_l_fa, mouse_vl_dir, 'mouse_vl')
            except Exception as e:
                logger.error(f'Mouse VL scoring failed: {e}')
        
        # Sample VL
        if not os.path.isfile(sample_vl_csv):
            try:
                run_abnativ(abnativ_exec, light_type, human_l_fa, sample_vl_dir, 'sample_vl')
            except Exception as e:
                logger.error(f'Sample VL scoring failed: {e}')

    # 读取VL分数
    vl_score_col = f'AbNatiV {light_type} Score'
    mouse_vl_mean, sample_vl_mean = None, None
    if os.path.isfile(mouse_vl_csv):
        mouse_vl_df = pd.read_csv(mouse_vl_csv)
        mouse_vl_mean = mouse_vl_df[vl_score_col].mean()
        logger.info(f'  Mouse {light_type} Score: {mouse_vl_mean:.4f}')
    if os.path.isfile(sample_vl_csv):
        sample_vl_df = pd.read_csv(sample_vl_csv)
        sample_vl_mean = sample_vl_df[vl_score_col].mean()
        logger.info(f'  Sample {light_type} Score: {sample_vl_mean:.4f}')
    if mouse_vl_mean and sample_vl_mean:
        logger.info(f'  {light_type} Improvement: {sample_vl_mean - mouse_vl_mean:.4f}')

    # ============ 3. T20 评分 ============
    logger.info('=' * 60)
    logger.info('3. T20 Scoring (online)...')
    
    # Mouse T20
    mouse_t20_csv = os.path.join(t20_base_dir, 'mouse_t20_score.csv')
    if len(mouse_df) > 0 and not os.path.isfile(mouse_t20_csv):
        try:
            mouse_t20_input = os.path.join(base_dir, 'temp_mouse_t20_input.csv')
            mouse_df_t20 = mouse_df.copy()
            mouse_df_t20['Specific'] = 'humanization'
            mouse_df_t20.to_csv(mouse_t20_input, index=False)
            
            t20_result = t20_main(mouse_t20_input)
            if os.path.isfile(t20_result):
                result_df = pd.read_csv(t20_result)
                result_df.to_csv(mouse_t20_csv, index=False)
                os.remove(t20_result)
            os.remove(mouse_t20_input)
        except Exception as e:
            logger.error(f'Mouse T20 scoring failed: {e}')

    # Sample T20
    sample_t20_csv = os.path.join(t20_base_dir, 'sample_t20_score.csv')
    if not os.path.isfile(sample_t20_csv):
        try:
            sample_t20_input = os.path.join(base_dir, 'temp_sample_t20_input.csv')
            human_df_t20 = human_df.copy()
            human_df_t20['Specific'] = 'humanization'
            human_df_t20.to_csv(sample_t20_input, index=False)
            
            t20_result = t20_main(sample_t20_input)
            if os.path.isfile(t20_result):
                result_df = pd.read_csv(t20_result)
                result_df.to_csv(sample_t20_csv, index=False)
                os.remove(t20_result)
            os.remove(sample_t20_input)
        except Exception as e:
            logger.error(f'Sample T20 scoring failed: {e}')

    # 读取T20分数
    mouse_h_t20, mouse_l_t20 = None, None
    sample_h_t20, sample_l_t20 = None, None
    
    if os.path.isfile(mouse_t20_csv):
        mouse_t20_df = pd.read_csv(mouse_t20_csv)
        mouse_h_t20 = mouse_t20_df['h_score'].mean()
        mouse_l_t20 = mouse_t20_df['l_score'].mean()
        logger.info(f'  Mouse T20 Heavy: {mouse_h_t20:.2f}')
        logger.info(f'  Mouse T20 Light: {mouse_l_t20:.2f}')
    
    if os.path.isfile(sample_t20_csv):
        sample_t20_df = pd.read_csv(sample_t20_csv)
        sample_h_t20 = sample_t20_df['h_score'].mean()
        sample_l_t20 = sample_t20_df['l_score'].mean()
        logger.info(f'  Sample T20 Heavy: {sample_h_t20:.2f}')
        logger.info(f'  Sample T20 Light: {sample_l_t20:.2f}')
    
    if mouse_h_t20 and sample_h_t20:
        logger.info(f'  T20 Heavy Improvement: {sample_h_t20 - mouse_h_t20:.2f}')
    if mouse_l_t20 and sample_l_t20:
        logger.info(f'  T20 Light Improvement: {sample_l_t20 - mouse_l_t20:.2f}')

    # ============ 4. Germline Identity 评分 ============
    logger.info('=' * 60)
    logger.info('4. Germline Identity Scoring...')
    
    # Mouse Germline Identity
    mouse_germline_csv = os.path.join(germline_base_dir, 'mouse_germline_identity.csv')
    if len(mouse_df) > 0 and not os.path.isfile(mouse_germline_csv):
        mouse_gi_results = cal_group_germline_identity(mouse_df, h_col='hseq', l_col='lseq', name_col='name')
        mouse_gi_df = pd.DataFrame(mouse_gi_results)
        mouse_gi_df.to_csv(mouse_germline_csv, index=False)
    
    # Sample Germline Identity
    sample_germline_csv = os.path.join(germline_base_dir, 'sample_germline_identity.csv')
    if not os.path.isfile(sample_germline_csv):
        sample_gi_results = cal_group_germline_identity(human_df, h_col='hseq', l_col='lseq', name_col='name')
        sample_gi_df = pd.DataFrame(sample_gi_results)
        sample_gi_df.to_csv(sample_germline_csv, index=False)

    # 读取Germline Identity分数
    mouse_h_gi, mouse_l_gi = None, None
    sample_h_gi, sample_l_gi = None, None
    
    if os.path.isfile(mouse_germline_csv):
        mouse_gi_df = pd.read_csv(mouse_germline_csv)
        mouse_h_gi = mouse_gi_df['h_germline_identity'].mean()
        mouse_l_gi = mouse_gi_df['l_germline_identity'].mean()
        logger.info(f'  Mouse Heavy Germline Identity: {mouse_h_gi:.4f}')
        logger.info(f'  Mouse Light Germline Identity: {mouse_l_gi:.4f}')
    
    if os.path.isfile(sample_germline_csv):
        sample_gi_df = pd.read_csv(sample_germline_csv)
        sample_h_gi = sample_gi_df['h_germline_identity'].mean()
        sample_l_gi = sample_gi_df['l_germline_identity'].mean()
        logger.info(f'  Sample Heavy Germline Identity: {sample_h_gi:.4f}')
        logger.info(f'  Sample Light Germline Identity: {sample_l_gi:.4f}')
    
    if mouse_h_gi and sample_h_gi:
        logger.info(f'  Heavy GI Improvement: {sample_h_gi - mouse_h_gi:.4f}')
    if mouse_l_gi and sample_l_gi:
        logger.info(f'  Light GI Improvement: {sample_l_gi - mouse_l_gi:.4f}')

    # ============ 5. OASis 评分 ============
    logger.info('=' * 60)
    logger.info('5. OASis Scoring (BioPhi)...')

    if biophi_exec and oasis_db:
        # Mouse OASis (BioPhi输出XLSX，再转换为CSV，然后删除XLSX)
        mouse_oasis_xlsx = os.path.join(oasis_base_dir, 'mouse_oasis.xlsx')
        mouse_oasis_csv = os.path.join(oasis_base_dir, 'mouse_oasis.csv')
        if len(mouse_df) > 0 and not os.path.isfile(mouse_oasis_csv):
            try:
                run_oasis(biophi_exec, mouse_paired_fa, oasis_db, mouse_oasis_xlsx)
                # 读取XLSX并保存为CSV，然后删除XLSX
                if os.path.isfile(mouse_oasis_xlsx):
                    mouse_oasis_df = pd.read_excel(mouse_oasis_xlsx, sheet_name='OASis Curves', index_col=0)
                    mouse_oasis_df.to_csv(mouse_oasis_csv)
                    os.remove(mouse_oasis_xlsx)
            except Exception as e:
                logger.error(f'Mouse OASis scoring failed: {e}')

        # Sample OASis
        sample_oasis_xlsx = os.path.join(oasis_base_dir, 'sample_oasis.xlsx')
        sample_oasis_csv = os.path.join(oasis_base_dir, 'sample_oasis.csv')
        if not os.path.isfile(sample_oasis_csv):
            try:
                run_oasis(biophi_exec, human_paired_fa, oasis_db, sample_oasis_xlsx)
                # 读取XLSX并保存为CSV，然后删除XLSX
                if os.path.isfile(sample_oasis_xlsx):
                    sample_oasis_df = pd.read_excel(sample_oasis_xlsx, sheet_name='OASis Curves', index_col=0)
                    sample_oasis_df.to_csv(sample_oasis_csv)
                    os.remove(sample_oasis_xlsx)
            except Exception as e:
                logger.error(f'Sample OASis scoring failed: {e}')

        # 读取OASis分数（从CSV读取）
        mouse_oasis_50, sample_oasis_50 = None, None

        if os.path.isfile(mouse_oasis_csv):
            try:
                mouse_oasis_df = pd.read_csv(mouse_oasis_csv, index_col=0)
                mouse_oasis_50 = mouse_oasis_df['50%'].mean()
                logger.info(f'  Mouse OASis 50%: {mouse_oasis_50:.4f}')
            except Exception as e:
                logger.error(f'Reading mouse OASis failed: {e}')

        if os.path.isfile(sample_oasis_csv):
            try:
                sample_oasis_df = pd.read_csv(sample_oasis_csv, index_col=0)
                sample_oasis_50 = sample_oasis_df['50%'].mean()
                logger.info(f'  Sample OASis 50%: {sample_oasis_50:.4f}')
            except Exception as e:
                logger.error(f'Reading sample OASis failed: {e}')

        if mouse_oasis_50 is not None and sample_oasis_50 is not None:
            logger.info(f'  OASis 50% Improvement: {sample_oasis_50 - mouse_oasis_50:.4f}')
    else:
        logger.warning('OASis scoring skipped (BioPhi or DB not available)')

    # ============ 汇总输出 ============
    logger.info('=' * 60)
    logger.info('SUMMARY')
    logger.info('=' * 60)
    logger.info('Output directories:')
    logger.info(f'  VH: {vh_base_dir}')
    logger.info(f'  VL ({light_type}): {vl_base_dir}')
    logger.info(f'  T20: {t20_base_dir}')
    logger.info(f'  Germline Identity: {germline_base_dir}')
    logger.info(f'  OASis: {oasis_base_dir}')
    logger.info('=' * 60)
    logger.info('Evaluation completed!')


if __name__ == '__main__':
    # 设置 LD_LIBRARY_PATH
    current_ld_library_path = os.getenv("LD_LIBRARY_PATH", "")
    conda_lib = os.path.expanduser('/mnt/wucy/miniconda3/envs/abnativ/lib')
    if os.path.exists(conda_lib):
        os.environ['LD_LIBRARY_PATH'] = conda_lib + ':' + current_ld_library_path

    import argparse
    parser = argparse.ArgumentParser(description='Antibody humanization evaluation')
    parser.add_argument('sample_path', type=str, help='Path to sample_humanization_result.csv')
    args = parser.parse_args()
    
    main(args.sample_path)
