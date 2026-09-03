""" This script only consider for the nanobody. """
import os.path
import sys
current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, current_dir)


import numpy as np
import torch
from tqdm import tqdm
import argparse
import pandas as pd
from abnumber import Chain
from anarci import anarci, number
from copy import deepcopy
import re
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

from utils.Robustness.anti_numbering import get_regions
from dataset.Robustness.preprocess import HEAVY_POSITIONS_dict, HEAVY_CDR_INDEX, INPAINT_HEAVY_CDR_INDEX
from dataset.Robustness.oas_pair_dataset_new import light_pad_cdr, HEAVY_REGION_INDEX, LIGHT_REGION_INDEX
from utils.Robustness.tokenizer import Tokenizer
from utils.Robustness.train_utils import model_selected
from utils.Robustness.misc import get_new_log_dir, get_logger, seed_all

# Finetune package
from model.Robustness.nanoencoder.abnativ_model import AbNatiV_Model
from model.Robustness.nanoencoder.model import NanoAntiTFNet

REGION_LENGTH = (26, 12, 17, 10, 38, 30, 11)

#将纳米抗体链保存为FASTA格式
def save_nano(heavy_chains, path):
    with open(path, 'w') as f:
        for heavy in heavy_chains:
            Chain.to_fasta(heavy, f, description='VH')


def seqs_to_fasta(df, save_path, version=None):
    assert version is not None, print('Need to given specific version.')
    seq_list = []
    seq_description_list = []
    for i in df.index:
        hseq = df.loc[i]['hseq']
        desci = 'VH' + f'{version}_{i}'
        seq_list.append(hseq)
        seq_description_list.append(desci)
    seqs_records = [SeqRecord(Seq(seq), id=seq_descrip) for seq, seq_descrip in zip(seq_list, seq_description_list)]
    with open(save_path, 'w') as f:
        SeqIO.write(seqs_records, f, 'fasta')

#比较序列各区域长度是否小于等于预设长度
def compare_length(length_list):
    small = True
    for i, lg in enumerate(length_list):
        if lg <= REGION_LENGTH[i]:
            continue
        else:
            small = False
    return small

#按区域长度分割原始序列
def get_diff_region_aa_seq(raw_seq, length_list):
    split_aa_seq_list = []
    start_lg = 0
    for lg in length_list:
        end_lg = start_lg + lg
        aa_seq = raw_seq[start_lg:end_lg]
        split_aa_seq_list.append(aa_seq)
        start_lg = end_lg
    assert ''.join(split_aa_seq_list) == raw_seq, 'Split length has wrong.'
    return split_aa_seq_list

#将氨基酸序列转换为位置字典
def get_pad_seq(aa_seq):
    seq_dict = {}
    results = number(aa_seq, scheme='imgt')

    for key, value in results[0]:
        str_key = str(key[0]) + key[1].strip()
        seq_dict[str_key] = value
    return seq_dict

#复制重链区域索引并转为tensor
def get_input_element(nano_aa):
    h_seq_dict = get_pad_seq(nano_aa)
    h_cdr_region = deepcopy(HEAVY_REGION_INDEX)
    h_pad_region = torch.tensor(h_cdr_region)

    nano_pad_initial_seq = ['-'] * len(HEAVY_CDR_INDEX)
    for key, value in h_seq_dict.items():
        try:
            pos_idx = HEAVY_POSITIONS_dict[key]
            nano_pad_initial_seq[pos_idx] = value
        except KeyError:
            nkey = re.findall(r'\d+', key)
            nkey = int(nkey[0])
            if (27 <= nkey <= 38) or (56 <= nkey <= 65) or ( 105 <= nkey <= 117):
                print("Heavy CDR has problem.")
            else:
                print('H Position {} is not in predefine dict, which can be ignored.'.format(key))
    return h_pad_region, nano_pad_initial_seq


def batch_input_element(nano_sq, inpaint_sample=False, batch_size=10):

    nano_pad_region, nano_pad_initial_seq = get_input_element(nano_sq)
 # 根据采样模式选择掩码
    # Get mask. Do not change CDR region.
    if not inpaint_sample:
        nano_heavy_index = HEAVY_CDR_INDEX
    else:
        nano_heavy_index = INPAINT_HEAVY_CDR_INDEX
    nano_mask = torch.tensor(nano_heavy_index) == 0

#初始化tokenizer并将序列转换为token索引
    # initial mask.
    ms_tokenizer = Tokenizer()
    nano_pad_seq_tokenize = ms_tokenizer.seq2idx(nano_pad_initial_seq)
    #创建初始掩码逻辑
    fram_h = ~(torch.tensor(nano_heavy_index) != 0) * nano_pad_seq_tokenize
    fram_pad_mask = (fram_h != 21)   
    nano_mask = fram_pad_mask * nano_mask
    nano_pad_seq_tokenize[nano_mask] = ms_tokenizer.idx_msk
    #将单样本扩展到批量大小
    nano_pad_region = nano_pad_region.unsqueeze(0).expand(batch_size, -1).clone()
    nano_pad_seq_tokenize = nano_pad_seq_tokenize.unsqueeze(0).expand(batch_size, -1).clone()
    #获取需要生成的氨基酸位置索引
    nano_loc = np.arange(len(nano_heavy_index))
    nano_loc = nano_loc[nano_mask]
    #批量准备输入数据
    return nano_pad_seq_tokenize, nano_pad_region, nano_loc, ms_tokenizer
## 返回：tokenized序列、区域信息、生成位置、tokenizer


#读取纳米抗体CSV文件
def get_nano_line(fpath):
    df_vhh = pd.read_csv(fpath)
    return df_vhh

#从结果CSV中提取人源化序列
# Def a read function. only output the sample human result.
def out_humanization_df(path):
    sample_df = pd.read_csv(path)
    human_df = sample_df[sample_df['Specific'] == 'humanization'].reset_index()
    return human_df

#将单个序列保存为单独的FASTA文
def save_seq_to_fasta(save_dir, save_df, species):
    for idx, line in save_df.iterrows():
        hseq=line['hseq']
        name = f'{idx}_{species}_H'
        save_fpath = os.path.join(save_dir, f'{idx}_{species}.fasta')
        with open(save_fpath, 'w') as f:
            seq_record = SeqRecord(Seq(hseq), id=name)
            SeqIO.write(seq_record, f, "fasta")

#为人源化序列创建FASTA和PDB输出目录
def split_fasta_for_save(fpath):
    sample_human_df = out_humanization_df(fpath)

    # Create file for save fa and pdb
    # [MODIFIED] Save inside the subdirectory
    base_dir = os.path.dirname(fpath)
    fa_fpath = os.path.join(base_dir, 'sample_human_fa')
    pdb_fpath = os.path.join(base_dir, 'sample_human_pdb')
    os.makedirs(fa_fpath, exist_ok=True)
    os.makedirs(pdb_fpath, exist_ok=True)

    save_seq_to_fasta(fa_fpath, sample_human_df, 'human')

#使用字符串分区提取特定模型参数
def get_multi_model_state(ckpt):
    abnativ_state_dict = {
        k.partition('eval_abnativ_model.')[2]: v for k, v in ckpt['model'].items() if k.startswith('eval_abnativ_model.')
    }
    infilling_state_dict = {
        k.partition('infilling_pretrain.')[2]: v for k, v in ckpt['model'].items() if
        k.startswith('infilling_pretrain.')
    }
    return abnativ_state_dict, None, infilling_state_dict



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        default='checkpoints/Robustness/nanobody/hudiffnb.pt'
                    )
    parser.add_argument('--data_fpath', type=str,
                        default='data/Robustness/shark349.csv'
                    )
    parser.add_argument('--batch_size', type=int,
                        default=1
                    )
    parser.add_argument('--sample_number', type=int,
                        default=1
                        )
    parser.add_argument('--try_number', type=int,
                        default=10
                        )
    parser.add_argument('--seed', type=int,
                        default=2023
                        )
    # [NEW] Round control
    parser.add_argument('--n_rounds', type=int,
                        default=3,
                        help='Number of independent sampling rounds'
                        )
    # [NEW] Temperature control
    parser.add_argument('--temperature', type=float,
                        default=1.0,
                        help='Sampling temperature. >1.0 for more diversity, <1.0 for more conservative.'
                        )
    parser.add_argument('--sample_order', type=str,
                        default='shuffle')
    parser.add_argument('--sample_method', type=str,
                        default='gen', choices=['gen', 'rl_gen'])
    parser.add_argument('--length_limit', type=str,
                        default='not_equal')
    parser.add_argument('--model', type=str,
                        default='finetune_vh', choices=['pretrain', 'finetune_vh'])
    parser.add_argument('--fa_version', type=str,
                        default='v_nano')
    parser.add_argument('--inpaint_sample', type=eval,
                        default=False)
    parser.add_argument('--structure', type=eval,
                        default=False)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Optional exact root directory for this sampling run')
    args = parser.parse_args()

    print(f"Mode: {'Inpaint/Motif-Preservation' if args.inpaint_sample else 'De Novo/Standard'}")
    print(f"Temperature: {args.temperature}")
    print(f"Total Rounds: {args.n_rounds}")

    batch_size = args.batch_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if 'filter' in args.data_fpath:
        data_sample = 'abnativ_select'
    elif 'nanobert18' in args.data_fpath:
        data_sample = 'nanobert18'
    else:
        data_sample = os.path.splitext(os.path.basename(args.data_fpath))[0]

    # Root Log Directory Name
    sample_tag = f'rounds{args.n_rounds}_{data_sample}_{args.model}'

    # Create ROOT log dir.  An explicit directory keeps batch experiments in a
    # deterministic layout; the historical timestamped behavior remains the
    # default for existing callers.
    if args.output_dir:
        root_log_dir = os.path.abspath(args.output_dir)
        os.makedirs(root_log_dir, exist_ok=True)
    else:
        root_log_dir = get_new_log_dir(
            root=os.path.join(current_dir, 'results', 'Robustness', 'Nb', data_sample),
            prefix=sample_tag
        )
    # Logger for the root process
    logger = get_logger('root', root_log_dir)
    logger.info(f"Root Output Directory: {root_log_dir}")
    logger.info(args.ckpt)

    # --- Load Model (Once) ---
    if args.model == 'pretrain':
        ckpt = torch.load(args.ckpt, map_location=device,weights_only=False)
        config = ckpt['config']
        model = model_selected(config).to(device)
        model.load_state_dict(ckpt['model'])
        model.eval()

    elif args.model == 'finetune_vh':
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        config = ckpt['config']
        abnativ_state, _, infilling_state = get_multi_model_state(ckpt)
        
        hparams = ckpt['abnativ_params']
        abnativ_model = AbNatiV_Model(hparams)
        abnativ_model.load_state_dict(abnativ_state)
        abnativ_model.to(device)
        infilling_params = ckpt['infilling_params']
        infilling_model = NanoAntiTFNet(**infilling_params)
        infilling_model.load_state_dict(infilling_state)
        infilling_model.to(device)

        model_dict = {
            'abnativ': abnativ_model,
            'infilling': infilling_model,
            'target_infilling': infilling_model,
        }

        # Config adjustments
        config.model['equal_weight'] = True
        config.model['vhh_nativeness'] = False
        config.model['human_threshold'] = None
        config.model['human_all_seq'] = False
        config.model['temperature'] = False 

        framework_model = model_selected(config, pretrained_model=model_dict, tokenizer=Tokenizer())
        model = framework_model.infilling_pretrain
        model.eval()

    
    # Load data once
    nano_df = get_nano_line(args.data_fpath)
    
    # --- Loop for multiple rounds with Subdirectories ---
    for round_idx in range(args.n_rounds):
        # 1. Setup Seed and Subdirectory
        current_seed = args.seed + round_idx
        seed_all(current_seed)
        
        # [NEW] Naming convention: Seed + Inpaint Mode + Temperature
        subdir_name = f"Seed{current_seed}_Inpaint{args.inpaint_sample}_Temp{args.temperature}"
        round_dir = os.path.join(root_log_dir, subdir_name)
        os.makedirs(round_dir, exist_ok=True)
        
        # Setup Logger for this round
        round_logger = get_logger(f'round_{round_idx}', round_dir)
        round_logger.info(f"=== Starting Round {round_idx + 1}/{args.n_rounds} ===")
        round_logger.info(f"Saving to: {round_dir}")

        # 2. Initialize CSV for THIS round
        save_fpath = os.path.join(round_dir, 'sample_humanization_result.csv')
        with open(save_fpath, 'w', encoding='UTF-8') as f:
            f.write('Specific,name,hseq,seed,temperature\n')

        # 3. Processing Loop
        for idx, nano_line in tqdm(enumerate(nano_df.itertuples()), total=len(nano_df.index), desc=f"Round {round_idx+1}"):
            sample_number = args.sample_number
            try_num = args.try_number
            # Historical nanobody inputs use ``vhhseq`` while antibody
            # evaluation tables expose the heavy chain as ``h_seq``.
            row_data = nano_line._asdict()
            nano_vhh = row_data.get('vhhseq', row_data.get('h_seq', row_data.get('hseq')))
            if not isinstance(nano_vhh, str) or not nano_vhh:
                raise ValueError(
                    "Input CSV must contain a non-empty 'vhhseq', 'h_seq', or 'hseq' column"
                )
            
            # Prepare Input
            nano_pad_token, nano_pad_region, nano_loc, ms_tokenizer = batch_input_element(
                nano_vhh,
                inpaint_sample=args.inpaint_sample,
                batch_size=batch_size
            )
            
            origin = 'nano'
            name = f"{idx}" # Keep original ID for simpler mapping
            
            # Record Input Sequence (Raw) to this round's CSV
            with open(save_fpath, 'a', encoding='UTF-8') as f:
                f.write(f'{origin},{name},{nano_vhh},{current_seed},{args.temperature}\n')

            if args.sample_order == 'shuffle':
                np.random.shuffle(nano_loc)
            
            # Sampling
            while sample_number > 0 and try_num > 0:
                all_token = ms_tokenizer.toks
                with torch.no_grad():
                    for i in nano_loc:
                        nano_prediction = model(
                            nano_pad_token.to(device),
                            nano_pad_region.to(device),
                            H_chn_type=None
                        )

                        nano_pred = nano_prediction[:, i, :len(all_token)-1]
                        
                        # [NEW] Apply Temperature
                        if args.temperature != 1.0:
                            nano_pred = nano_pred / args.temperature
                            
                        nano_soft = torch.nn.functional.softmax(nano_pred, dim=1)
                        nano_sample = torch.multinomial(nano_soft, num_samples=1)
                        nano_pad_token[:, i] = nano_sample.squeeze()

                nano_untokenized = [ms_tokenizer.idx2seq(s) for s in nano_pad_token]
                
                for _, g_h in enumerate(nano_untokenized):
                    if sample_number == 0:
                        break

                    # round_logger.info(f"ID {idx} Gen: {g_h[:20]}...")
                    with open(save_fpath, 'a', encoding='UTF-8') as f:
                        try:
                            sample_origin = 'humanization'
                            sample_name = f"{idx}_human_sample"
                            # Verify validity
                            test_chain = Chain(g_h, scheme='imgt')
                            
                            f.write(f'{sample_origin},{sample_name},{g_h},{current_seed},{args.temperature}\n')
                            sample_number -= 1
                        except Exception as e:
                            if try_num == 1:
                                sample_origin = 'humanization_failed'
                                sample_name = f"{idx}_failed"
                                f.write(f'{sample_origin},{sample_name},{g_h},{current_seed},{args.temperature}\n')
                                round_logger.warning(f"Failed seq ID {idx}: {e}")
                        try_num -= 1

        # 4. Generate FASTA for this round
        fasta_save_fpath = os.path.join(round_dir, 'sample_identity.fa')
        round_logger.info('Generating FASTA...')
        sample_df = pd.read_csv(save_fpath)
        sample_human_df = sample_df[sample_df['Specific'] == 'humanization'].reset_index(drop=True)
        seqs_to_fasta(sample_human_df, fasta_save_fpath, version=args.fa_version)

        # 5. Split for Structure (if enabled)
        if args.structure:
            split_fasta_for_save(save_fpath)

    logger.info('All Sampling Rounds Completed.')
