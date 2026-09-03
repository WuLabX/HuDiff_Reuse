"""
Patent Evaluation Module
用于专利抗体分析的评价指标
"""

from abnumber import Chain


def cal_all_preservation(chain1, chain2):
    """计算全序列保持率 (Calculate full sequence preservation)"""
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

