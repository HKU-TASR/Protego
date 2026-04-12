"""
Retrieval Randomness Analysis Tool
===================================
用于分析同一用户不同query图片得到的检索结果是否呈现随机性/多样性。

主要指标：
1. Jaccard相似度：衡量不同query结果集的重叠程度（越低=越随机）
2. 身份多样性：检索结果中出现的不同身份数量
3. 熵值：衡量身份分布的均匀程度（越高=越随机）
4. 图片重叠率：具体图片在多个query结果中出现的频率

可视化：
1. Query间相似度热力图
2. 身份出现频率条形图
3. UpSet图展示结果集重叠
4. 箱线图展示各query结果的多样性
"""

import pickle
import os
import sys
import argparse
from typing import List, Set, Any, Dict, Tuple
from collections import Counter, defaultdict
from pathlib import Path

import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec


# ============================================================================
# 核心统计指标
# ============================================================================

def pairwise_jaccard(sets: List[Set[Any]]) -> Tuple[float, np.ndarray]:
    """
    计算所有集合对之间的Jaccard相似度
    返回：平均Jaccard值和完整的相似度矩阵
    """
    n = len(sets)
    if n < 2:
        return 1.0, np.array([[1.0]])
    
    jaccard_matrix = np.zeros((n, n))
    total_jaccard = 0.0
    count = 0
    
    for i in range(n):
        jaccard_matrix[i, i] = 1.0
        for j in range(i + 1, n):
            intersection = len(sets[i].intersection(sets[j]))
            union = len(sets[i].union(sets[j]))
            if union > 0:
                jaccard = intersection / union
                jaccard_matrix[i, j] = jaccard
                jaccard_matrix[j, i] = jaccard
                total_jaccard += jaccard
                count += 1
    
    avg_jaccard = total_jaccard / count if count > 0 else 0.0
    return avg_jaccard, jaccard_matrix


def get_entropy(sets: List[Set[Any]]) -> float:
    """计算元素在所有集合中出现频率的熵"""
    element_counts = {}
    total_sets = len(sets)
    for s in sets:
        for element in s:
            if element not in element_counts:
                element_counts[element] = 0
            element_counts[element] += 1
    entropy = 0.0
    for count in element_counts.values():
        p = count / total_sets
        entropy -= p * np.log2(p)
    return entropy


def get_jsdiv(sets: List[Set[Any]]) -> float:
    """计算Jensen-Shannon散度"""
    n = len(sets)
    if n < 2:
        return 0.0
    all_elements = set().union(*sets)
    prob_distributions = []
    for s in sets:
        prob_distribution = np.array([1 if elem in s else 0 for elem in all_elements], dtype=float)
        prob_distribution /= prob_distribution.sum() if prob_distribution.sum() > 0 else 1
        prob_distributions.append(prob_distribution)
    avg_distribution = np.mean(prob_distributions, axis=0)
    js_divergence = 0.0
    for pd in prob_distributions:
        kl_div = np.sum(pd * np.log2((pd + 1e-10) / (avg_distribution + 1e-10)))
        js_divergence += kl_div
    js_divergence /= n
    return js_divergence


# ============================================================================
# 随机性基线和统计检验
# ============================================================================

def compute_random_baseline_jaccard(n_queries: int, k_retrieved: int, 
                                     database_size: int, n_simulations: int = 1000) -> Dict[str, float]:
    """
    通过Monte Carlo模拟计算随机检索的期望Jaccard相似度
    
    Args:
        n_queries: query数量
        k_retrieved: 每个query检索的结果数
        database_size: 数据库大小
        n_simulations: 模拟次数
    
    Returns:
        包含期望值和置信区间的字典
    """
    jaccard_samples = []
    
    for _ in range(n_simulations):
        # 随机生成n_queries个检索结果集
        random_sets = [set(np.random.choice(database_size, k_retrieved, replace=False)) 
                       for _ in range(n_queries)]
        
        # 计算pairwise Jaccard
        jaccards = []
        for i in range(n_queries):
            for j in range(i + 1, n_queries):
                intersection = len(random_sets[i].intersection(random_sets[j]))
                union = len(random_sets[i].union(random_sets[j]))
                if union > 0:
                    jaccards.append(intersection / union)
        
        if jaccards:
            jaccard_samples.append(np.mean(jaccards))
    
    return {
        'mean': np.mean(jaccard_samples),
        'std': np.std(jaccard_samples),
        'ci_lower': np.percentile(jaccard_samples, 2.5),
        'ci_upper': np.percentile(jaccard_samples, 97.5),
    }


def compute_normalized_entropy(counter: Counter, n_queries: int) -> float:
    """
    计算归一化熵 (Normalized Entropy)
    
    H_norm = H_actual / H_max
    H_max = log2(N) where N is the number of unique elements
    
    返回值范围 [0, 1]，接近1表示均匀分布（高随机性）
    """
    n_unique = len(counter)
    if n_unique <= 1:
        return 1.0
    
    # 计算实际熵
    total = sum(counter.values())
    actual_entropy = 0.0
    for count in counter.values():
        p = count / total
        if p > 0:
            actual_entropy -= p * np.log2(p)
    
    # 最大熵（均匀分布）
    max_entropy = np.log2(n_unique)
    
    return actual_entropy / max_entropy if max_entropy > 0 else 1.0


def chi_square_uniformity_test(counter: Counter) -> Dict[str, float]:
    """
    卡方检验：检验分布是否与均匀分布显著不同
    
    Returns:
        chi2: 卡方统计量
        p_value: p值（>0.05表示不能拒绝均匀分布假设）
        is_uniform: 是否可认为是均匀分布 (p > 0.05)
    """
    from scipy import stats
    
    observed = np.array(list(counter.values()))
    n = len(observed)
    total = sum(observed)
    expected = np.full(n, total / n)  # 均匀分布的期望值
    
    # 卡方检验
    chi2, p_value = stats.chisquare(observed, expected)
    
    return {
        'chi2': chi2,
        'p_value': p_value,
        'is_uniform': p_value > 0.05,
        'interpretation': 'Cannot reject uniform distribution' if p_value > 0.05 
                          else 'Distribution differs from uniform'
    }


def calculate_identity_diversity(retrieval_results: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    计算身份多样性相关指标
    
    Args:
        retrieval_results: {query_image: [retrieved_image1, retrieved_image2, ...]}
    
    Returns:
        包含各种多样性指标的字典
    """
    # 提取身份信息
    def get_identity(path: str) -> str:
        return path.split('|')[0]
    
    def get_image_path(path: str) -> str:
        return path.split('|')[1].replace('(protected)', '') if '|' in path else path.replace('(protected)', '')
    
    all_identities = []  # 所有检索结果中的身份
    all_images = []  # 所有检索结果中的具体图片
    identity_sets = []  # 每个query的身份集合
    image_sets = []  # 每个query的图片集合
    
    for query, retrievals in retrieval_results.items():
        query_identities = set()
        query_images = set()
        for r in retrievals:
            identity = get_identity(r)
            image = get_image_path(r)
            all_identities.append(identity)
            all_images.append(image)
            query_identities.add(identity)
            query_images.add(image)
        identity_sets.append(query_identities)
        image_sets.append(query_images)
    
    # 统计
    identity_counter = Counter(all_identities)
    image_counter = Counter(all_images)
    
    # 计算指标
    n_queries = len(retrieval_results)
    n_unique_identities = len(identity_counter)
    n_unique_images = len(image_counter)
    
    # 计算Jaccard相似度
    avg_identity_jaccard, identity_jaccard_matrix = pairwise_jaccard(identity_sets)
    avg_image_jaccard, image_jaccard_matrix = pairwise_jaccard(image_sets)
    
    # 计算每个元素出现在多少个query中的比例
    identity_overlap_rate = np.mean([c / n_queries for c in identity_counter.values()])
    image_overlap_rate = np.mean([c / n_queries for c in image_counter.values()])
    
    # 计算归一化熵
    identity_norm_entropy = compute_normalized_entropy(identity_counter, n_queries)
    image_norm_entropy = compute_normalized_entropy(image_counter, n_queries)
    
    # 计算随机基线（估计数据库大小）
    k_retrieved = len(list(retrieval_results.values())[0]) if retrieval_results else 23
    estimated_db_size = max(n_unique_images * 3, 1000)  # 估计数据库大小
    random_baseline = compute_random_baseline_jaccard(
        n_queries, k_retrieved, estimated_db_size, n_simulations=500
    )
    
    # 卡方检验
    try:
        identity_chi2_test = chi_square_uniformity_test(identity_counter)
        image_chi2_test = chi_square_uniformity_test(image_counter)
    except Exception:
        identity_chi2_test = {'chi2': None, 'p_value': None, 'is_uniform': None}
        image_chi2_test = {'chi2': None, 'p_value': None, 'is_uniform': None}
    
    return {
        'n_queries': n_queries,
        'n_unique_identities': n_unique_identities,
        'n_unique_images': n_unique_images,
        'identity_counter': identity_counter,
        'image_counter': image_counter,
        'identity_sets': identity_sets,
        'image_sets': image_sets,
        'avg_identity_jaccard': avg_identity_jaccard,
        'avg_image_jaccard': avg_image_jaccard,
        'identity_jaccard_matrix': identity_jaccard_matrix,
        'image_jaccard_matrix': image_jaccard_matrix,
        'identity_overlap_rate': identity_overlap_rate,
        'image_overlap_rate': image_overlap_rate,
        'identity_entropy': get_entropy(identity_sets),
        'image_entropy': get_entropy(image_sets),
        # 新增的统计指标
        'identity_norm_entropy': identity_norm_entropy,
        'image_norm_entropy': image_norm_entropy,
        'random_baseline_jaccard': random_baseline,
        'identity_chi2_test': identity_chi2_test,
        'image_chi2_test': image_chi2_test,
        'k_retrieved': k_retrieved,
    }


# ============================================================================
# 可视化函数
# ============================================================================

def plot_jaccard_heatmap(jaccard_matrix: np.ndarray, query_names: List[str], 
                         title: str, save_path: str = None):
    """绘制Jaccard相似度热力图"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 简化query名称（只保留图片编号）
    short_names = [q.split('/')[-1].split('.')[0][-8:] for q in query_names]
    
    sns.heatmap(jaccard_matrix, annot=True if len(query_names) <= 15 else False, 
                fmt='.2f', cmap='RdYlBu_r', vmin=0, vmax=1,
                xticklabels=short_names, yticklabels=short_names, ax=ax)
    
    ax.set_title(f'{title}\n(Mean Jaccard: {jaccard_matrix[np.triu_indices(len(query_names), 1)].mean():.3f})', 
                 fontsize=14)
    ax.set_xlabel('Query Image')
    ax.set_ylabel('Query Image')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_identity_frequency(identity_counter: Counter, n_queries: int, 
                            title: str, top_k: int = 30, save_path: str = None):
    """绘制身份出现频率条形图"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 取出现频率最高的top_k个身份
    top_identities = identity_counter.most_common(top_k)
    identities = [x[0] for x in top_identities]
    counts = [x[1] for x in top_identities]
    
    # 颜色编码：出现在所有query中的用红色，否则用蓝色渐变
    colors = ['#d62728' if c == n_queries else plt.cm.Blues(0.3 + 0.7 * c / n_queries) 
              for c in counts]
    
    bars = ax.bar(range(len(identities)), counts, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=n_queries, color='red', linestyle='--', alpha=0.5, label='All queries')
    ax.axhline(y=n_queries/2, color='orange', linestyle='--', alpha=0.5, label='50% queries')
    
    ax.set_xlabel('Identity', fontsize=12)
    ax.set_ylabel('Appearance Count', fontsize=12)
    ax.set_title(f'{title}\n(Total unique: {len(identity_counter)}, Shown top {top_k})', fontsize=14)
    ax.set_xticks(range(len(identities)))
    ax.set_xticklabels(identities, rotation=45, ha='right', fontsize=8)
    ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_overlap_histogram(counter: Counter, n_queries: int, 
                           title: str, save_path: str = None):
    """绘制元素重叠次数的直方图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    counts = list(counter.values())
    
    # 绘制直方图
    bins = range(1, n_queries + 2)
    ax.hist(counts, bins=bins, edgecolor='black', alpha=0.7, color='steelblue', align='left')
    
    # 添加统计信息
    unique_once = sum(1 for c in counts if c == 1)  # 只出现一次的数量
    appear_all = sum(1 for c in counts if c == n_queries)  # 出现在所有query中的数量
    
    ax.axvline(x=np.mean(counts), color='red', linestyle='--', 
               label=f'Mean: {np.mean(counts):.2f}')
    ax.axvline(x=np.median(counts), color='orange', linestyle='--', 
               label=f'Median: {np.median(counts):.0f}')
    
    # 添加文本标注
    textstr = f'Unique elements: {len(counts)}\n'
    textstr += f'Appear once: {unique_once} ({100*unique_once/len(counts):.1f}%)\n'
    textstr += f'Appear in all: {appear_all} ({100*appear_all/len(counts):.1f}%)'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    ax.set_xlabel('Number of Queries Containing Element', fontsize=12)
    ax.set_ylabel('Number of Elements', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.set_xticks(range(1, n_queries + 1))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_diversity_summary(stats: Dict[str, Any], title: str, save_path: str = None):
    """绘制多样性指标汇总图"""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. 指标表格
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
    metrics_data = [
        ['Metric', 'Identity', 'Image'],
        ['Unique Count', f"{stats['n_unique_identities']}", f"{stats['n_unique_images']}"],
        ['Avg Jaccard', f"{stats['avg_identity_jaccard']:.3f}", f"{stats['avg_image_jaccard']:.3f}"],
        ['Overlap Rate', f"{stats['identity_overlap_rate']:.3f}", f"{stats['image_overlap_rate']:.3f}"],
        ['Entropy', f"{stats['identity_entropy']:.2f}", f"{stats['image_entropy']:.2f}"],
    ]
    
    table = ax1.table(cellText=metrics_data, loc='center', cellLoc='center',
                      colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # 设置表头样式
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax1.set_title('Diversity Metrics Summary', fontsize=12, fontweight='bold', pad=20)
    
    # 2. Jaccard分布对比
    ax2 = fig.add_subplot(gs[0, 1])
    identity_jaccards = stats['identity_jaccard_matrix'][np.triu_indices(stats['n_queries'], 1)]
    image_jaccards = stats['image_jaccard_matrix'][np.triu_indices(stats['n_queries'], 1)]
    
    bp = ax2.boxplot([identity_jaccards, image_jaccards], labels=['Identity', 'Image'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('#5B9BD5')
    bp['boxes'][1].set_facecolor('#ED7D31')
    
    ax2.set_ylabel('Jaccard Similarity')
    ax2.set_title('Pairwise Jaccard Distribution', fontsize=12)
    ax2.set_ylim(0, 1)
    
    # 3. 重叠次数分布
    ax3 = fig.add_subplot(gs[0, 2])
    identity_counts = list(stats['identity_counter'].values())
    image_counts = list(stats['image_counter'].values())
    
    ax3.hist(identity_counts, bins=range(1, stats['n_queries'] + 2), alpha=0.6, 
             label='Identity', color='#5B9BD5', align='left')
    ax3.hist(image_counts, bins=range(1, stats['n_queries'] + 2), alpha=0.6,
             label='Image', color='#ED7D31', align='left')
    ax3.set_xlabel('Appearance Count')
    ax3.set_ylabel('Number of Elements')
    ax3.set_title('Overlap Distribution', fontsize=12)
    ax3.legend()
    
    # 4. Top身份频率
    ax4 = fig.add_subplot(gs[1, :2])
    top_identities = stats['identity_counter'].most_common(20)
    identities = [x[0] for x in top_identities]
    counts = [x[1] for x in top_identities]
    
    colors = ['#d62728' if c == stats['n_queries'] else '#5B9BD5' for c in counts]
    ax4.barh(range(len(identities)), counts, color=colors)
    ax4.set_yticks(range(len(identities)))
    ax4.set_yticklabels(identities, fontsize=9)
    ax4.set_xlabel('Appearance Count')
    ax4.set_title(f'Top 20 Identities (Total unique: {stats["n_unique_identities"]})', fontsize=12)
    ax4.axvline(x=stats['n_queries'], color='red', linestyle='--', alpha=0.5)
    ax4.invert_yaxis()
    
    # 5. 随机性解释
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    # 判断随机性程度
    jaccard_level = "high" if stats['avg_identity_jaccard'] < 0.3 else ("moderate" if stats['avg_identity_jaccard'] < 0.6 else "low")
    
    explanation = f"""
    Randomness Analysis:
    
    * Identity Jaccard: {stats['avg_identity_jaccard']:.3f}
      -> Overlap is {jaccard_level}
      
    * Image Jaccard: {stats['avg_image_jaccard']:.3f}
      -> Very low image overlap
      
    * Unique identities: {stats['n_unique_identities']}
      (from {stats['n_queries']} queries)
      
    * Interpretation:
      Jaccard < 0.3 = High randomness
      Jaccard 0.3-0.6 = Moderate
      Jaccard > 0.6 = Low randomness
    """
    
    ax5.text(0.1, 0.9, explanation, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_query_retrieval_matrix(retrieval_results: Dict[str, List[str]], 
                                title: str, save_path: str = None, top_k_identities: int = 30):
    """
    绘制Query-Identity矩阵图
    展示每个query检索到哪些身份（以及检索到几张该身份的图片）
    """
    def get_identity(path: str) -> str:
        return path.split('|')[0]
    
    # 收集所有身份
    all_identities = set()
    query_identity_counts = {}  # {query: {identity: count}}
    
    for query, retrievals in retrieval_results.items():
        identity_count = Counter([get_identity(r) for r in retrievals])
        query_identity_counts[query] = identity_count
        all_identities.update(identity_count.keys())
    
    # 按总出现次数排序身份
    identity_total = Counter()
    for ic in query_identity_counts.values():
        identity_total.update(ic)
    
    top_identities = [x[0] for x in identity_total.most_common(top_k_identities)]
    
    # 构建矩阵
    queries = list(retrieval_results.keys())
    matrix = np.zeros((len(queries), len(top_identities)))
    
    for i, query in enumerate(queries):
        for j, identity in enumerate(top_identities):
            matrix[i, j] = query_identity_counts[query].get(identity, 0)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(16, max(8, len(queries) * 0.4)))
    
    # 使用自定义颜色映射
    cmap = plt.cm.YlOrRd
    cmap.set_under('white')
    
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0.5)
    
    # 简化query名称
    short_queries = [q.split('/')[-1].split('.')[0][-12:] for q in queries]
    
    ax.set_xticks(range(len(top_identities)))
    ax.set_xticklabels(top_identities, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(queries)))
    ax.set_yticklabels(short_queries, fontsize=8)
    
    ax.set_xlabel('Retrieved Identity', fontsize=12)
    ax.set_ylabel('Query Image', fontsize=12)
    ax.set_title(f'{title}\n(Showing top {top_k_identities} identities by frequency)', fontsize=14)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Number of Retrieved Images')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


# ============================================================================
# 数据加载和主函数
# ============================================================================

def load_yaml_retrieval(yaml_path: str, scenario: str = '2a') -> Dict[str, List[str]]:
    """
    从YAML文件加载检索结果
    
    Args:
        yaml_path: YAML文件路径
        scenario: 场景名称 (如 '2a', '2b')
    
    Returns:
        {query_image_path: [retrieved_image1, retrieved_image2, ...]}
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if scenario not in data:
        raise ValueError(f"Scenario '{scenario}' not found. Available: {list(data.keys())}")
    
    return data[scenario]


def analyze_retrieval_randomness(yaml_path: str, scenario: str = '2a', 
                                  output_dir: str = None, user_name: str = None):
    """
    完整分析检索结果的随机性
    
    Args:
        yaml_path: YAML文件路径
        scenario: 场景名称
        output_dir: 输出目录
        user_name: 用户名（用于标题）
    """
    print(f"\n{'='*60}")
    print(f"Analyzing Retrieval Randomness")
    print(f"{'='*60}")
    print(f"File: {yaml_path}")
    print(f"Scenario: {scenario}")
    
    # 加载数据
    retrieval_results = load_yaml_retrieval(yaml_path, scenario)
    
    if user_name is None:
        # 从文件名或query中提取用户名
        first_query = list(retrieval_results.keys())[0]
        user_name = first_query.split('|')[0] if '|' in first_query else Path(yaml_path).stem
    
    print(f"User: {user_name}")
    print(f"Number of queries: {len(retrieval_results)}")
    
    # 计算统计指标
    stats = calculate_identity_diversity(retrieval_results)
    
    # 打印统计结果
    print(f"\n{'='*40}")
    print("DIVERSITY METRICS")
    print(f"{'='*40}")
    print(f"Number of unique identities: {stats['n_unique_identities']}")
    print(f"Number of unique images: {stats['n_unique_images']}")
    print(f"\nIdentity-level:")
    print(f"  - Avg Jaccard Similarity: {stats['avg_identity_jaccard']:.4f}")
    print(f"  - Overlap Rate: {stats['identity_overlap_rate']:.4f}")
    print(f"  - Entropy: {stats['identity_entropy']:.4f}")
    print(f"  - Normalized Entropy: {stats['identity_norm_entropy']:.4f}")
    print(f"\nImage-level:")
    print(f"  - Avg Jaccard Similarity: {stats['avg_image_jaccard']:.4f}")
    print(f"  - Overlap Rate: {stats['image_overlap_rate']:.4f}")
    print(f"  - Entropy: {stats['image_entropy']:.4f}")
    print(f"  - Normalized Entropy: {stats['image_norm_entropy']:.4f}")
    
    # 随机基线对比
    print(f"\n{'='*40}")
    print("RANDOM BASELINE COMPARISON")
    print(f"{'='*40}")
    rb = stats['random_baseline_jaccard']
    print(f"Random Baseline Jaccard (Monte Carlo, n=500):")
    print(f"  - Expected: {rb['mean']:.4f} ± {rb['std']:.4f}")
    print(f"  - 95% CI: [{rb['ci_lower']:.4f}, {rb['ci_upper']:.4f}]")
    print(f"  - Actual Identity Jaccard: {stats['avg_identity_jaccard']:.4f}")
    
    if stats['avg_identity_jaccard'] <= rb['ci_upper']:
        print(f"  ✓ PASS: Actual Jaccard is within or below random baseline")
    else:
        print(f"  ✗ FAIL: Actual Jaccard exceeds random baseline")
    
    # 统计检验结果
    print(f"\n{'='*40}")
    print("STATISTICAL TESTS")
    print(f"{'='*40}")
    id_test = stats['identity_chi2_test']
    img_test = stats['image_chi2_test']
    
    print(f"Chi-Square Uniformity Test (H0: distribution is uniform):")
    if id_test['p_value'] is not None:
        print(f"  Identity: χ²={id_test['chi2']:.2f}, p={id_test['p_value']:.4f}")
        print(f"            → {id_test['interpretation']}")
    if img_test['p_value'] is not None:
        print(f"  Image:    χ²={img_test['chi2']:.2f}, p={img_test['p_value']:.4f}")
        print(f"            → {img_test['interpretation']}")
    
    print(f"\nNormalized Entropy (1.0 = perfectly uniform):")
    print(f"  Identity: {stats['identity_norm_entropy']:.4f}", end="")
    print(f" {'✓' if stats['identity_norm_entropy'] > 0.9 else '○' if stats['identity_norm_entropy'] > 0.7 else '✗'}")
    print(f"  Image:    {stats['image_norm_entropy']:.4f}", end="")
    print(f" {'✓' if stats['image_norm_entropy'] > 0.9 else '○' if stats['image_norm_entropy'] > 0.7 else '✗'}")
    
    # 随机性判断
    print(f"\n{'='*40}")
    print("RANDOMNESS ASSESSMENT")
    print(f"{'='*40}")
    
    # 综合判断
    is_random = (
        stats['avg_identity_jaccard'] <= rb['ci_upper'] * 1.5 and  # Jaccard接近随机基线
        stats['identity_norm_entropy'] > 0.7  # 归一化熵较高
    )
    
    if stats['avg_identity_jaccard'] < 0.3:
        print("✓ HIGH RANDOMNESS: Different queries retrieve very different identities")
    elif stats['avg_identity_jaccard'] < 0.6:
        print("○ MODERATE RANDOMNESS: Some overlap between query results")
    else:
        print("✗ LOW RANDOMNESS: Query results are highly similar")
    
    if stats['avg_image_jaccard'] < 0.1:
        print("✓ Image-level: Extremely low overlap (different specific images)")
    
    # 生成可视化
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = f"{user_name}_{scenario}"
        
        print(f"\nGenerating visualizations in: {output_dir}")
        
        # 1. 汇总图
        plot_diversity_summary(
            stats, 
            f"Retrieval Randomness Analysis: {user_name} ({scenario})",
            os.path.join(output_dir, f"{base_name}_summary.png")
        )
        
        # 2. 身份Jaccard热力图
        query_names = list(retrieval_results.keys())
        plot_jaccard_heatmap(
            stats['identity_jaccard_matrix'],
            query_names,
            f"Identity Jaccard Similarity: {user_name}",
            os.path.join(output_dir, f"{base_name}_identity_jaccard.png")
        )
        
        # 3. 图片Jaccard热力图
        plot_jaccard_heatmap(
            stats['image_jaccard_matrix'],
            query_names,
            f"Image Jaccard Similarity: {user_name}",
            os.path.join(output_dir, f"{base_name}_image_jaccard.png")
        )
        
        # 4. 身份频率图
        plot_identity_frequency(
            stats['identity_counter'],
            stats['n_queries'],
            f"Identity Frequency: {user_name}",
            save_path=os.path.join(output_dir, f"{base_name}_identity_freq.png")
        )
        
        # 5. 重叠直方图
        plot_overlap_histogram(
            stats['identity_counter'],
            stats['n_queries'],
            f"Identity Overlap Distribution: {user_name}",
            os.path.join(output_dir, f"{base_name}_identity_overlap_hist.png")
        )
        
        plot_overlap_histogram(
            stats['image_counter'],
            stats['n_queries'],
            f"Image Overlap Distribution: {user_name}",
            os.path.join(output_dir, f"{base_name}_image_overlap_hist.png")
        )
        
        # 6. Query-Identity矩阵
        plot_query_retrieval_matrix(
            retrieval_results,
            f"Query-Identity Matrix: {user_name}",
            os.path.join(output_dir, f"{base_name}_query_identity_matrix.png")
        )
    
    return stats


def compare_multiple_users(yaml_paths: List[str], scenario: str = '2a', 
                           output_dir: str = None):
    """
    比较多个用户的检索随机性
    """
    all_stats = []
    user_names = []
    
    for yaml_path in yaml_paths:
        user_name = Path(yaml_path).stem.split('_')[-2]  # 假设文件名格式包含用户名
        user_names.append(user_name)
        
        retrieval_results = load_yaml_retrieval(yaml_path, scenario)
        stats = calculate_identity_diversity(retrieval_results)
        stats['user'] = user_name
        all_stats.append(stats)
    
    # 绘制对比图
    if output_dir and len(all_stats) > 1:
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Jaccard对比
        identity_jaccards = [s['avg_identity_jaccard'] for s in all_stats]
        image_jaccards = [s['avg_image_jaccard'] for s in all_stats]
        
        x = range(len(user_names))
        width = 0.35
        
        axes[0].bar([i - width/2 for i in x], identity_jaccards, width, label='Identity', color='#5B9BD5')
        axes[0].bar([i + width/2 for i in x], image_jaccards, width, label='Image', color='#ED7D31')
        axes[0].set_ylabel('Avg Jaccard Similarity')
        axes[0].set_title('Jaccard Similarity Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(user_names, rotation=45, ha='right')
        axes[0].legend()
        axes[0].set_ylim(0, 1)
        
        # 唯一身份数对比
        unique_identities = [s['n_unique_identities'] for s in all_stats]
        unique_images = [s['n_unique_images'] for s in all_stats]
        
        axes[1].bar([i - width/2 for i in x], unique_identities, width, label='Identities', color='#5B9BD5')
        axes[1].bar([i + width/2 for i in x], unique_images, width, label='Images', color='#ED7D31')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Unique Elements Count')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(user_names, rotation=45, ha='right')
        axes[1].legend()
        
        # 熵对比
        identity_entropy = [s['identity_entropy'] for s in all_stats]
        image_entropy = [s['image_entropy'] for s in all_stats]
        
        axes[2].bar([i - width/2 for i in x], identity_entropy, width, label='Identity', color='#5B9BD5')
        axes[2].bar([i + width/2 for i in x], image_entropy, width, label='Image', color='#ED7D31')
        axes[2].set_ylabel('Entropy')
        axes[2].set_title('Diversity Entropy')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(user_names, rotation=45, ha='right')
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'user_comparison.png'), dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {output_dir}")
        plt.close()
    
    return all_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze retrieval randomness for a user')
    parser.add_argument('yaml_path', type=str, help='Path to YAML retrieval results file')
    parser.add_argument('--scenario', type=str, default='2a', help='Scenario name (default: 2a)')
    parser.add_argument('--output', type=str, default=None, help='Output directory for plots')
    parser.add_argument('--user', type=str, default=None, help='User name for title')
    
    args = parser.parse_args()
    
    # 默认输出目录
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.yaml_path), 'randomness_analysis')
    
    analyze_retrieval_randomness(
        args.yaml_path,
        scenario=args.scenario,
        output_dir=args.output,
        user_name=args.user
    )
