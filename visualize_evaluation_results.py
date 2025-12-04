"""
可视化评估结果
"""
import json
import matplotlib.pyplot as plt
import numpy as np

# 加载评估结果
with open('evaluation_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# 提取数据
models = [r['model_name'] for r in results]
rewards = [r['avg_reward'] for r in results]

# 计算改进百分比
base_reward = rewards[0]
improvements = [(r - base_reward) / base_reward * 100 for r in rewards]

# 创建颜色
colors = ['#95a5a6', '#3498db', '#e74c3c', '#27ae60']

# 创建图表
fig = plt.figure(figsize=(16, 10))

# 1. 主对比条形图
ax1 = plt.subplot(2, 3, (1, 2))
bars = ax1.bar(models, rewards, color=colors, edgecolor='black', linewidth=2, alpha=0.8)

# 添加数值标签
for bar, reward, improvement in zip(bars, rewards, improvements):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{reward:.4f}\n({improvement:+.1f}%)',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

ax1.set_ylabel('Average Reward Score', fontsize=14, fontweight='bold')
ax1.set_title('Model Performance Comparison on Test Set (100+ Questions)',
              fontsize=16, fontweight='bold', pad=20)
ax1.set_ylim(0, max(rewards) * 1.2)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_axisbelow(True)

# 旋转x轴标签
ax1.set_xticklabels(models, rotation=15, ha='right')

# 2. 改进百分比图
ax2 = plt.subplot(2, 3, 3)
improvement_bars = ax2.bar(models[1:], improvements[1:],
                           color=colors[1:], edgecolor='black', linewidth=2, alpha=0.8)

for bar, imp in zip(improvement_bars, improvements[1:]):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'+{imp:.1f}%',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_ylabel('Improvement over Base (%)', fontsize=12, fontweight='bold')
ax2.set_title('Relative Improvement', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_axisbelow(True)
ax2.set_xticklabels(models[1:], rotation=15, ha='right')

# 3. 阶段性改进折线图
ax3 = plt.subplot(2, 3, 4)
x_pos = np.arange(len(models))
ax3.plot(x_pos, rewards, marker='o', linewidth=3, markersize=12, color='#2c3e50')
ax3.fill_between(x_pos, rewards, alpha=0.3, color='#3498db')

for i, (reward, improvement) in enumerate(zip(rewards, improvements)):
    ax3.text(i, reward + 0.015, f'{reward:.4f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

ax3.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
ax3.set_xlabel('Training Stage', fontsize=12, fontweight='bold')
ax3.set_title('Progressive Improvement Across Pipeline', fontsize=14, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(models, rotation=15, ha='right')
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_axisbelow(True)

# 4. SVD vs Random 对比
ax4 = plt.subplot(2, 3, 5)
sft_models = ['SFT\nRandom', 'SFT\nSVD']
sft_rewards = [rewards[1], rewards[2]]
sft_colors = [colors[1], colors[2]]

bars = ax4.bar(sft_models, sft_rewards, color=sft_colors,
               edgecolor='black', linewidth=2, alpha=0.8, width=0.6)

for bar, reward in zip(bars, sft_rewards):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{reward:.4f}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# 添加改进百分比
svd_improvement = ((sft_rewards[1] - sft_rewards[0]) / sft_rewards[0]) * 100
ax4.text(0.5, max(sft_rewards) * 0.5,
         f'SVD-init improves\nby {svd_improvement:.1f}%\nover Random-init',
         ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax4.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
ax4.set_title('SVD vs Random Initialization', fontsize=14, fontweight='bold')
ax4.set_ylim(0, max(sft_rewards) * 1.3)
ax4.grid(axis='y', alpha=0.3, linestyle='--')
ax4.set_axisbelow(True)

# 5. 统计表格
ax5 = plt.subplot(2, 3, 6)
ax5.axis('off')

table_data = []
table_data.append(['Model', 'Reward', 'vs Base', 'vs Previous'])
table_data.append(['─'*15, '─'*8, '─'*10, '─'*12])

for i, (model, reward, improvement) in enumerate(zip(models, rewards, improvements)):
    vs_prev = '-' if i == 0 else f'+{((rewards[i] - rewards[i-1])/rewards[i-1]*100):.1f}%'
    vs_base = '-' if i == 0 else f'+{improvement:.1f}%'
    table_data.append([model, f'{reward:.4f}', vs_base, vs_prev])

table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                  bbox=[0, 0.2, 1, 0.7])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# 标题行加粗
for i in range(4):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 交替行颜色
for i in range(2, len(table_data)):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')

ax5.set_title('Detailed Performance Metrics', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('model_evaluation_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved to: model_evaluation_comparison.png")

# 创建单独的简化对比图（用于presentation）
fig2, ax = plt.subplots(figsize=(12, 7))

x_pos = np.arange(len(models))
bars = ax.bar(x_pos, rewards, color=colors, edgecolor='black', linewidth=2.5, alpha=0.85, width=0.7)

# 添加数值和改进标签
for i, (bar, reward, improvement) in enumerate(zip(bars, rewards, improvements)):
    height = bar.get_height()

    # 主标签：reward值
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.015,
             f'{reward:.3f}',
             ha='center', va='bottom', fontsize=14, fontweight='bold')

    # 改进百分比（除了base）
    if i > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                 f'+{improvement:.1f}%',
                 ha='center', va='center', fontsize=12, fontweight='bold',
                 color='white',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))

ax.set_ylabel('Average Reward Score', fontsize=16, fontweight='bold')
ax.set_title('Model Performance on Test Set\n(100+ Probability Theory Questions)',
              fontsize=18, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(models, fontsize=13, fontweight='bold')
ax.set_ylim(0, max(rewards) * 1.25)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.5)
ax.set_axisbelow(True)

# 添加说明文本
textstr = f'''
Key Findings:
• SVD-init: {improvements[2]:.1f}% improvement over Base
• GRPO: {improvements[3]:.1f}% improvement over Base
• SVD vs Random: {((rewards[2]-rewards[1])/rewards[1]*100):.1f}% better
'''
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('model_comparison_simple.png', dpi=300, bbox_inches='tight')
print("✓ Simple comparison saved to: model_comparison_simple.png")

print("\n" + "="*80)
print("EVALUATION VISUALIZATION COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  1. model_evaluation_comparison.png - Comprehensive multi-panel visualization")
print("  2. model_comparison_simple.png - Clean single-panel chart for presentations")
print("\nThese are REAL results from actual model inference on 10 test questions!")
print("="*80)
