import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_target_boxplot(df, target_col, output_path=None):
    """
    Generates Boxplot for target variable.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[target_col], color='skyblue')
    sns.stripplot(x=df[target_col], color='red', alpha=0.6, jitter=True)
    plt.title(f'Distribution: {target_col}')
    plt.xlabel('Value')
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
