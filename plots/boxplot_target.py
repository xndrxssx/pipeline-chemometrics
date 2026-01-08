import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_target_boxplot(y_train, y_test, target_col, output_path=None):
    """
    Generates Horizontal Boxplot comparing Train vs Test distributions with individual points.
    """
    # Create DF for Seaborn
    df_train = pd.DataFrame({target_col: y_train, 'Set': 'Train'})
    df_test = pd.DataFrame({target_col: y_test, 'Set': 'Test'})
    df = pd.concat([df_train, df_test], ignore_index=True)
    
    plt.figure(figsize=(10, 5))
    
    # Horizontal Boxplot
    # 'x' is the numeric value, 'y' is the category
    sns.boxplot(data=df, x=target_col, y='Set', palette="pastel", orient='h', showfliers=False)
    
    # Overlay individual points (Stripplot)
    sns.stripplot(data=df, x=target_col, y='Set', orient='h', color='black', size=4, alpha=0.5, jitter=True)
    
    plt.title(f'Distribution Check: {target_col}')
    plt.xlabel(f'{target_col} Value')
    plt.ylabel('Dataset Partition')
    plt.grid(True, axis='x', linestyle=':', alpha=0.6)
    
    # Add stats annotation
    train_mean = y_train.mean()
    test_mean = y_test.mean()
    stats_text = (f"Train Mean: {train_mean:.2f} (n={len(y_train)})\n"
                  f"Test Mean:  {test_mean:.2f} (n={len(y_test)})")
    
    plt.text(0.98, 0.05, stats_text, transform=plt.gca().transAxes, 
             fontsize=9, ha='right', va='bottom',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
