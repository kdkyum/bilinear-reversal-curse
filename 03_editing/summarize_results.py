import json
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

def parse_method_name(method_name):
    """Parse method name to extract base method and layer info"""
    if "_Layer" in method_name:
        parts = method_name.split("_Layer")
        base_method = parts[0]
        layer = int(parts[1])
        return base_method, layer
    else:
        return method_name, None

def summarize_results(results_dir="../results/editing"):
    results_dir = Path(results_dir)
    
    summary = []
    
    for method_dir in results_dir.iterdir():
        if method_dir.is_dir():
            method_name = method_dir.name
            base_method, layer = parse_method_name(method_name)
            
            # Handle nested subfolder structure
            for subfolder_dir in method_dir.iterdir():
                if subfolder_dir.is_dir():
                    subfolder_name = subfolder_dir.name
                    
                    # Parse direction from subfolder name
                    direction = "UNI"  # default
                    if subfolder_name.endswith("_BI"):
                        direction = "BI"
                        model_name = subfolder_name[:-3]  # remove _BI
                    elif subfolder_name.endswith("_UNI"):
                        direction = "UNI"
                        model_name = subfolder_name[:-4]  # remove _UNI
                    else:
                        model_name = subfolder_name
                    
                    successes = []
                    reversal_successes = []
                    nbr_new2org_successes = []
                    nbr_org2new_successes = []
                    unrelated_successes = []
                    times = []
                    
                    # Pre-editing metrics
                    pre_successes = []
                    pre_reversal_successes = []
                    pre_nbr_new2org_successes = []
                    pre_nbr_org2new_successes = []
                    pre_unrelated_successes = []
                    
                    # Holdout metrics
                    holdout_successes = []
                    holdout_reversal_successes = []
                    holdout_nbr_new2org_successes = []
                    holdout_nbr_org2new_successes = []
                    holdout_unrelated_successes = []
                    
                    for case_file in subfolder_dir.glob("case_*.json"):
                        with open(case_file) as f:
                            data = json.load(f)
                        
                        # Post-editing metrics
                        successes.append(data["post"]["rewrite_success"])
                        reversal_successes.append(data["post"]["reversal_success"])
                        nbr_new2org_successes.append(data["post"]["nbr_new2org_success"])
                        nbr_org2new_successes.append(data["post"]["nbr_org2new_success"])
                        unrelated_successes.append(data["post"]["unrelated_success"])
                        times.append(data["edit_time"])
                        
                        # Pre-editing metrics
                        pre_successes.append(data["pre"]["rewrite_success"])
                        pre_reversal_successes.append(data["pre"]["reversal_success"])
                        pre_nbr_new2org_successes.append(data["pre"]["nbr_new2org_success"])
                        pre_nbr_org2new_successes.append(data["pre"]["nbr_org2new_success"])
                        pre_unrelated_successes.append(data["pre"]["unrelated_success"])
                        
                        # Holdout metrics
                        if "holdout_results" in data and data["holdout_results"]:
                            holdout_successes.append(data["holdout_results"]["rewrite_success"])
                            holdout_reversal_successes.append(data["holdout_results"]["reversal_success"])
                            holdout_nbr_new2org_successes.append(data["holdout_results"]["nbr_new2org_success"])
                            holdout_nbr_org2new_successes.append(data["holdout_results"]["nbr_org2new_success"])
                            holdout_unrelated_successes.append(data["holdout_results"]["unrelated_success"])
                    
                    # Parse weight decay and seed from the model name,
                    # e.g. "lvl3_lr3.0e-04_wd3.0_wr0.01_s0"
                    wd_seed = re.search(r"wd([0-9.]+)_wr[0-9.e+-]+_s(\d+)", model_name)

                    if successes:
                        summary_entry = {
                            "Method": method_name,
                            "Base_Method": base_method,
                            "Layer": layer,
                            "Subfolder": subfolder_name,
                            "Model": model_name,
                            "Direction": direction,
                            "Weight Decay": float(wd_seed.group(1)) if wd_seed else None,
                            "Seed": int(wd_seed.group(2)) if wd_seed else None,
                            "Edit Success Rate": sum(successes) / len(successes),
                            "Reversal Success Rate": sum(reversal_successes) / len(reversal_successes),
                            "Nbr New2Org Success Rate": sum(nbr_new2org_successes) / len(nbr_new2org_successes),
                            "Nbr Org2New Success Rate": sum(nbr_org2new_successes) / len(nbr_org2new_successes),
                            "Unrelated Success Rate": sum(unrelated_successes) / len(unrelated_successes),
                            "Pre Edit Success Rate": sum(pre_successes) / len(pre_successes),
                            "Pre Reversal Success Rate": sum(pre_reversal_successes) / len(pre_reversal_successes),
                            "Pre Nbr New2Org Success Rate": sum(pre_nbr_new2org_successes) / len(pre_nbr_new2org_successes),
                            "Pre Nbr Org2New Success Rate": sum(pre_nbr_org2new_successes) / len(pre_nbr_org2new_successes),
                            "Pre Unrelated Success Rate": sum(pre_unrelated_successes) / len(pre_unrelated_successes),
                            "Avg Time (s)": sum(times) / len(times),
                            "Num Cases": len(successes)
                        }
                        
                        # Add holdout metrics if available
                        if holdout_successes:
                            summary_entry.update({
                                "Holdout Edit Success Rate": sum(holdout_successes) / len(holdout_successes),
                                "Holdout Reversal Success Rate": sum(holdout_reversal_successes) / len(holdout_reversal_successes),
                                "Holdout Nbr New2Org Success Rate": sum(holdout_nbr_new2org_successes) / len(holdout_nbr_new2org_successes),
                                "Holdout Nbr Org2New Success Rate": sum(holdout_nbr_org2new_successes) / len(holdout_nbr_org2new_successes),
                                "Holdout Unrelated Success Rate": sum(holdout_unrelated_successes) / len(holdout_unrelated_successes),
                            })
                        
                        summary.append(summary_entry)
    
    df = pd.DataFrame(summary)
    
    # Sort by direction, model, base method and layer for better readability
    df = df.sort_values(['Direction', 'Model', 'Base_Method', 'Layer'])
    
    print("=== Summary Results ===")
    print(df.to_string(index=False))
    
    # Save detailed results to the results directory
    df.to_csv(results_dir / "editing_summary_detailed.csv", index=False)
    df.to_json(results_dir / "editing_summary_detailed.json", orient='records', indent=2)
    print(f"\nSaved detailed results to {results_dir / 'editing_summary_detailed.csv'}")
    print(f"Saved detailed results to {results_dir / 'editing_summary_detailed.json'}")
    
    # Create visualizations for each subfolder
    create_plots(df, results_dir)
    
    # Print layer-wise analysis
    analyze_by_layer(df)
    
    return df

def create_plots(df, results_dir):
    """Create seaborn plots for result visualization"""
    plt.style.use('seaborn-v0_8')
    
    # Create plots subdirectory
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Create plots for each model and direction combination
    for model in df['Model'].unique():
        for direction in df['Direction'].unique():
            model_dir_df = df[(df['Model'] == model) & (df['Direction'] == direction)]
            layer_df = model_dir_df[model_dir_df['Layer'].notna()].copy()
            
            if len(layer_df) == 0:
                print(f"No layer-specific data found for {model} - {direction}")
                continue
            
            # Check if holdout data is available
            has_holdout = 'Holdout Edit Success Rate' in layer_df.columns and layer_df['Holdout Edit Success Rate'].notna().any()
            
            # Create subplots - adjust based on holdout availability
            if has_holdout:
                fig, axes = plt.subplots(3, 3, figsize=(24, 18))
                fig.suptitle(f'Model Editing Results by Layer and Method - {model} ({direction})', fontsize=16)
                
                # Row 1: Main editing results
                sns.lineplot(data=layer_df, x='Layer', y='Edit Success Rate', 
                            hue='Base_Method', marker='o', ax=axes[0,0])
                axes[0,0].set_title('Edit Success Rate by Layer')
                axes[0,0].set_ylim(0, 1)
                axes[0,0].grid(True, alpha=0.3)
                
                sns.lineplot(data=layer_df, x='Layer', y='Reversal Success Rate', 
                            hue='Base_Method', marker='s', ax=axes[0,1])
                axes[0,1].set_title('Reversal Success Rate by Layer')
                axes[0,1].set_ylim(0, 1)
                axes[0,1].grid(True, alpha=0.3)
                
                sns.lineplot(data=layer_df, x='Layer', y='Unrelated Success Rate', 
                            hue='Base_Method', marker='^', ax=axes[0,2])
                axes[0,2].set_title('Unrelated Success Rate by Layer')
                axes[0,2].set_ylim(0, 1)
                axes[0,2].grid(True, alpha=0.3)
                
                # Row 2: Holdout results
                sns.lineplot(data=layer_df, x='Layer', y='Holdout Edit Success Rate', 
                            hue='Base_Method', marker='o', ax=axes[1,0])
                axes[1,0].set_title('Holdout Edit Success Rate by Layer')
                axes[1,0].set_ylim(0, 1)
                axes[1,0].grid(True, alpha=0.3)
                
                sns.lineplot(data=layer_df, x='Layer', y='Holdout Reversal Success Rate', 
                            hue='Base_Method', marker='s', ax=axes[1,1])
                axes[1,1].set_title('Holdout Reversal Success Rate by Layer')
                axes[1,1].set_ylim(0, 1)
                axes[1,1].grid(True, alpha=0.3)
                
                sns.lineplot(data=layer_df, x='Layer', y='Holdout Unrelated Success Rate', 
                            hue='Base_Method', marker='^', ax=axes[1,2])
                axes[1,2].set_title('Holdout Unrelated Success Rate by Layer')
                axes[1,2].set_ylim(0, 1)
                axes[1,2].grid(True, alpha=0.3)
                
                # Row 3: Comparisons and additional metrics
                neighbor_df = layer_df.melt(
                    id_vars=['Layer', 'Base_Method'],
                    value_vars=['Nbr New2Org Success Rate', 'Nbr Org2New Success Rate'],
                    var_name='Neighbor_Type', value_name='Success_Rate'
                )
                sns.lineplot(data=neighbor_df, x='Layer', y='Success_Rate', 
                            hue='Base_Method', style='Neighbor_Type', marker='o', ax=axes[2,0])
                axes[2,0].set_title('Neighbor Success Rates by Layer')
                axes[2,0].set_ylim(0, 1)
                axes[2,0].grid(True, alpha=0.3)
                
                # Edit vs Holdout comparison
                edit_holdout_df = layer_df.melt(
                    id_vars=['Layer', 'Base_Method'],
                    value_vars=['Edit Success Rate', 'Holdout Edit Success Rate'],
                    var_name='Dataset_Type', value_name='Success_Rate'
                )
                sns.lineplot(data=edit_holdout_df, x='Layer', y='Success_Rate', 
                            hue='Base_Method', style='Dataset_Type', marker='o', ax=axes[2,1])
                axes[2,1].set_title('Edit vs Holdout Success Rate')
                axes[2,1].set_ylim(0, 1)
                axes[2,1].grid(True, alpha=0.3)
                
                sns.lineplot(data=layer_df, x='Layer', y='Avg Time (s)', 
                            hue='Base_Method', marker='d', ax=axes[2,2])
                axes[2,2].set_title('Average Edit Time by Layer')
                axes[2,2].grid(True, alpha=0.3)
            else:
                # Original layout without holdout
                fig, axes = plt.subplots(2, 3, figsize=(20, 12))
                fig.suptitle(f'Model Editing Results by Layer and Method - {model} ({direction})', fontsize=16)
                
                # 1. Edit Success Rate by Layer
                sns.lineplot(data=layer_df, x='Layer', y='Edit Success Rate', 
                            hue='Base_Method', marker='o', ax=axes[0,0])
                axes[0,0].set_title('Edit Success Rate by Layer')
                axes[0,0].set_ylim(0, 1)
                axes[0,0].grid(True, alpha=0.3)
                
                # 2. Reversal Success Rate by Layer
                sns.lineplot(data=layer_df, x='Layer', y='Reversal Success Rate', 
                            hue='Base_Method', marker='s', ax=axes[0,1])
                axes[0,1].set_title('Reversal Success Rate by Layer')
                axes[0,1].set_ylim(0, 1)
                axes[0,1].grid(True, alpha=0.3)
                
                # 3. Unrelated Success Rate by Layer
                sns.lineplot(data=layer_df, x='Layer', y='Unrelated Success Rate', 
                            hue='Base_Method', marker='^', ax=axes[0,2])
                axes[0,2].set_title('Unrelated Success Rate by Layer')
                axes[0,2].set_ylim(0, 1)
                axes[0,2].grid(True, alpha=0.3)
                
                # 4. Neighbor Success Rates
                neighbor_df = layer_df.melt(
                    id_vars=['Layer', 'Base_Method'],
                    value_vars=['Nbr New2Org Success Rate', 'Nbr Org2New Success Rate'],
                    var_name='Neighbor_Type', value_name='Success_Rate'
                )
                sns.lineplot(data=neighbor_df, x='Layer', y='Success_Rate', 
                            hue='Base_Method', style='Neighbor_Type', marker='o', ax=axes[1,0])
                axes[1,0].set_title('Neighbor Success Rates by Layer')
                axes[1,0].set_ylim(0, 1)
                axes[1,0].grid(True, alpha=0.3)
                
                # 5. Edit Time by Layer
                sns.lineplot(data=layer_df, x='Layer', y='Avg Time (s)', 
                            hue='Base_Method', marker='d', ax=axes[1,1])
                axes[1,1].set_title('Average Edit Time by Layer')
                axes[1,1].grid(True, alpha=0.3)
                
                # 6. All Success Rates Comparison
                success_df = layer_df.melt(
                    id_vars=['Layer', 'Base_Method'],
                    value_vars=['Edit Success Rate', 'Reversal Success Rate', 'Unrelated Success Rate'],
                    var_name='Success_Type', value_name='Success_Rate'
                )
                sns.lineplot(data=success_df, x='Layer', y='Success_Rate', 
                            hue='Base_Method', style='Success_Type', marker='o', ax=axes[1,2])
                axes[1,2].set_title('All Success Rates Comparison')
                axes[1,2].set_ylim(0, 1)
                axes[1,2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_filename = plots_dir / f'editing_results_by_layer_{model.replace("/", "_")}_{direction}.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {plot_filename}")
            plt.show()
            
            # Create heatmap for success rates - now includes holdout if available
            if len(layer_df['Base_Method'].unique()) > 1:
                num_heatmaps = 8 if has_holdout else 5
                fig, axes = plt.subplots(1, num_heatmaps, figsize=(6*num_heatmaps, 6))
                
                # Main editing heatmaps
                pivot_edit = layer_df.pivot(index='Layer', columns='Base_Method', values='Edit Success Rate')
                pivot_reversal = layer_df.pivot(index='Layer', columns='Base_Method', values='Reversal Success Rate')
                pivot_unrelated = layer_df.pivot(index='Layer', columns='Base_Method', values='Unrelated Success Rate')
                pivot_nbr_new2org = layer_df.pivot(index='Layer', columns='Base_Method', values='Nbr New2Org Success Rate')
                pivot_nbr_org2new = layer_df.pivot(index='Layer', columns='Base_Method', values='Nbr Org2New Success Rate')
                
                sns.heatmap(pivot_edit, annot=True, cmap='RdYlGn', center=0.5, 
                           ax=axes[0], cbar_kws={'label': 'Success Rate'})
                axes[0].set_title(f'Edit Success Rate - {model} ({direction})')
                
                sns.heatmap(pivot_reversal, annot=True, cmap='RdYlGn', center=0.5,
                           ax=axes[1], cbar_kws={'label': 'Success Rate'})
                axes[1].set_title(f'Reversal Success Rate - {model} ({direction})')
                
                sns.heatmap(pivot_unrelated, annot=True, cmap='RdYlGn', center=0.5,
                           ax=axes[2], cbar_kws={'label': 'Success Rate'})
                axes[2].set_title(f'Unrelated Success Rate - {model} ({direction})')
                
                sns.heatmap(pivot_nbr_new2org, annot=True, cmap='RdYlGn', center=0.5,
                           ax=axes[3], cbar_kws={'label': 'Success Rate'})
                axes[3].set_title(f'Nbr New2Org Success Rate - {model} ({direction})')
                
                sns.heatmap(pivot_nbr_org2new, annot=True, cmap='RdYlGn', center=0.5,
                           ax=axes[4], cbar_kws={'label': 'Success Rate'})
                axes[4].set_title(f'Nbr Org2New Success Rate - {model} ({direction})')
                
                # Add holdout heatmaps if available
                if has_holdout:
                    pivot_holdout_edit = layer_df.pivot(index='Layer', columns='Base_Method', values='Holdout Edit Success Rate')
                    pivot_holdout_reversal = layer_df.pivot(index='Layer', columns='Base_Method', values='Holdout Reversal Success Rate')
                    pivot_holdout_unrelated = layer_df.pivot(index='Layer', columns='Base_Method', values='Holdout Unrelated Success Rate')
                    
                    sns.heatmap(pivot_holdout_edit, annot=True, cmap='RdYlGn', center=0.5,
                               ax=axes[5], cbar_kws={'label': 'Success Rate'})
                    axes[5].set_title(f'Holdout Edit Success Rate - {model} ({direction})')
                    
                    sns.heatmap(pivot_holdout_reversal, annot=True, cmap='RdYlGn', center=0.5,
                               ax=axes[6], cbar_kws={'label': 'Success Rate'})
                    axes[6].set_title(f'Holdout Reversal Success Rate - {model} ({direction})')
                    
                    sns.heatmap(pivot_holdout_unrelated, annot=True, cmap='RdYlGn', center=0.5,
                               ax=axes[7], cbar_kws={'label': 'Success Rate'})
                    axes[7].set_title(f'Holdout Unrelated Success Rate - {model} ({direction})')
                
                plt.tight_layout()
                heatmap_filename = plots_dir / f'editing_results_heatmap_{model.replace("/", "_")}_{direction}.png'
                plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
                print(f"Saved heatmap to {heatmap_filename}")
                plt.show()

def analyze_by_layer(df):
    """Analyze results by layer"""
    print("\n=== Analysis by Model and Direction ===")
    
    # Check if holdout data is available
    has_holdout = 'Holdout Edit Success Rate' in df.columns and df['Holdout Edit Success Rate'].notna().any()
    
    for model in df['Model'].unique():
        for direction in df['Direction'].unique():
            model_dir_df = df[(df['Model'] == model) & (df['Direction'] == direction)]
            layer_df = model_dir_df[model_dir_df['Layer'].notna()]
            
            if len(layer_df) == 0:
                continue
            
            print(f"\n--- {model} ({direction}) ---")
            
            # Report pre-editing performance first
            print(f"\n=== PRE-EDITING BASELINE PERFORMANCE ===")
            pre_summary = layer_df.agg({
                'Pre Edit Success Rate': 'mean',
                'Pre Reversal Success Rate': 'mean',
                'Pre Nbr New2Org Success Rate': 'mean',
                'Pre Nbr Org2New Success Rate': 'mean',
                'Pre Unrelated Success Rate': 'mean'
            }).round(3)
            
            print(f"Baseline Edit Success Rate: {pre_summary['Pre Edit Success Rate']:.3f}")
            print(f"Baseline Reversal Success Rate: {pre_summary['Pre Reversal Success Rate']:.3f}")
            print(f"Baseline Nbr New2Org Success Rate: {pre_summary['Pre Nbr New2Org Success Rate']:.3f}")
            print(f"Baseline Nbr Org2New Success Rate: {pre_summary['Pre Nbr Org2New Success Rate']:.3f}")
            print(f"Baseline Unrelated Success Rate: {pre_summary['Pre Unrelated Success Rate']:.3f}")
            
            # Best performing layers for each method
            for method in layer_df['Base_Method'].unique():
                method_df = layer_df[layer_df['Base_Method'] == method]
                
                best_edit_layer = method_df.loc[method_df['Edit Success Rate'].idxmax()]
                best_reversal_layer = method_df.loc[method_df['Reversal Success Rate'].idxmax()]
                best_nbr_new2org_layer = method_df.loc[method_df['Nbr New2Org Success Rate'].idxmax()]
                best_nbr_org2new_layer = method_df.loc[method_df['Nbr Org2New Success Rate'].idxmax()]
                best_unrelated_layer = method_df.loc[method_df['Unrelated Success Rate'].idxmax()]
                
                print(f"\n{method}:")
                print(f"  Best Edit Layer: {best_edit_layer['Layer']} (Success: {best_edit_layer['Edit Success Rate']:.3f})")
                print(f"  Best Reversal Layer: {best_reversal_layer['Layer']} (Success: {best_reversal_layer['Reversal Success Rate']:.3f})")
                print(f"  Best Nbr New2Org Layer: {best_nbr_new2org_layer['Layer']} (Success: {best_nbr_new2org_layer['Nbr New2Org Success Rate']:.3f})")
                print(f"  Best Nbr Org2New Layer: {best_nbr_org2new_layer['Layer']} (Success: {best_nbr_org2new_layer['Nbr Org2New Success Rate']:.3f})")
                print(f"  Best Unrelated Layer: {best_unrelated_layer['Layer']} (Success: {best_unrelated_layer['Unrelated Success Rate']:.3f})")
                
                # Overall performance summary (post-editing)
                print(f"  Avg Edit Success: {method_df['Edit Success Rate'].mean():.3f} (vs baseline: {method_df['Pre Edit Success Rate'].mean():.3f})")
                print(f"  Avg Reversal Success: {method_df['Reversal Success Rate'].mean():.3f} (vs baseline: {method_df['Pre Reversal Success Rate'].mean():.3f})")
                print(f"  Avg Nbr New2Org Success: {method_df['Nbr New2Org Success Rate'].mean():.3f} (vs baseline: {method_df['Pre Nbr New2Org Success Rate'].mean():.3f})")
                print(f"  Avg Nbr Org2New Success: {method_df['Nbr Org2New Success Rate'].mean():.3f} (vs baseline: {method_df['Pre Nbr Org2New Success Rate'].mean():.3f})")
                print(f"  Avg Unrelated Success: {method_df['Unrelated Success Rate'].mean():.3f} (vs baseline: {method_df['Pre Unrelated Success Rate'].mean():.3f})")
                print(f"  Avg Edit Time: {method_df['Avg Time (s)'].mean():.2f}s")
                
                # Add holdout analysis if available
                if has_holdout and method_df['Holdout Edit Success Rate'].notna().any():
                    print(f"  === HOLDOUT PERFORMANCE ===")
                    print(f"  Avg Holdout Edit Success: {method_df['Holdout Edit Success Rate'].mean():.3f}")
                    print(f"  Avg Holdout Reversal Success: {method_df['Holdout Reversal Success Rate'].mean():.3f}")
                    print(f"  Avg Holdout Nbr New2Org Success: {method_df['Holdout Nbr New2Org Success Rate'].mean():.3f}")
                    print(f"  Avg Holdout Nbr Org2New Success: {method_df['Holdout Nbr Org2New Success Rate'].mean():.3f}")
                    print(f"  Avg Holdout Unrelated Success: {method_df['Holdout Unrelated Success Rate'].mean():.3f}")
            
            # Layer performance across methods
            print(f"\n=== Layer Performance for {model} ({direction}) ===")
            layer_cols = ['Edit Success Rate', 'Reversal Success Rate', 'Nbr New2Org Success Rate', 
                         'Nbr Org2New Success Rate', 'Unrelated Success Rate', 'Avg Time (s)']
            
            if has_holdout:
                layer_cols.extend(['Holdout Edit Success Rate', 'Holdout Reversal Success Rate', 
                                 'Holdout Nbr New2Org Success Rate', 'Holdout Nbr Org2New Success Rate', 
                                 'Holdout Unrelated Success Rate'])
            
            layer_summary = layer_df.groupby('Layer').agg({col: 'mean' for col in layer_cols}).round(3)
            print(layer_summary.to_string())
    
    # Compare UNI vs BI performance
    if len(df['Direction'].unique()) > 1:
        print(f"\n=== UNI vs BI Direction Comparison ===")
        
        # Pre-editing baseline comparison
        print(f"\nPre-editing baseline performance:")
        pre_direction_summary = df.groupby(['Direction']).agg({
            'Pre Edit Success Rate': 'mean',
            'Pre Reversal Success Rate': 'mean',
            'Pre Nbr New2Org Success Rate': 'mean',
            'Pre Nbr Org2New Success Rate': 'mean',
            'Pre Unrelated Success Rate': 'mean'
        }).round(3)
        print(pre_direction_summary.to_string())
        
        # Post-editing performance comparison
        print(f"\nPost-editing performance:")
        agg_cols = {
            'Edit Success Rate': 'mean',
            'Reversal Success Rate': 'mean',
            'Nbr New2Org Success Rate': 'mean',
            'Nbr Org2New Success Rate': 'mean',
            'Unrelated Success Rate': 'mean',
            'Avg Time (s)': 'mean'
        }
        
        if has_holdout:
            agg_cols.update({
                'Holdout Edit Success Rate': 'mean',
                'Holdout Reversal Success Rate': 'mean',
                'Holdout Nbr New2Org Success Rate': 'mean',
                'Holdout Nbr Org2New Success Rate': 'mean',
                'Holdout Unrelated Success Rate': 'mean'
            })
        
        direction_summary = df.groupby(['Direction', 'Base_Method']).agg(agg_cols).round(3)
        print(direction_summary.to_string())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Summarize editing results')
    parser.add_argument('--results_dir', type=str, default="../results/editing",
                        help='Directory containing results to summarize')
    args = parser.parse_args()
    summarize_results(args.results_dir)