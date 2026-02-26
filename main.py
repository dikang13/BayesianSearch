import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    import warnings
    warnings.filterwarnings('ignore')

    # 1. Space Setup
    np.random.seed(42)
    n_cands = 32
    n_conds = 8
    n_pairs = n_cands * n_conds

    # 2D search space: Candidates x Conditions
    X_cands = np.linspace(0, 1, n_cands)
    X_conds = np.linspace(0, 1, n_conds)
    X_grid = np.array(np.meshgrid(X_cands, X_conds)).T.reshape(-1, 2)

    # Ground truth generator
    kernel = ConstantKernel(1.0) * RBF(length_scale=[0.15, 0.15])
    gp_true = GaussianProcessRegressor(kernel=kernel, optimizer=None)

    # 2. Simulation Setup
    n_trials = 8
    budget = 80 # Maximum number of experiments to run out of 256

    metrics = {
        'Freq': {'cum_regret': np.zeros(budget), 'hit_budget': [], 'std_opt': np.zeros(budget)},
        'BO_Non_NoUp': {'cum_regret': np.zeros(budget), 'hit_budget': [], 'std_opt': np.zeros(budget)},
        'BO_Inf_NoUp': {'cum_regret': np.zeros(budget), 'hit_budget': [], 'std_opt': np.zeros(budget)},
        'BO_Non_Seq': {'cum_regret': np.zeros(budget), 'hit_budget': [], 'std_opt': np.zeros(budget)},
        'BO_Inf_Seq': {'cum_regret': np.zeros(budget), 'hit_budget': [], 'std_opt': np.zeros(budget)}
    }

    labels = {
        'Freq': '1) Frequentist',
        'BO_Non_NoUp': '2) BO (Non-info, No Update)',
        'BO_Inf_NoUp': '3) BO (Info Prior, No Update)',
        'BO_Non_Seq': '4) BO (Non-info, Sequential)',
        'BO_Inf_Seq': '5) BO (Info Prior, Sequential)'
    }

    colors = {
        'Freq': '#7f8c8d', 
        'BO_Non_NoUp': '#3498db', 
        'BO_Inf_NoUp': '#f39c12', 
        'BO_Non_Seq': '#9b59b6', 
        'BO_Inf_Seq': '#e74c3c'
    }

    linestyles = {
        'Freq': '-', 
        'BO_Non_NoUp': '--', 
        'BO_Inf_NoUp': '-', 
        'BO_Non_Seq': '-.', 
        'BO_Inf_Seq': '-'
    }

    print(f"Running {n_trials} trials...")

    for trial in range(n_trials):
        # Latent true landscape
        Y_true = gp_true.sample_y(X_grid, random_state=trial).ravel()
        opt_idx = np.argmax(Y_true)
        opt_val = Y_true[opt_idx]
    
        # QSAR Prior (Ground truth + noise)
        qsar_prior = Y_true + np.random.normal(0, 0.5, n_pairs)
    
        # Approaches 1, 2, 3: Static Rankings
        order_freq = np.random.permutation(n_pairs)
        order_non_noup = np.random.permutation(n_pairs) 
        order_inf_noup = np.argsort(qsar_prior)[::-1]
    
        for name, order in [('Freq', order_freq), ('BO_Non_NoUp', order_non_noup), ('BO_Inf_NoUp', order_inf_noup)]:
            cum_regret = 0
            hit_b = budget 
            for b in range(budget):
                idx = order[b]
                cum_regret += (opt_val - Y_true[idx])
                metrics[name]['cum_regret'][b] += cum_regret
                metrics[name]['std_opt'][b] += 1.0 # No learning means uncertainty stays at 1.0
            
                if idx == opt_idx and hit_b == budget:
                    hit_b = b + 1
            metrics[name]['hit_budget'].append(hit_b)

        # Approach 4: BO Sequential Update (Non-informative Prior)
        gp_non_seq = GaussianProcessRegressor(kernel=kernel, alpha=1e-3, normalize_y=True, optimizer=None)
        cum_regret = 0
        hit_b = budget
        sampled_non_seq = []
    
        for b in range(budget):
            if b == 0:
                idx = np.random.choice(n_pairs) # Start blind
                std_opt = 1.0
            else:
                X_train = X_grid[sampled_non_seq]
                y_train = Y_true[sampled_non_seq]
                gp_non_seq.fit(X_train, y_train)
            
                mu, sigma = gp_non_seq.predict(X_grid, return_std=True)
                ucb = mu + 1.96 * sigma
                ucb[sampled_non_seq] = -np.inf
                idx = np.argmax(ucb)
            
                _, std_opt_arr = gp_non_seq.predict(X_grid[opt_idx:opt_idx+1], return_std=True)
                std_opt = std_opt_arr[0]
            
            sampled_non_seq.append(idx)
            cum_regret += (opt_val - Y_true[idx])
            metrics['BO_Non_Seq']['cum_regret'][b] += cum_regret
            metrics['BO_Non_Seq']['std_opt'][b] += std_opt
        
            if idx == opt_idx and hit_b == budget:
                hit_b = b + 1
            
        metrics['BO_Non_Seq']['hit_budget'].append(hit_b)

        # Approach 5: BO Sequential Update (Informative Prior)
        gp_inf_seq = GaussianProcessRegressor(kernel=kernel, alpha=1e-3, normalize_y=False, optimizer=None)
        cum_regret = 0
        hit_b = budget
        sampled_inf_seq = []
    
        for b in range(budget):
            if b == 0:
                idx = np.argmax(qsar_prior) # Start with best guess
                std_opt = 1.0
            else:
                X_train = X_grid[sampled_inf_seq]
                y_train = Y_true[sampled_inf_seq] - qsar_prior[sampled_inf_seq]
                gp_inf_seq.fit(X_train, y_train)
            
                mu, sigma = gp_inf_seq.predict(X_grid, return_std=True)
                mu += qsar_prior
                ucb = mu + 1.96 * sigma
                ucb[sampled_inf_seq] = -np.inf
                idx = np.argmax(ucb)
            
                _, std_opt_arr = gp_inf_seq.predict(X_grid[opt_idx:opt_idx+1], return_std=True)
                std_opt = std_opt_arr[0]
            
            sampled_inf_seq.append(idx)
            cum_regret += (opt_val - Y_true[idx])
            metrics['BO_Inf_Seq']['cum_regret'][b] += cum_regret
            metrics['BO_Inf_Seq']['std_opt'][b] += std_opt
        
            if idx == opt_idx and hit_b == budget:
                hit_b = b + 1
            
        metrics['BO_Inf_Seq']['hit_budget'].append(hit_b)

    # Average out metrics across all trials
    for name in metrics:
        metrics[name]['cum_regret'] /= n_trials
        metrics[name]['std_opt'] /= n_trials

    print("Simulations complete. Generating plots...")

    # 3. Visualization Code
    budgets_arr = np.arange(1, budget + 1)
    plt.rcParams.update({'font.size': 11})

    # Plot A: Cumulative Regret
    plt.figure(figsize=(10, 6))
    for name in metrics:
        plt.plot(budgets_arr, metrics[name]['cum_regret'], label=labels[name], 
                 color=colors[name], linestyle=linestyles[name], linewidth=2.5)
    plt.title('A) Cumulative Regret vs. Number of Experiments')
    plt.xlabel('Number of Experiments (Candidate-Condition Pairs)')
    plt.ylabel('Cumulative Regret')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('cumulative_regret.png')
    plt.close()

    # Plot B: Budget to First Hit
    plt.figure(figsize=(10, 6))
    means = [np.mean(metrics[name]['hit_budget']) for name in metrics]
    stds = [np.std(metrics[name]['hit_budget']) for name in metrics]
    bars = plt.bar(list(labels.values()), means, yerr=stds, capsize=6, 
                   color=list(colors.values()), alpha=0.85, edgecolor='black')
    plt.title('B) Budget Required to Find True Optimal Pair')
    plt.ylabel('Number of Experiments (Lower is Better)')
    plt.xticks(rotation=20, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}', 
                 ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('budget_to_hit.png')
    plt.close()

    # Plot C: Narrowing of Confidence
    plt.figure(figsize=(10, 6))
    for name in metrics:
        plt.plot(budgets_arr, metrics[name]['std_opt'], label=labels[name], 
                 color=colors[name], linestyle=linestyles[name], linewidth=2.5)
    plt.title('C) Uncertainty ($\sigma$) at True Optimal Pair over Time')
    plt.xlabel('Number of Experiments')
    plt.ylabel('Posterior Standard Deviation (Uncertainty)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('confidence_narrowing.png')
    plt.close()

    print("Plots saved as PNG files in the current directory.")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
    