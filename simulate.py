import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import entropy

    return entropy, np, plt


@app.cell
def _(np):
    # --- Environment & Styling Configuration ---
    GRID_SIZE = 15
    SOURCE_POS = (13, 13)
    START_POS = (2, 2)

    # Trajectory visual configuration
    gap = 2
    length = 8
    tail_dist = gap + length
    arrow_style = dict(facecolor='white', edgecolor='black', width=2, headwidth=6, shrink=0.05)

    def calculate_hit_rate_simple(x, y, source_x, source_y):
        dist = np.sqrt((x - source_x)**2 + (y - source_y)**2)
        return np.exp(-dist / 4.0)

    def get_neighbors_simple(pos):
        x, y = pos
        moves = [(x, y), (x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        return [(nx, ny) for nx, ny in moves if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE]


    return GRID_SIZE, SOURCE_POS, START_POS, arrow_style


@app.cell
def _(GRID_SIZE, np):
    def calculate_hit_rate(x, y, source_x, source_y):
        """Generates a multimodal landscape with local optima and valleys."""

        # 1. Main Source Plume (Global Optimum)
        dist_to_source = np.sqrt((x - source_x)**2 + (y - source_y)**2)
        global_plume = np.exp(-dist_to_source / 4.0)

        # 2. Distractor Pockets (Local Optima)
        # Format: ((x, y), amplitude, spread)
        distractors = [
            ((4, 10), 0.7, 1.2),  
            ((10, 4), 0.8, 1.0),
            ((8, 8), 0.5, 1.5)
        ]
        local_peaks = 0
        for (dx, dy), amp, spread in distractors:
            dist = np.sqrt((x - dx)**2 + (y - dy)**2)
            local_peaks += amp * np.exp(-dist / spread)

        # 3. Terrain Ripples (Narrow Valleys)
        # Increases the frequency of the bumps across the grid
        ripples = 0.1 * np.sin(x * 1.8) * np.cos(y * 1.8)

        # Combine everything. 
        # We use max(0.001, ...) to ensure the hit rate never hits absolute zero 
        # or goes negative, which would break the Bayesian log-likelihood math in Infotaxis.
        total_rate = global_plume + local_peaks + ripples
        return max(0.001, total_rate)

    def get_neighbors(pos):
        x, y = pos
        moves = [(x, y), (x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        return [(nx, ny) for nx, ny in moves if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE]

    return calculate_hit_rate, get_neighbors


@app.cell
def _(GRID_SIZE, SOURCE_POS, START_POS, calculate_hit_rate, get_neighbors, np):
    # --- Search Strategies ---
    def simulate_grid_search():
        trajectory = []
        for y in range(GRID_SIZE):
            x_range = range(GRID_SIZE) if y % 2 == 0 else range(GRID_SIZE-1, -1, -1)
            for x in x_range:
                trajectory.append((x, y))
                if (x, y) == SOURCE_POS:
                    return trajectory
        return trajectory

    def simulate_greedy_search():
        pos = START_POS
        trajectory = [pos]

        for _ in range(200):
            if pos == SOURCE_POS:
                break

            neighbors = get_neighbors(pos)
            rates = [calculate_hit_rate(nx, ny, SOURCE_POS[0], SOURCE_POS[1]) for nx, ny in neighbors]

            best_move = neighbors[np.argmax(rates)]

            if best_move == pos and len(trajectory) > 1:
                 best_move = neighbors[np.random.randint(len(neighbors))]

            pos = best_move
            trajectory.append(pos)

        return trajectory

    # def simulate_infotaxis():
    #     pos = START_POS
    #     trajectory = [pos]

    #     belief = np.ones((GRID_SIZE, GRID_SIZE))
    #     belief /= np.sum(belief)

    #     for _ in range(150):
    #         if pos == SOURCE_POS:
    #             break

    #         actual_rate = calculate_hit_rate(pos[0], pos[1], SOURCE_POS[0], SOURCE_POS[1])
    #         hit = np.random.poisson(actual_rate) > 0

    #         for i in range(GRID_SIZE):
    #             for j in range(GRID_SIZE):
    #                 expected_rate = calculate_hit_rate(pos[0], pos[1], i, j)
    #                 likelihood = expected_rate if hit else (1 - expected_rate)
    #                 belief[i, j] *= likelihood

    #         belief /= np.sum(belief)

    #         neighbors = get_neighbors(pos)
    #         best_move = pos
    #         min_entropy = float('inf')

    #         for nx, ny in neighbors:
    #             expected_H = entropy(belief.flatten()) - belief[nx, ny] * 0.2 
    #             if expected_H < min_entropy:
    #                 min_entropy = expected_H
    #                 best_move = (nx, ny)

    #         pos = best_move
    #         trajectory.append(pos)

    #     return trajectory

    return simulate_greedy_search, simulate_grid_search


@app.cell
def _(
    GRID_SIZE,
    SOURCE_POS,
    START_POS,
    calculate_hit_rate,
    entropy,
    get_neighbors,
    np,
):
    def simulate_infotaxis():
        pos = START_POS
        trajectory = [pos]
    
        # Initialize uniform belief prior
        belief = np.ones((GRID_SIZE, GRID_SIZE))
        belief /= np.sum(belief)
    
        # Precompute the forward model: hit_rates[x, y, source_x, source_y]
        # This speeds up the Bayesian look-ahead massively.
        forward_model = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE, GRID_SIZE))
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                for sx in range(GRID_SIZE):
                    for sy in range(GRID_SIZE):
                        forward_model[x, y, sx, sy] = calculate_hit_rate(x, y, sx, sy)
                    
        for _ in range(150):
            if pos == SOURCE_POS: break
        
            # 1. Observe and Update Belief Map
            actual_rate = calculate_hit_rate(pos[0], pos[1], SOURCE_POS[0], SOURCE_POS[1])
            hit = np.random.poisson(actual_rate) > 0
        
            # Pull the expected rates for our current position against all hypothetical sources
            likelihood = forward_model[pos[0], pos[1], :, :]
            if not hit:
                likelihood = 1 - likelihood
            
            belief *= likelihood
            sum_belief = np.sum(belief)
        
            if sum_belief > 0:
                belief /= sum_belief
            else:
                # Reset prior if the math underflows (agent gets hopelessly lost)
                belief = np.ones((GRID_SIZE, GRID_SIZE)) / (GRID_SIZE**2) 
            
            # 2. Choose Next Move via Rigorous Expected Entropy
            neighbors = get_neighbors(pos)
            best_move = pos
            min_expected_entropy = float('inf')
        
            for nx, ny in neighbors:
                # Probability the source is exactly at this neighbor
                p_source = belief[nx, ny]
            
                # Expected hit rates at this neighbor for all hypothetical source locations
                R = forward_model[nx, ny, :, :]
            
                # Marginal probability of observing a hit/miss at this neighbor
                p_hit = np.sum(belief * R)
                p_miss = 1 - p_hit
            
                # Simulate the posterior belief map IF we get a hit
                belief_hit = belief * R
                sum_bh = np.sum(belief_hit)
                H_hit = entropy((belief_hit / sum_bh).flatten()) if sum_bh > 0 else 0
            
                # Simulate the posterior belief map IF we get a miss
                belief_miss = belief * (1 - R)
                sum_bm = np.sum(belief_miss)
                H_miss = entropy((belief_miss / sum_bm).flatten()) if sum_bm > 0 else 0
            
                # Calculate the rigorous expected entropy
                expected_H = (1 - p_source) * (p_hit * H_hit + p_miss * H_miss)
            
                if expected_H < min_expected_entropy:
                    min_expected_entropy = expected_H
                    best_move = (nx, ny)
                
            pos = best_move
            trajectory.append(pos)
        
        return trajectory

    return (simulate_infotaxis,)


@app.cell
def _(
    GRID_SIZE,
    SOURCE_POS,
    START_POS,
    arrow_style,
    calculate_hit_rate,
    np,
    plt,
    simulate_greedy_search,
    simulate_grid_search,
    simulate_infotaxis,
):
    # Generate the landscape "bumpiness" for the heatmap
    landscape = np.zeros((GRID_SIZE, GRID_SIZE))
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            landscape[y, x] = calculate_hit_rate(x, y, SOURCE_POS[0], SOURCE_POS[1])

    # --- Visualization ---
    def plot_trajectory(ax, trajectory, title):
        ax.set_xlim(-0.5, GRID_SIZE - 0.5)
        ax.set_ylim(-0.5, GRID_SIZE - 0.5)
        ax.set_title(title, fontsize=16)

        # Display the heatmap background
        im = ax.imshow(
            landscape, 
            origin='lower', 
            cmap='viridis', 
            alpha=0.7, 
            extent=[-0.5, GRID_SIZE-0.5, -0.5, GRID_SIZE-0.5]
        )

        ax.plot(SOURCE_POS[0], SOURCE_POS[1], marker='*', color='gold', markersize=20, markeredgecolor='red', label='Target')

        for i in range(len(trajectory) - 1):
            p1 = trajectory[i]
            p2 = trajectory[i+1]
            if p1 != p2:
                ax.annotate("", xy=p2, xytext=p1, arrowprops=arrow_style)

        ax.plot(START_POS[0], START_POS[1], 'go', markersize=10, label='Start')

        # Only add legend to the first plot to keep it clean
        if title.startswith("Grid"):
            ax.legend(loc='upper left')

        ax.tick_params(labelbottom=False, labelleft=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    traj_grid = simulate_grid_search()
    traj_greedy = simulate_greedy_search()
    traj_info = simulate_infotaxis()

    plot_trajectory(axes[0], traj_grid, "Grid Search")
    plot_trajectory(axes[1], traj_greedy, "Greedy Search")
    plot_trajectory(axes[2], traj_info, "Infotaxis")

    # Add a colorbar to the figure to show concentration intensity
    fig.colorbar(axes[0].images[0], ax=axes.ravel().tolist(), label="Hit Concentration", shrink=0.7)

    plt.show()
    fig.savefig("./plots/simulated_trajectories_complex.png")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
