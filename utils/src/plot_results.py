import os
import matplotlib.pyplot as plt
import pandas as pd

def parse_times(times_file):
    """
    Parse the time file from the solver.
    It saves a plot for each graph type where the average time is 
    plotted with the an increasing number of cores.
    """

    cur_dir = os.getcwd()

    work_dir = cur_dir.split("src")[0]
    file_path = os.path.join(work_dir, "results", times_file)

    print("Path", file_path)

    df = pd.read_csv(file_path, sep=" ", header=None)
    df.columns = ["graph", "times", "cores"]

    with open("./results/statistics.csv", 'a') as file:
        file.write("graph_size cores mean std speed_up strong_efficiency\n")
        for graph in df.graph.unique():

            # Dataframe
            res_df = pd.DataFrame(
                columns=[
                         "cores", 
                         "times", 
                         "std", 
                         "speedup", 
                         "strong_efficiency"
                        ]
                )

            seq_avg = df.times[(df.graph == graph) & (df.cores == 1)].mean()
            seq_std = df.times[(df.graph == graph) & (df.cores == 1)].std()

            res_df.loc[0] = [1, seq_avg, seq_std, 1, 1]

            for i, core in enumerate(sorted(df.cores.unique())):

                mean = df[(df.graph == graph) & (df.cores == core)].mean().times
                std = df[(df.graph == graph) & (df.cores == core)].std().times

                speedup = seq_avg / mean
                str_eff = seq_avg / (core * mean)

                res_df.loc[i] = [int(core), mean, std, speedup, str_eff] 

                file.write(
                    f"{int(graph)} {int(core)} {mean} {speedup} {str_eff}\n"
                )

            # save the plots
            time_plot   = res_df.plot(
                                x="cores", y="times", 
                                label=f"{graph} vertices"
                        ).get_figure()
            speedup_plt = res_df.plot(
                                x="cores", y="speedup", 
                                label=f"{graph} vertices"
                        ).get_figure()
            eff_plot    = res_df.plot(
                            x = "cores", y="strong_efficiency", 
                            label=f"{graph} vertices"
                        ).get_figure()
            # Save result
            time_plot.savefig(f"./results/{graph}_time.png")
            speedup_plt.savefig(f"./results/{graph}_speed_up.png")
            eff_plot.savefig(f"./results/{graph}_str_eff.png")
            
    print("-"*50)
    print("All saved")
    print("-"*50)

if __name__ == "__main__":
    parse_times("times.txt")

