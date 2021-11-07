def eplot_plot(exp):
        ax = sns.lmplot(x="steps",y="average_entropy",data=exp.scores_df,fit_reg=False)
        ax.set_xticklabels(rotation=30)
        ax.set(yscale="log")
        ax2 = plt.gca()
        ax2.set_title(exp.command_string())
        plt.ylim(min(exp.scores_df['average_entropy'])/3, max(exp.scores_df['average_entropy'])*3)