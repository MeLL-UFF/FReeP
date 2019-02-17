import pandas as pd
import matplotlib
matplotlib.use('agg')
import seaborn as sns

class GraphGenerator():
    
    def bar_graph(self, data, filename, x, y, hue, y_label):  
        graph = sns.catplot(x=x, y=y, hue=hue, data=data,
                        height=6, kind="bar", palette="muted")
        graph.despine(left=True)
        graph.set_ylabels(y_label)                                                                                                          
        graph.fig.savefig(filename)