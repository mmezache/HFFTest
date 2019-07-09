import matplotlib.pyplot as plt


def getFig(nsubplot, name):
    fig, ax =  plt.subplots(nsubplot)
    fig.tight_layout()
    fig.canvas.set_window_title(name)
    
    return fig, ax
