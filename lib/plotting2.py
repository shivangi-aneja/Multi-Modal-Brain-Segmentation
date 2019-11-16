import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("darkgrid")


def create_line_chart(ft, xc, classes, filename):
    fig, ax = plt.subplots()
    ax.plot(classes, xc, marker="o", linewidth=4, markersize=12, label="XceptionNet", markerfacecolor='firebrick', color='salmon')
    ax.plot(classes, ft, marker="v", markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4, label="Ours")
    ax.legend()
    fig.tight_layout()
    plt.setp(ax.get_xticklabels(), rotation=0, horizontalalignment='right')
    plt.savefig(filename)


classes = ['0', '1', '2', '3', '4', '5', '10', '15', '25', '50', '100']
f2f_ft = [99.0, 88.075, 86.275, 83.52, 91.17, 90.85, 92.047, 91.805, 93.38, 93.325, 93.86]
f2f_xc = [99.525, 92.7, 89.29, 89.375, 89.46, 92.795, 90.64, 91.755, 91.0375, 91.3175, 91.45]
f2f_data = {'class' : classes, 'ft': f2f_ft, 'xc': f2f_xc}


fs_ft = [50.725, 66.68, 74.85, 77.75, 77.1, 73.67, 75.1, 80.60, 81.125, 86.67, 90.0]
fs_xc = [50.275, 61.065, 61.53, 65.0125, 65.6, 65.477, 67.61, 66.79, 67.3125, 67.125, 67.2]
fs_data = {'ft': fs_ft, 'xc': fs_xc}

create_line_chart(f2f_ft, f2f_xc, classes, 'f2f.png')
create_line_chart(fs_ft, fs_xc, classes, 'fs.png')
