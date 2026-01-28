# -*- coding: utf-8 -*-

__author__ = 'Rodrigo Lopez Aburto'
__email__  = 'lopezaburtorodrigo@gmail.com' 
__date__   = '2024-05-07'
__version__ = '1.0'
__copyright__ = 'Copyright (C) 2023 Rodrigo Lopez Aburto'
__license__ = 'GNU GPL Version 3.0'

__last_editor__ = 'Sinai Morales-Chávez'
__last_date_edition__ = '2026-01-28'  
__last_version__ = '1.2'

"""
Esta biblioteca contiene funciones para el análisis estadístico y visualización de registros geofísicos de pozo. 
Las funciones generan algunas de las gráficas y tablas empleadas durante el proceso de análisis de datos de registros 
geofísicos de pozo. A manera de tutorial, el archivo "Tutorial_GraphStat.ipynb" contiene una descripción del uso de cada 
una de las funciones, así como un ejemplo de su uso aplicado a datos reales de registros geofísicos de pozo.
""" 

## Bibliotecas utilizadas

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib import __version__ as mpl_version
from sklearn import __version__ as sklearn_version
from sklearn import metrics
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from tabulate import tabulate
from tabulate import __version__ as tab_version
import missingno as msno
import math

#####################################################################################################################################
def print_list_of_functions(function_name):
    functions = function_name.__dict__
    functions_info = [(name, value) for name, value in functions.items() if callable(value)]
    functions_info.sort(key=lambda x: x[0])
    
    quarter_len = len(functions_info) // 4
    functions_info_col1 = functions_info[:quarter_len]
    functions_info_col2 = functions_info[quarter_len:2*quarter_len]
    functions_info_col3 = functions_info[2*quarter_len:3*quarter_len]
    functions_info_col4 = functions_info[3*quarter_len:]
    
    functions_info_list = [[col1[0], col2[0], col3[0], col4[0]] for col1, col2, col3, col4 in zip(functions_info_col1, functions_info_col2, functions_info_col3, functions_info_col4)]
    
    headers = ['Table of functions of the', f'{function_name.__name__} library', f'Total: {len(functions_info)} functions', '', '']
    print(tabulate(functions_info_list, headers=headers))

####################################################################################################################################

def Statistics(Data, list_names, language='esp', decimals=4, scientific_notation=False, 
               save_csv=False, csv_filename='statistics', 
               save_png=False, png_filename='statistics', 
               log_transform=False):
    """
    Generates a comprehensive table of descriptive statistics for the specified variables.

    This function acts as a unified interface for Exploratory Data Analysis (EDA), 
    accepting both Pandas DataFrames and dictionary-like structures containing 
    NumPy arrays. It calculates measures of 
    central tendency, dispersion, and distribution shape (skewness/kurtosis).

    Parameters
    ----------
    Data : pd.DataFrame, dict, or openpnm.network.Network
        The input data source. It can be:
        - A pandas DataFrame where columns are variables.
        - A dictionary or array-like structures.
    list_names : list of str
        A list of column names (if Data is DataFrame) or keys (if Data is dict) 
        to be analyzed. Keys not found in `Data` will be skipped with a warning.
    language : {'esp', 'eng'}, optional
        The language for the output table headers. 
        'esp' for Spanish (default) or 'eng' for English.
    decimals : int, optional
        The number of decimal places to round the results to. Default is 4.
    scientific_notation : bool, optional
        If True, formats the output values in scientific notation (e.g., 1.23e-05). 
        Default is False.
    save_csv : bool, optional
        If True, saves the resulting statistics table as a .csv file. Default is False.
    csv_filename : str, optional
        The base name for the CSV file (without extension). Default is 'statistics'.
    save_png : bool, optional
        If True, renders and saves the statistics table as a .png image. Default is False.
    png_filename : str, optional
        The base name for the PNG file (without extension). Default is 'statistics'.
    log_transform : bool, optional
        If True, applies a natural logarithm transformation (np.log) to the data 
        before calculating statistics. Non-positive values (<= 0) are filtered out 
        automatically to avoid domain errors. Default is False.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the calculated statistics. Rows represent the metrics 
        (Mean, Median, Std Dev, etc.) and columns represent the variables from `list_names`.

    Raises
    ------
    ValueError
        If `language` is not 'esp' or 'eng'.

    Notes
    -----
    The function internally converts NumPy arrays to Pandas Series to utilize 
    robust statistical methods (like .skew() and .kurtosis()) and handle missing 
    values (.isna()) consistently across data types.
    """
    
    if language == 'esp':
        Headers = ['Total Muestras', 'Valores nulos', 'Valores no nulos', 'Minimo', '1er Cuartil', 'Mediana', 'Media',
                   '3er Cuartil', 'Maximo', 'Rango', 'Rango Intercuartil',
                   'Varianza', 'Desviacion Estandar', 'Simetria', 'Curtosis']
        integer_rows = ['Total Muestras', 'Valores nulos', 'Valores no nulos']
    elif language == 'eng':
        Headers = ['Total Samples', 'Null Samples', 'No Null Samples', 'Minimum', '1st Quartile', 'Median', 'Mean',
                   '3rd Quartile', 'Maximum', 'Range', 'Interquartile Range',
                   'Variance', 'Standard Deviation', 'Skewness', 'Kurtosis']
        integer_rows = ['Total Samples', 'Null Samples', 'No Null Samples']
    else:
        raise ValueError("The 'language' parameter must be 'esp' or 'eng'.")

    Result = pd.DataFrame(data=np.zeros((len(Headers), len(list_names))), 
                          columns=list_names, index=Headers)

    for name in list_names:
        try:
            raw_data = Data[name]
        except KeyError:
            print(f"Warning: The variable '{name}' was not found in the provided Data.")
            continue
            
        data_column = pd.Series(raw_data).copy()
        
        if data_column.ndim > 1:
            data_column = pd.Series(data_column.values.flatten())

        if log_transform:
            data_column = data_column[data_column > 0]
            if len(data_column) == 0:
                print(f"Warning: Variable '{name}' has no positive values for log transform.")
                continue
            data_column = np.log(data_column)
        
        Null_samples = int(data_column.isna().sum())
        Total_samples = int(data_column.shape[0])
        True_values = Total_samples - Null_samples
        
        data_column.dropna(inplace=True)

        if len(data_column) == 0:
            Result[name] = 0.0
            Result.loc[Headers[0], name] = Total_samples
            Result.loc[Headers[1], name] = Null_samples
            continue

        Minimum = data_column.min()
        Quartiles = data_column.quantile([0.25, 0.5, 0.75])
        Mean = data_column.mean()
        Maximum = data_column.max()
        Range = Maximum - Minimum
        IQR = Quartiles[0.75] - Quartiles[0.25]
        Variance = data_column.var()
        Standard_dev = data_column.std()
        Skewness = data_column.skew()
        Kurtosis = data_column.kurtosis()

        Result[name] = [Total_samples, Null_samples, True_values, Minimum, Quartiles[0.25], Quartiles[0.5], 
                        Mean, Quartiles[0.75], Maximum, Range, IQR, Variance, 
                        Standard_dev, Skewness, Kurtosis]
        
    if scientific_notation:
        formatter_sci = lambda x: f'{x:.{decimals}e}' if pd.notnull(x) and isinstance(x, (int, float)) else x
        
        try:
            Result_Str = Result.map(formatter_sci)
        except AttributeError:
            Result_Str = Result.applymap(formatter_sci)
        
        for row in integer_rows:
            if row in Result.index:
                Result_Str.loc[row] = Result.loc[row].apply(lambda x: f"{int(x)}" if pd.notnull(x) else x)
        
        Result = Result_Str
            
    else:
        Result = Result.round(decimals)

        
    if save_csv:
        Result.to_csv(f"{csv_filename}.csv")

    if save_png:
        try:
            fig, ax = plt.subplots(figsize=(11, len(Result) * 0.5))
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=Result.values, colLabels=Result.columns, 
                             rowLabels=Result.index, loc='center', cellLoc='center', 
                             colLoc='center')
            
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1.2, 1.2)
            
            plt.savefig(f"{png_filename}.png", bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error saving PNG: {e}")

    return Result

####################################################################################################################################

def missing_data_plot(data_depth, depth_units, well_data, figsize=(20, 17), save_png=False, png_filename='missing_data', title='WELL LOGS DATA'):
    """
    Plots the well log data with missing data visualization.

    Parameters:
    data_depth (pd.Series): Depth data for the well logs.
    depth_units (str): Units of depth (e.g., 'm' for meters).
    well_data (pd.DataFrame): Well log data with possible missing values.
    figsize (tuple): Size of the figure, default is (20, 17).
    save_png (bool): If True, save the plot as a PNG file. Default is True.
    png_filename (str): Filename for the saved PNG file, without extension. Default is 'missing_data'.

    This function creates a figure with three subplots:
    1. The depth values on the left, with grid lines and centered depth labels.
    2. A missingno matrix plot showing the presence of missing data in the well log data.
    3. A plot on the right showing the count of missing data points per record, filled with colors.
       The colors represent the number of missing data points per record, up to 20 colors.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=figsize, gridspec_kw={'width_ratios': [0.05, 3, 1]})

    ax1.plot(np.sort(data_depth.values), np.sort(data_depth.values), marker='o', linestyle='', color='none')

    ax1.set_xticks([])
    ax1.set_yticks(ax1.get_yticks())
    ax1.invert_yaxis()

    ax1.tick_params(axis='y', labelsize=20, direction='in')
    ax1.set_ylabel(f'DEPTH ({depth_units})', fontdict={'fontsize': 20})

    ax1.grid(visible=True, which='major', axis='both')

    missing_percentage = well_data.isnull().mean() * 100
    well_data_renamed = well_data.rename(columns=lambda col: f"{col} ({100-missing_percentage[col]:.0f}%)" )

    msno.matrix(well_data_renamed, ax=ax2, color=(0.65, 0.65, 0.65), label_rotation=0, sparkline=False)

    ax2.set_yticks([])
    ax2.set_title(title, fontdict={'fontsize': 15})

    missing_counts = well_data.isnull().sum(axis=1)

    color_palette = [
        'green', 'yellow', 'orange', 'red', 'blue', 'purple', 'brown', 'pink', 'grey', 'black',
        'cyan', 'magenta', 'lime', 'gold', 'navy', 'lavender', 'olive', 'teal', 'maroon', 'violet'
    ]

    ax3.plot(missing_counts.values, range(len(missing_counts)), color='black', lw=1)

    max_missing = min(20, missing_counts.max())
    for i in range(1, max_missing + 1):
        mask = missing_counts.values == i
        ax3.fill_betweenx(range(len(missing_counts)), 0, missing_counts.values, where=mask, color=color_palette[i-1], alpha=1)

    ax3.set_ylim(ax2.get_ylim())
    ax3.grid(visible=True, which='major', axis='both')
    ax3.set_yticks([])
    ax3.tick_params(axis='x', labelsize=16, top=True)
    ax3.xaxis.set_ticks_position('both')
    ax3.set_title('MISSING COLUMNS', fontdict={'fontsize': 16})

    legend_patches = [mpatches.Patch(color=color_palette[i-1], label=f'{i} missing columns') for i in range(1, max_missing + 1)]
    ax3.legend(handles=legend_patches, loc='upper right', fontsize=10)

    plt.tight_layout()

    if save_png:
        plt.savefig(png_filename + '.png', bbox_inches='tight')

    plt.show()

####################################################################################################################################

def Plot_Hist(Data, variable_name=None, n_bins='sturges', units='', size=10, language='esp', 
              mean_line=True, median_line=True, relative_frequency=False,
              show_title=True, save_fig=False, fig_filename='histogram', 
              outlier_limits=False, log_transform=False):
    """
    Function to plot the histogram and boxplot of the data with optional log transformation
    
    Parameters
    ----------
    Data : pd.Series
        Data used to construct the histogram and boxplot.
    variable_name=int or str, optional
        Name of variable. Default plot is 'Variable'
    n_bins : int or str, optional
        Number of bins for the histogram. Default is 'sturges'.
    units : str, optional
        Units of the data. Default is ''.
    size : int, optional
        Size of the plot. Default is 10.
    language : str, optional
        Language of the text in the plot ('esp' for Spanish, 'eng' for English). Default is 'esp'.
    mean_line : bool, optional
        Whether to show a vertical line for the mean. Default is True.
    median_line : bool, optional
        Whether to show a vertical line for the median. Default is True.
    relative_frequency : bool, optional
        Whether to plot the histogram with relative frequencies. Default is False.
    show_title : bool, optional
        Whether to show the title of the plot. Default is True.
    save_fig : bool, optional
        Whether to save the figure as an image file. Default is False.
    fig_filename : str, optional
        Filename for the saved figure. Default is 'histogram'.
    outlier_limits : bool, optional
        Whether to show the outlier limits from the boxplot on the histogram. Default is False.
    log_transform : bool, optional
        Whether to apply log transformation to the data. Default is False.

    Returns
    -------
    fig, axs : matplotlib figure and axes
        The figure and axes of the plot.
    """

    if not isinstance(Data, pd.Series):
        Data = pd.Series(Data)
        
    if variable_name is not None:
        Data.name = variable_name

    elif Data.name is None:
        Data.name = "Variable"
    

    Data = Data.copy()
    Data.dropna(inplace=True)

    if log_transform:
        Data = Data[Data > 0]
        Data = np.log(Data)
        if units: units = f"log({units})"

    if language == 'esp':
        title = 'Histograma y Boxplot de %s' % Data.name
        x_label = Data.name + ' [%s]' % units
        y_label_abs = 'Frecuencia Absoluta'
        y_label_rel = 'Frecuencia Relativa [%]'
        mean_label = 'Media'
        median_label = 'Mediana'
        annotation_offset = (0, 0.2)
    elif language == 'eng':
        title = 'Histogram and Boxplot of %s' % Data.name
        x_label = Data.name + ' [%s]' % units
        y_label_abs = 'Absolute Frequency'
        y_label_rel = 'Relative Frequency'
        mean_label = 'Mean'
        median_label = 'Median'
        annotation_offset = (0, 0.2)
    else:
        raise ValueError("language must be 'esp' or 'eng'")

    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [0.75, 5]}, figsize=(size, size*0.75), sharex=True)

    if show_title:
        fig.suptitle(title, size=size*1.5)

    boxplot = axs[0].boxplot(Data, vert=False, meanline=False, 
                             showmeans=True, widths=0.70, 
                             patch_artist=True,
                             boxprops={'facecolor': 'silver'}, 
                             medianprops={'linewidth': '1.4', 'color': 'blue', 'linestyle': '--'}, 
                             meanprops={'marker': 'D', 'markeredgecolor': 'red',
                                        'markerfacecolor': 'red'})
    axs[0].set_yticks([])

    whiskers = [item.get_xdata() for item in boxplot['whiskers']]
    if len(whiskers) > 0:
        lower_whisker, upper_whisker = whiskers[0][1], whiskers[1][1]
    else:
        lower_whisker, upper_whisker = Data.min(), Data.max()

    if relative_frequency:
        hist_data = np.histogram(Data, bins=n_bins)
        hist_heights = hist_data[0] / len(Data) * 100
        axs[1].bar(hist_data[1][:-1], hist_heights, width=np.diff(hist_data[1]), edgecolor='black', facecolor='silver', align='edge')
        y_label = y_label_rel
    else:
        hist_heights, bins, _ = axs[1].hist(Data, bins=n_bins, edgecolor='black', facecolor='silver')
        hist_data = (hist_heights, bins)
        y_label = y_label_abs

    bin_centers = np.diff(hist_data[1]) * 0.5 + hist_data[1][:-1]
    
    if len(bin_centers) < 50:
        for n, (height, x) in enumerate(zip(hist_heights, bin_centers)):
            if height > 0:
                val = int(height) if not relative_frequency else round(height, 2)
                axs[1].annotate(f"{val}", xy=(x, height), xytext=annotation_offset,
                                textcoords="offset points", ha='center', va='bottom', size=10)

    axs[1].set_xlabel(x_label)
    axs[1].set_ylabel(y_label)
    
    if mean_line: axs[1].axvline(x=Data.mean(), color='r', linestyle='--', label=mean_label)
    if median_line: axs[1].axvline(x=Data.median(), color='blue', linestyle='--', label=median_label)
    if outlier_limits:
        axs[1].axvline(x=lower_whisker, color='gray', linestyle='--')
        axs[1].axvline(x=upper_whisker, color='gray', linestyle='--')
    
    axs[1].legend(loc='upper right', fontsize='large')
    plt.subplots_adjust(wspace=0, hspace=0)
    
    if save_fig: plt.savefig(fig_filename+'.png', bbox_inches='tight')
    
    return fig, axs
    
####################################################################################################################################

def Count_Classes(Data, Label, save_csv=False, csv_filename='count_classes'):
    """
    Function to count the classes of a pd.DataFrame

    Parameters
    ----------
    Data : pd.DataFrame
        Data in which a categorical variable will be counted.
    Label : str
        Name of the column where the categorical data is located.
    save_csv : bool, optional
        If True, save the result as a CSV file. Default is False.
    csv_filename : str, optional
        Filename for the saved CSV file, without extension. Default is 'count_classes'.

    Returns
    -------
    pd.DataFrame with the absolute count and percentage.
    """
    Data = Data.copy()
    Data.dropna(inplace=True)

    if Label in Data:
        Categories_original = pd.Series(Data[Label]).value_counts()
        df_Categories_original = pd.concat([Categories_original, Categories_original], axis=1)
        df_Categories_original.reset_index(inplace=True)
        df_Categories_original.columns = [Label, 'COUNT', 'PERCENT']
        df_Categories_original['PERCENT'] = df_Categories_original['PERCENT'].astype(float)
        for i in range(len(Categories_original)):
            df_Categories_original.loc[i, 'PERCENT'] = (df_Categories_original.loc[i, 'COUNT'] * 100.0 / len(Data))
        df_Categories_original.sort_values(by=[Label], inplace=True)
        df_Categories_original.reset_index(inplace=True)
        df_Categories_original.drop(['index'], axis=1, inplace=True)

        if not df_Categories_original['PERCENT'].sum().round(2) == 100:
            print("The sum of 'PERCENT' is not equal to 100. Please check the dataset.")
        
        if save_csv:
            df_Categories_original.to_csv(csv_filename + '.csv', index=False)
        
        return df_Categories_original
    else:
        Categories_Set = pd.Series(Data).value_counts()
        df_Categories_Set = pd.concat([Categories_Set, Categories_Set], axis=1)
        df_Categories_Set.reset_index(inplace=True)
        df_Categories_Set.columns = [Label, 'COUNT', 'PERCENT']
        df_Categories_Set['PERCENT'] = df_Categories_Set['PERCENT'].astype(float)
        for i in range(len(Categories_Set)):
            df_Categories_Set.loc[i, 'PERCENT'] = (df_Categories_Set.loc[i, 'COUNT'] * 100.0 / len(Data))
        df_Categories_Set.sort_values(by=[Label], inplace=True)
        df_Categories_Set.reset_index(inplace=True)
        df_Categories_Set.drop(['index'], axis=1, inplace=True)

        # Check if the sum of 'PERCENT' is 100
        if not df_Categories_Set['PERCENT'].sum().round(2) == 100:
            print("The sum of 'PERCENT' is not equal to 100. Please check the dataset.")
        
        if save_csv:
            df_Categories_Set.to_csv(csv_filename + '.csv', index=False)
        
        return df_Categories_Set

####################################################################################################################################

def barplot_Classes(Data, Class, Label, figsize=(12, 5), save_fig=False, fig_filename='count_classes', palette='bright'):
    """
    Function to create a bar plot of a categorical variable and annotate the bars with the count of each category.

    Parameters
    ----------
    Data : pd.DataFrame
        Data in which a categorical variable will be counted.
    Class : str
        Name of the categorical column to be counted and plotted.
    Label : str
        Name of the column where the categorical data is located.
    figsize : tuple 
        Size of the figure, default is (12, 5).
    save_fig : bool, optional
        If True, save the plot as a PNG file. The default is False.
    fig_filename : str, optional
        Filename for the saved PNG file, without extension. The default is 'count_classes'.
    palette : str or list, optional
        Color palette for the bar plot. Default is 'bright'.

    Returns
    -------
    None
    """
    
    Data = Data.copy()
    Data.dropna(inplace=True)

    fig, ax = plt.subplots(figsize=figsize)
    
    # Check if palette is a list
    if isinstance(palette, list):
        colors = palette
    else:
        colors = sns.color_palette(palette)

    barplot = sns.barplot(ax=ax, data=Data,
                          x=Class,
                          y=Label,
                          palette=colors,
                          hue=Class,
                          dodge=False)
    
    for p in barplot.patches:
        height = p.get_height()
        if height > 0:
            barplot.annotate(f'{height:.2f}', 
                             (p.get_x() + p.get_width() / 2., height),
                             ha='center', va='center', 
                             xytext=(0, 5), 
                             textcoords='offset points')
    
    ax.set_title('Class ' + Class)
    ax.legend()

    if save_fig:
        plt.savefig(fig_filename + '.png', bbox_inches='tight')
    
    plt.show()

####################################################################################################################################

def pairplot_classes(Data, Label, Class, size=2, palette_colors='bright', units=None, save_fig=False, fig_filename='pair_classes'):
    """
    Function to create a pairplot of the classes in a DataFrame with customizable color palette and unit labels.

    Parameters
    ----------
    Data : pd.DataFrame
        Data in which the pairplot will be created. Must include the columns specified in `Label` and `Class`.
    Label : list of str
        List of column names to be included in the pairplot.
    Class : str
        Name of the column where the categorical data is located.
    size : int or float, optional
        Size of each facet in the grid, aspect ratio will be kept to 1. Default is 2.
    palette_colors : str or list of str, optional
        Color palette for the pairplot. Can be a string name for a seaborn palette (default is 'bright') or a list of colors.
    units : list of str, optional
        List of units corresponding to the variables in `Label`. These units will be displayed next to the variable names in the plot.
    save_fig : bool, optional
        If True, save the plot as a PNG file. Default is False.
    fig_filename : str, optional
        Filename for the saved PNG file, without extension. Default is 'pair_classes'.

    Returns
    -------
    None

    """
    Data = Data.copy()
    Data.dropna(inplace=True)

    # Check if units are provided and concatenate them with the label names
    if units is not None:
        if len(units) != len(Label):
            raise ValueError("The length of 'units' must match the length of 'Label'")
        label_with_units = [f"{label} [{unit}]" for label, unit in zip(Label, units)]
    else:
        label_with_units = Label

    data = Data[Label].copy()
    data.columns = label_with_units  # Use labels with units if provided
    data[Class] = Data[Class].copy()

    # Plot with the chosen color palette
    pairplot = sns.pairplot(data, hue=Class, palette=palette_colors, diag_kind='kde', 
                            corner=True, height=size, aspect=1)
    
    pairplot.fig.suptitle(f'Pairplot of {Class}', y=1.02, size=10*(size))

    for ax in pairplot.axes.flat:
        if ax is not None:
            ax.set_xlabel(ax.get_xlabel(), fontsize=6*size)
            ax.set_ylabel(ax.get_ylabel(), fontsize=6*size)
            ax.set_title(ax.get_title(), fontsize=5*size)
            ax.tick_params(axis='both', which='major', labelsize=5*size)

    pairplot._legend.set_title(pairplot._legend.get_title().get_text())
    plt.setp(pairplot._legend.get_title(), fontsize=5*size)
    plt.setp(pairplot._legend.get_texts(), fontsize=5*size)

    if save_fig:
        plt.savefig(fig_filename + '.png', bbox_inches='tight')
    
    plt.show()

####################################################################################################################################

def vertical_proportions(results, category_names, figsize=(10, 5), 
                         save_fig=False, fig_filename='vertical_proportions',
                         palette_colors=None, xlabel='', ylabel='', title=''):
    """
    Function to create a vertical bar plot representing the proportion of each category 
    across multiple items or questions. This function handles lists of different lengths by 
    padding shorter lists with zeros, ensuring all categories are plotted uniformly.

    Parameters
    ----------
    results : dict
        A dictionary where keys represent labels (e.g., wells or questions), and values are 
        lists of proportions or counts for each category. Each list may have a different length.
    category_names : list of str
        List of category labels (e.g., the names of the categories being represented in the 
        vertical bars). Each name corresponds to one section of the horizontal bars.
    figsize : tuple, optional
        Size of the figure to be created, with the default size set to (10, 5).
    save_fig : bool, optional
        If True, the figure will be saved as a PNG file. Default is False.
    fig_filename : str, optional
        The filename for saving the figure if save_fig is True. It does not include the 
        file extension. Default is 'vertical_proportions'.
    palette_colors : list of str, optional
        List of color codes or names to be used for each category. If None, the 'RdYlGn' 
        colormap will be used with a gradient. Default is None.
    xlabel : str, optional
        Label for the x-axis. Default is an empty string.
    ylabel : str, optional
        Label for the y-axis. Default is an empty string.
    title : str, optional
        Title for the plot. Default is an empty string.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects
        The created figure and axes for further customization or display. These can be used 
        to further tweak the appearance of the plot or save it in other formats.

    Notes
    -----
    - The function automatically pads lists in `results` to have the same length, ensuring 
      consistent plotting even when categories differ across labels.
    - Bar labels are displayed with two decimal places for proportions/counts greater than 0.
    - When `palette_colors` is not provided, a colormap with a gradient is used.
    - The x-axis and y-axis labels, as well as the title, can be customized through the 
      `xlabel`, `ylabel`, and `title` parameters.
    - To avoid layout issues, the function applies `plt.tight_layout()` to adjust the spacing 
      and ensure all labels are visible.
    """

    # Find the maximum length of all lists in results
    max_len = max(len(v) for v in results.values())

    # Pad the lists with zeros where needed
    padded_results = {k: v + [0] * (max_len - len(v)) for k, v in results.items()}

    labels = list(padded_results.keys())
    data = np.array(list(padded_results.values()))
    data_cum = data.cumsum(axis=1)

    # If no colors are provided, use 'RdYlGn' colormap
    if palette_colors is None:
        category_colors = plt.colormaps['RdYlGn'](np.linspace(0.15, 0.85, max_len))
    else:
        category_colors = palette_colors

    fig, ax = plt.subplots(figsize=figsize)
    ax.invert_yaxis()
    ax.set_xlim(0, np.sum(data, axis=1).max())

    # Create the horizontal bar chart with the padded data
    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        # Choose text color based on the brightness of the bar color
        if isinstance(color, str):
            text_color = 'white' if np.mean(mcolors.to_rgb(color)) < 0.5 else 'darkgrey'
        else:
            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'darkgrey'

        # Add labels with two decimal places, but only for values > 0
        ax.bar_label(rects, labels=[f'{w:.2f}' if w > 0 else '' for w in widths], label_type='center', color=text_color)

    ax.legend(ncols=len(category_names), bbox_to_anchor=(0, 1), loc='lower left', fontsize='small')

    # Set the xlabel and ylabel, and ensure they are displayed
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12)

    # Adjust layout to ensure all labels are visible
    plt.tight_layout()

    if save_fig:
        plt.savefig(fig_filename + '.png', bbox_inches='tight')

    return fig, ax

####################################################################################################################################

def Class_Boxplots(Data, Label, Clase, classes_order=None, units='', 
                   Limits='auto', figsize=[10, 6], 
                   global_median=False, global_mean=False, 
                   connect_median_line=False, connect_mean_line=True,
                   log_scale=False, language='esp', save_fig=False, fig_filename='class_boxplots'):
    """
    Function to plot boxplots of a property filtered by a categorical variable (class), with customizable order of classes.

    Parameters
    ----------
    Data : pd.DataFrame
        Dataset, must include the variable to construct the boxplots and the class to filter.
    Label : string
        Name of the property to plot, must match the column name in the pd.DataFrame.
    Clase : string
        Name of the categorical variable to filter the boxplots, must match the column name in the pd.DataFrame.
    classes_order : list, optional
        List of classes in the desired order for plotting the boxplots. Default is None, which plots all classes sorted.
    units : str, optional
        Units of the variable, which will be added to the y-axis label. Default is an empty string.
    Limits : list, optional
        Values for the y-axis limits. The default is 'auto' and uses the maximum and minimum value of the property.
    figsize : list, optional
        Width and height values to generate the plot. The default is [10, 6].
    global_median : bool, optional
        Option to mark the value of the global median. The default is False.
    global_mean : bool, optional
        Option to mark the value of the global mean. The default is False.
    connect_mean_line : bool, optional
        Option to show the line connecting the global means. The default is True.
    connect_median_line : bool, optional
        Option to show the line connecting the global medians. The default is False.
    log_scale : bool, optional
        Option to use a logarithmic scale for the y-axis. The default is False.
    language : str, optional
        Language of the text in the plot ('esp' for Spanish, 'eng' for English). The default is 'esp'.
    save_fig : bool, optional
        If True, save the plot as a PNG file. The default is False.
    fig_filename : str, optional
        Filename for the saved PNG file, without extension. The default is 'class_boxplots'.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects
    """

    if language == 'esp':
        title = f'Boxplot de {Label} por clases ({Clase})'
        global_mean_label = 'Media (Global)'
        global_median_label = 'Mediana (Global)'
    elif language == 'eng':
        title = f'Boxplot of {Label} by classes ({Clase})'
        global_mean_label = 'Mean (Global)'
        global_median_label = 'Median (Global)'
    else:
        raise ValueError("language must be 'esp' or 'eng'")

    Data_no_nan = Data.dropna(subset=[Label, Clase])

    # Si se proporciona una lista de clases, usarla en lugar de ordenar las clases
    if classes_order is not None:
        Data_no_nan = Data_no_nan[Data_no_nan[Clase].isin(classes_order)]
        Data_no_nan[Clase] = pd.Categorical(Data_no_nan[Clase], categories=classes_order, ordered=True)
    else:
        classes_order = sorted(Data_no_nan[Clase].unique())  # Si no se proporciona, usar el orden por defecto (ordenado)

    # Crear figura
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(x=Clase, y=Label, data=Data_no_nan, ax=ax, order=classes_order, showmeans=True,
                meanprops={"marker": "D", "markeredgecolor": "red", "markerfacecolor": "red", "markersize": 7},
                medianprops={"linestyle": "--", "linewidth": 1.5, "color": "blue"},
                boxprops={'facecolor': 'silver', 'edgecolor': 'black'})

    ax.set_title(title, size=20, y=1.05)
    ax.set_xlabel(Clase)
    ax.set_ylabel(f'{Label} [{units}]')
    ax.grid()

    # Ajustar los límites del eje y
    if Limits == 'auto':
        if log_scale:
            ax.set_ylim(0.001, Data_no_nan[Label].max())
        else:
            ax.set_ylim(Data_no_nan[Label].min(), Data_no_nan[Label].max())
    else:
        ax.set_ylim(Limits[0], Limits[1])

    if log_scale:
        ax.set_yscale('log')

    if global_mean:
        ax.axhline(y=Data_no_nan[Label].mean(), color='r', linestyle='-', label=global_mean_label, alpha=1)
    if global_median:
        ax.axhline(y=Data_no_nan[Label].median(), color='darkblue', linestyle='-', label=global_median_label, alpha=1)

    if connect_mean_line:
        mean_points = [Data_no_nan[Data_no_nan[Clase] == class_label][Label].mean() for class_label in classes_order]
        for i in range(len(mean_points) - 1):
            ax.plot([i, i+1], [mean_points[i], mean_points[i+1]], color='red', linestyle='--', label='Mean')

    if connect_median_line:
        median_points = [Data_no_nan[Data_no_nan[Clase] == class_label][Label].median() for class_label in classes_order]
        for i in range(len(median_points) - 1):
            ax.plot([i, i+1], [median_points[i], median_points[i+1]], color='blue', linestyle='--', label='Median')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    if save_fig:
        plt.savefig(fig_filename + '.png', bbox_inches='tight')

    return fig, ax

####################################################################################################################################

def PCA_model(n_comp, data):
    '''
    Description:
    This function performs Principal Component Analysis (PCA) on the provided dataset.

    Parameters:
    n_comp (int): The number of principal components to compute.
    data (array-like): The dataset to be transformed.

    Returns:
    tuple: A tuple containing:
        - PC_list (list): A list of principal component names (e.g., ['PC1', 'PC2', ..., 'PCn']).
        - pca_model (PCA object): The fitted PCA model.
        - pca_data (array-like): The transformed data in the principal component space.
    '''
    PC_list = ['PC' + str(i) for i in range(1, n_comp + 1)]
    pca_model = PCA(n_components=n_comp, random_state=0)
    pca_data = pca_model.fit_transform(data) 
    return PC_list, pca_model, pca_data

####################################################################################################################################

def Tabla_PCA(modelo, save_csv=False, csv_filename='pca_table'):
    """
    Function to estimate the contribution to the variance of each of the principal components of a model.

    Parameters
    ----------
    modelo : sklearn.model
        Fitted model (data type specific to sklearn).
    save_csv : bool, optional
        Whether to save the table as a CSV file. Default is False.
    csv_filename : str, optional
        The name of the CSV file to save the table. Default is 'pca_table.csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with the results of the explained variance and the explained variance ratio.
    """
    df_pca_table1 = pd.DataFrame(data=modelo.explained_variance_, columns=['Explained Variance'])
    df_pca_table2 = pd.DataFrame(data=modelo.explained_variance_ratio_, columns=['Explained Variance Ratio'])
    componentes = ['PC ' + str(i + 1) for i in range(modelo.n_components)]
    df_componentes = pd.DataFrame(data=componentes, columns=['Principal Component'])
    df_pca_table = pd.concat([df_componentes, df_pca_table1, df_pca_table2], axis=1)
    
    if save_csv:
        df_pca_table.to_csv(csv_filename + '.csv', index=False)
    
    return df_pca_table


####################################################################################################################################

def Graficar_varianza_PCA(modelo, figsize=(10, 5), save_fig=False, fig_filename='PCA_variance_graph', language='esp'):
    """
    Function to generate the graph of explained and cumulative variance.

    Parameters
    ----------
    modelo : sklearn.model
        Fitted model (sklearn's own data type).
    save_fig : bool, optional
        If True, saves the figure. The default value is False.
    fig_filename : str, optional
        Name of the figure file. The default value is 'PCA_variance_graph'.
    language : str, optional
        Language of the texts in the graph ('esp' for Spanish, 'eng' for English). The default value is 'esp'.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects
        Graph of explained and cumulative variance.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if language == 'esp':
        ax.set_title('Varianza Explicada por Componentes Principales')
        ax.set(ylabel='Varianza Explicada', xlabel='Componentes Principales')
        varianza_label = 'Varianza explicada'
        acumulada_label = 'Varianza acumulada'
    elif language == 'eng':
        ax.set_title('Explained Variance by Principal Components')
        ax.set(ylabel='Explained Variance', xlabel='Principal Components')
        varianza_label = 'Explained variance'
        acumulada_label = 'Cumulative variance'
    else:
        raise ValueError("Unsupported language. Use 'esp' for Spanish or 'eng' for English.")
    
    componentes = range(1, len(modelo.explained_variance_) + 1)
    ax.bar(componentes, modelo.explained_variance_ratio_, alpha=0.6, label=varianza_label)
    
    varianza_acumulada = np.cumsum(modelo.explained_variance_ratio_)
    ax.plot(componentes, varianza_acumulada, color='red', marker='o', label=acumulada_label)
    
    # Añadir etiquetas de porcentaje en las barras
    for i, v in enumerate(modelo.explained_variance_ratio_):
        ax.text(i + 1, v + 0.01, f"{v:.2%}", ha='center', va='bottom')
    
    # Añadir etiquetas de porcentaje en la línea acumulativa
    for i, v in enumerate(varianza_acumulada):
        ax.text(i + 1, v + 0.01, f"{v:.2%}", ha='center', va='bottom')
    
    ax.legend()
    
    # Ajustar el eje X para que solo muestre números enteros
    ax.set_xticks(componentes)
    
    ax.grid(False)

    if save_fig:
        plt.savefig(fig_filename + '.png', bbox_inches='tight')
    
    plt.show()

    return fig, ax

####################################################################################################################################

def Scatterplot_PCA(Data, x, y, classes, size=5, save_fig=False, fig_filename='scatter_pca', language='esp', palette='bright'):
    """
    Function to generate a 2D scatter plot for PCA components

    Parameters
    ----------
    Data : DataFrame
        Data containing the PCA components.
    x : str
        Name of the column to be used for the x-axis.
    y : str
        Name of the column to be used for the y-axis.
    classes : str
        Name of the column containing class labels for coloring the points.
    save_fig : bool, optional
        If True, saves the figure. The default value is False.
    fig_filename : str, optional
        Name of the figure file. The default value is 'scatter_pca'.
    language : str, optional
        Language of the texts in the graph ('esp' for Spanish, 'eng' for English). The default value is 'esp'.
    palette : str, optional
        Color palette for the plot. The default value is 'bright'.

    Returns
    -------
    None
    """
    
    sns.lmplot(x=x, y=y, data=Data, fit_reg=False, legend=True, hue=classes, height=size, aspect=1, palette=palette)
    
    if language == 'esp':
        plt.title(f'Gráfico de dispersión con {x} y {y}')
        plt.xlabel(x)
        plt.ylabel(y)
    elif language == 'eng':
        plt.title(f'Scatter Plot with {x} and {y}')
        plt.xlabel(x)
        plt.ylabel(y)
    else:
        raise ValueError("Unsupported language. Use 'esp' for Spanish or 'eng' for English.")
    
    if save_fig:
        plt.savefig(fig_filename+'.png', bbox_inches='tight')
    
    plt.show()

####################################################################################################################################

def f_statistics_plot(flabel, fstatistics, figsize=(10, 5), save_fig=False, fig_filename='fstatistics_plot'):
    """
    Function to generate a horizontal bar plot of F-statistics

    Parameters
    ----------
    flabel : list
        List of labels for the F-statistics.
    fstatistics : list or array
        List or array of F-statistics values.
    figsize : tuple, optional
        Size of the figure. The default value is (10, 5).
    save_fig : bool, optional
        If True, saves the figure. The default value is False.
    fig_filename : str, optional
        Name of the figure file. The default value is 'fstatistics_plot'.

    Returns
    -------
    fig : Figure
        The matplotlib figure object.
    ax : Axes
        The matplotlib axes object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    class_plot = ax.barh(flabel, fstatistics, color=mcolors.TABLEAU_COLORS)
    
    ax.set_title('F-statistics (sorted)')
    
    # Ajustar las etiquetas dentro de las barras
    ax.bar_label(class_plot, fmt='%.1f', label_type='edge', padding=3)
    
    ax.invert_yaxis()
    ax.set_xlim(0, fstatistics.max() * 1.1)  # Aumentar el límite derecho para evitar el corte de las etiquetas

    if save_fig:
        plt.savefig(fig_filename + '.png', bbox_inches='tight')
    
    plt.show()
    return fig, ax

####################################################################################################################################

def mutual_info_plot(mlabel, minfo, figsize=(10, 5), save_fig=False, fig_filename='mutualinfo_plot'):
    """
    Function to generate a horizontal bar plot of Mutual Information

    Parameters
    ----------
    mlabel : list
        List of labels for the Mutual Information values.
    minfo : list or array
        List or array of Mutual Information values.
    figsize : tuple, optional
        Size of the figure. The default value is (10, 5).
    save_fig : bool, optional
        If True, saves the figure. The default value is False.
    fig_filename : str, optional
        Name of the figure file. The default value is 'mutualinfo_plot'.

    Returns
    -------
    fig : Figure
        The matplotlib figure object.
    ax : Axes
        The matplotlib axes object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    class_plot = ax.barh(mlabel, minfo, color=mcolors.TABLEAU_COLORS)
    
    ax.set_title('Mutual Information (sorted)')
    
    # Ajustar las etiquetas dentro de las barras
    ax.bar_label(class_plot, fmt='%.2f', label_type='edge', padding=3)
    
    ax.invert_yaxis()
    ax.set_xlim(0, minfo.max() * 1.1)  # Aumentar el límite derecho para evitar el corte de las etiquetas

    if save_fig:
        plt.savefig(fig_filename + '.png', bbox_inches='tight')
    
    plt.show()
    return fig, ax

####################################################################################################################################

def dendrogram_plot(Dendro_data, truncate='lastp', p=5, figsize=(10, 5), save_fig=False, fig_filename='dendrogram_plot'):
    """
    Function to generate a dendrogram plot

    Parameters
    ----------
    Dendro_data : array-like
        The hierarchical clustering encoded as a linkage matrix.
    truncate : str, optional
        Truncation mode for the dendrogram. The default value is 'lastp'.
    p : int, optional
        The number of last clusters to show in the dendrogram. The default value is 5.
    figsize : tuple, optional
        Size of the figure. The default value is (10, 5).
    save_fig : bool, optional
        If True, saves the figure. The default value is False.
    fig_filename : str, optional
        Name of the figure file. The default value is 'dendrogram_plot'.

    Returns
    -------
    fig : Figure
        The matplotlib figure object.
    ax : Axes
        The matplotlib axes object.
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    dendrogram(Dendro_data, truncate_mode=truncate, p=p, leaf_rotation=90., leaf_font_size=12., show_contracted=True, ax=ax)
    ax.set_title('Dendrogram'); 
    ax.set_xlabel('Samples'); 
    ax.set_ylabel('Euclidean Distance')

    if save_fig:
        plt.savefig(fig_filename+'.png', bbox_inches='tight')
        
    plt.show()
    return fig, ax

####################################################################################################################################

def Graficar_Set(Datos, Entrenamiento, Validacion, Etiqueta, Conteo='COUNT', figsize=(10, 5), save_fig=False, fig_filename='set_count_classes', color_palette=None):
    """
    Function to count classes for training and validation sets during the implementation of supervised methods,
    along with the class count of the complete dataset. The bars for each set (original, training, validation)
    will be displayed side by side for each class.

    Parameters
    ----------
    Datos : pd.DataFrame
        Original dataset.
    Entrenamiento : pd.DataFrame
        Training dataset.
    Validacion : pd.DataFrame
        Validation dataset.
    Etiqueta : string
        Name of the column with the classification.
    Conteo : string, optional
        Name of the column with the count values. Default is 'COUNT'.
    figsize : tuple, optional
        Size of the figure. Default is (10, 5).
    save_fig : bool, optional
        If True, saves the figure. Default is False.
    fig_filename : string, optional
        Name of the figure file. Default is 'set_count_classes'.
    color_palette : list, optional
        List of colors to use for the bars. Default is None, which uses seaborn's default palette.

    Returns
    -------
    Figure and axes with histograms and class counts.
    """

    # Definir las clases únicas y el número total de clases
    clases = Datos[Etiqueta].unique()
    n_clases = len(clases)
    
    # Crear el DataFrame consolidado con los valores de conteo para cada dataset
    datos_concat = pd.DataFrame({
        'Clase': np.repeat(clases, 3),  # Se repite cada clase tres veces (una para cada conjunto)
        'Conteo': np.concatenate([Datos[Conteo].values, Entrenamiento[Conteo].values, Validacion[Conteo].values]),
        'Conjunto': ['Original'] * len(Datos) + ['Entrenamiento'] * len(Entrenamiento) + ['Validacion'] * len(Validacion)
    })

    # Definir la paleta de colores si no se proporciona
    if color_palette is None:
        color_palette = sns.color_palette("bright", 3)  # Default palette if not provided
    
    # Definir ancho de barras
    bar_width = 0.2
    indices_clases = np.arange(n_clases)  # Posiciones para las clases
    
    fig, ax = plt.subplots(figsize=figsize)

    # Graficar cada grupo de barras
    original_bars = ax.bar(indices_clases - bar_width, Datos[Conteo], bar_width, label='Original', color=color_palette[0])
    entrenamiento_bars = ax.bar(indices_clases, Entrenamiento[Conteo], bar_width, label='Entrenamiento', color=color_palette[1])
    validacion_bars = ax.bar(indices_clases + bar_width, Validacion[Conteo], bar_width, label='Validacion', color=color_palette[2])

    # Añadir etiquetas para las barras
    ax.bar_label(original_bars, fmt='%.2f')
    ax.bar_label(entrenamiento_bars, fmt='%.2f')
    ax.bar_label(validacion_bars, fmt='%.2f')
    
    # Etiquetas y título
    ax.set_xlabel('Clases')
    ax.set_ylabel(Conteo)
    ax.set_title('Conteo de clases por conjunto de datos')
    ax.set_xticks(indices_clases)
    ax.set_xticklabels(clases)

    ax.legend()

    # Guardar la figura si es necesario
    if save_fig:
        plt.savefig(fig_filename + '.png', bbox_inches='tight')
    
    plt.show()
    
    return fig, ax

####################################################################################################################################

def ConfusionMatrix_plot(ConfusionMatrix, classes, figsize=(5, 5), colormap='viridis', save_fig=False, fig_filename='confusion_matrix'):
    '''
    Description:
    This function plots a confusion matrix using the provided data and allows customization of the plot's appearance.

    Parameters:
    ConfusionMatrix (array-like): The confusion matrix to be displayed.
    classes (list): The list of class labels to be displayed on the axes.
    figsize (tuple, optional): The size of the figure (default is (5, 5)).
    colormap (str, optional): The colormap to be used for the plot (default is 'viridis').
    save_fig (bool, optional): Whether to save the figure as a file (default is False).
    fig_filename (str, optional): The filename to use if saving the figure (default is 'confusion_matrix').

    Returns:
    tuple: A tuple containing the figure and axes objects of the plot.
    '''
    fig, ax = plt.subplots(figsize=figsize)
    disp = metrics.ConfusionMatrixDisplay(ConfusionMatrix, display_labels=classes)
    disp.plot(ax=ax, cmap=colormap, colorbar=False)
    ax.grid(visible=False)
    ax.set_title('Confusion Matrix')
    
    if save_fig:
        plt.savefig(fig_filename + '.png', bbox_inches='tight')
    
    return fig, ax

####################################################################################################################################

#####################################################################################################################################

def Boxplots(Data,Labels,Limits='auto',figsize=[8,10]):
    """
    Funcion para calcular boxplots de una serie de datos dada.

    Parameters
    ----------
    Data : pd.DataFrame
        Conjunto de datos con las propiedades a las que se les graficara el boxplot.
    Labels : List
        Lista de propiedades para graficar los boxplots. Los boxplots se generan en el mismo orden que esta variable. 
        Los nombres deben coincidir exactamente con los nombres de las columnas en el pd.DataFrame
    Limits : list, optional
        Valores para los limites del eje y en los boxplots. The default is 'auto' y este valor usa el minimo y el maximo de la
        primer propiedad graficada.
    figsize : list, optional
        Ancho y alto de la grafica. The default is [8,10].

    Returns
    -------
    Figuras y ejes.

    """
    Data.dropna(axis=0, inplace=True)
    fig, axs = plt.subplots(1, len(Labels), figsize=figsize, sharey=True)
    title = 'Boxplots de {log}'.format(log=[Label for Label in Labels])
    fig.suptitle(title, size=20, y=0.92)
    
    for num, label in enumerate(Labels):
        if len(Labels) == 1:
            axs.boxplot(Data[label],
                            vert=True, meanline=True, showmeans=True, widths=0.50, patch_artist=True, 
                            boxprops={'facecolor':'lightgrey'}, medianprops={'color':'blue','linewidth':2},
                            meanprops={'color':'red','linewidth':2})
            axs.set_title('%s' %label, y=-0.04)
            if Limits == 'auto':
                axs.set_ylim([Data[label].min(),Data[label].max()])
            else:
                axs.set_ylim(Limits[0], Limits[1])
                
            axs.set_xticks([]); axs.set_ylabel(label)
        else:
            axs[num].boxplot(Data[label],
                             vert=True, meanline=True, showmeans=True, widths=0.50, patch_artist=True,
                             boxprops={'facecolor':'lightgrey'}, medianprops={'color':'blue','linewidth':2},
                            meanprops={'color':'red','linewidth':2})
            axs[num].set_title('%s' %label, y=-0.04)
            axs[num].set_xticks([])
        
        plt.subplots_adjust(wspace=0, hspace=0)
    
    return(fig,axs)


#####################################################################################################################################

#####################################################################################################################################

def Plot_Well(Data, title, logs, colors, units, ref='DEPTH', ref_units='m', size=[17,17], median=False, mean=False, save_fig=False, fig_filename='well_plot'):
    """
    Function to plot well log data.
    
    Parameters
    ----------
    Data : pd.DataFrame
        Dataset that requires at least one continuous variable (well log data).
    title : str
        Title of the plot.
    logs : List
        Names of the well log data to be plotted. The curves are plotted in the same order as this list.
    colors : List
        Colors used to plot each of the log curves.
    units : List
        Units of measurement for the logs.
    ref : str, optional
        Name of the reference column (depth). The default is 'DEPTH'.
    ref_units : str, optional
        Units of the reference column (depth). The default is 'm'.
    size : List, optional
        Pair of integer values defining the size of the plot. The default is [17,17].
    median : bool, optional
        If True, plots a vertical line with the median. The default is False.
    mean : bool, optional
        If True, plots a vertical line with the mean. The default is False.
    save_fig : bool, optional
        If True, saves the figure as a PNG file. The default is False.
    fig_filename : str, optional
        The name of the PNG file to save the figure. The default is 'well_plot'.
    
    Returns
    -------
    Figure and axes.
    """
    
    if not isinstance(Data, pd.DataFrame):
        raise ValueError("Data debe ser un DataFrame de pandas.")
    
    for col in logs + [ref]:
        if col not in Data.columns:
            raise ValueError(f"La columna '{col}' no existe en Data.")
    
    fig, axes = plt.subplots(1, len(logs), figsize=size, sharey=True)
    fig.suptitle(title, fontsize=24, x=0.5, y=0.95, ha='center')
    fig.subplots_adjust(wspace=0.2)
    
    if len(logs) == 1:
        axes = [axes]
    
    for num, ax in enumerate(axes):
        if num == 0:
            axes[num] = plt.subplot2grid((1, len(logs)), (0, num), rowspan=1, colspan=1)
            axes[num].plot(Data[logs[num]].values, Data[ref].values, c=colors[num], lw=0.5)
            axes[num].set_xlim(math.floor(min(Data[logs[num]].values)), math.ceil(max(Data[logs[num]].values)))
            axes[num].set_ylim(math.floor(max(Data[ref].values)), math.floor(min(Data[ref].values)))
            axes[num].xaxis.set_ticks_position("top")
            axes[num].xaxis.set_label_position("top")
            axes[num].set_ylabel(ref + ' (%s)' % ref_units)
            axes[num].set_xlabel(logs[num] + '\n' + units[num])
            axes[num].grid()
            if median:
                axes[num].axvline(x=Data[logs[num]].median(), color='darkblue', linestyle='--', label='Global Median', linewidth=2)
            if mean:
                axes[num].axvline(x=Data[logs[num]].mean(), color='r', linestyle='--', label='Global Mean', linewidth=2)
            if median or mean:
                axes[num].legend(loc='best', fontsize='medium')
        else:
            axes[num] = plt.subplot2grid((1, len(logs)), (0, num), rowspan=1, colspan=1, sharey=axes[0])
            axes[num].plot(Data[logs[num]].values, Data[ref].values, c=colors[num], lw=0.5)
            axes[num].set_xlim(math.floor(min(Data[logs[num]].values)), math.ceil(max(Data[logs[num]].values)))
            axes[num].set_ylim(math.floor(max(Data[ref].values)), math.floor(min(Data[ref].values)))
            axes[num].xaxis.set_ticks_position("top")
            axes[num].xaxis.set_label_position("top")
            axes[num].set_xlabel(logs[num] + '\n' + units[num])
            axes[num].grid()
            if median:
                axes[num].axvline(x=Data[logs[num]].median(), color='darkblue', linestyle='--', label='Global Median', linewidth=2)
            if mean:
                axes[num].axvline(x=Data[logs[num]].mean(), color='r', linestyle='--', label='Global Mean', linewidth=2)
            if median or mean:
                axes[num].legend(loc='best', fontsize='medium')
                
    for ax in axes[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)  
    
    if save_fig:
        plt.savefig(fig_filename + '.png', bbox_inches='tight')
    
    return fig, axes

#####################################################################################################################################

def Plot_Well_Fill(Data, Clase, Paleta_colores, title, logs, colors, units, ref='DEPTH', ref_units='m', size=[17,17], median=False, mean=False, save_fig=False, fig_filename='well_fill_plot'):
    """
    Function to plot well log data with fill based on a categorical variable.

    Parameters
    ----------
    Data : pd.DataFrame
        Dataset that requires at least one continuous variable (well log data) and one categorical variable.
    Clase : str
        Name of the categorical variable used to fill the log curves.
    Paleta_colores : dict
        Dictionary with color-class pairs. This variable controls the legend.
    title : str
        Title of the plot.
    logs : List
        Names of the well log data to be plotted. The curves are plotted in the same order as this list.
    colors : List
        Colors used to plot each of the log curves.
    units : List
        Units of measurement for the logs.
    ref : str, optional
        Name of the reference column (depth). The default is 'DEPTH'.
    ref_units : str, optional
        Units of the reference column (depth). The default is 'm'.
    size : List, optional
        Pair of integer values defining the size of the plot. The default is [17,17].
    median : bool, optional
        If True, plots a vertical line with the median. The default is False.
    mean : bool, optional
        If True, plots a vertical line with the mean. The default is False.
    save_fig : bool, optional
        If True, saves the figure as a PNG file. The default is False.
    fig_filename : str, optional
        The name of the PNG file to save the figure. The default is 'well_fill_plot'.

    Returns
    -------
    Figure and axes.

    """

    if not isinstance(Data, pd.DataFrame):
        raise ValueError("Data debe ser un DataFrame de pandas.")
    
    for col in logs + [ref]:
        if col not in Data.columns:
            raise ValueError(f"La columna '{col}' no existe en Data.")
            
    patches = [mpatches.Patch(color=color, label=f"Group {label}") for color, label in Paleta_colores.items()]

    fig, axes = plt.subplots(1, len(logs), figsize=size, sharey=True)
    fig.suptitle(title, fontsize=24, x=0.5, y=0.95, ha='center')
    fig.subplots_adjust(wspace=0.2) 

    if len(logs) == 1:
        axes = [axes]

    legend_entries = patches.copy() 
    
    for num, ax in enumerate(axes):
        ax.plot(Data[logs[num]].values, Data[ref].values, c=colors[num], lw=0.5)

        for color, label in Paleta_colores.items():
            ax.fill_betweenx(Data[ref].values, Data[logs[num]].values, 
                             math.floor(min(Data[logs[num]].values)), where=Data[Clase] == label, color=color, interpolate=True)

        ax.set_xlim(math.floor(min(Data[logs[num]].values)), math.ceil(max(Data[logs[num]].values)))
        ax.set_ylim(math.floor(max(Data[ref].values)), math.floor(min(Data[ref].values)))
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.set_xlabel(f"{logs[num]}\n{units[num]}")
        ax.grid()

        if num == 0:
            ax.set_ylabel(f"{ref} ({ref_units})")

        if median:
            median_line = ax.axvline(x=Data[logs[num]].median(), color='darkblue', linestyle='--', label='Global Median', linewidth=2)
            if num == 0:
                legend_entries.append(median_line)
        if mean:
            mean_line = ax.axvline(x=Data[logs[num]].mean(), color='r', linestyle='--', label='Global Mean', linewidth=2)
            if num == 0:
                legend_entries.append(mean_line)

    for ax in axes[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)

    fig.legend(handles=legend_entries, bbox_to_anchor=(0.93, 0.85), loc=2, borderaxespad=0., fontsize=15)

    if save_fig:
        plt.savefig(fig_filename + '.png', bbox_inches='tight')

    return fig, axes

#####################################################################################################################################

####################################################################################################################################

def Remove_Outliers(Data, Label):
    """
    Funcion para remover los valores atipicos de un conjunto de datos. Se emplea el criterio de Tukey donde todos los valores 
    por encima de 1.5 veces el rango intercuantil mas el tercer cuantil se remueven asi como los valores 1.5 veces el rango 
    intercuantil menos el primer cuantil.

    Parameters
    ----------
    Data : pd.DataFrame
        Conjunto de datos a los que se les removeran los atipicos.
    Label : String
        Nombre de la columna con los datos a los cuales se les removeran los atipicos.

    Returns
    -------
    pd.DataFrame con las etiquetas de valores atipicos.

    """
    Q1 = Data[Label].quantile(0.25); Q3 = Data[Label].quantile(0.75); IQR = Q3 - Q1
    Data['outlier_%s' %Data[Label].name] = [1 if x > Q3 + 1.5*IQR or x < Q1 - 1.5*IQR else 0 for x in Data[Label]]
    return(Data)

####################################################################################################################################

def Silhouette_plots(Data, No_clusters):
    """
    Funcion para graficar los resultados de la implementacion del metodo de la silueta. 
    Se utiliza el metodo de KMeans para estimar los clusters y sus coeficientes de silueta.

    Parameters
    ----------
    Data : pd.DataFrame
        Datos de las caracteristicas (features) a las que se les calculara el coeficiente de la silueta.
    No_clusters : list of int
        Numero de clusters para los que se estimara el coeficiente de la silueta.

    Returns
    -------
    Figura. Graficas de dispersion y del coeficiente de silueta para cada numero de clusters.

    """
    ## Se inicializa la grafica en funcion de los numeros de clusters.
    range_n_clusters = No_clusters
    for n_clusters in range_n_clusters:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(Data.to_numpy()) + (n_clusters + 1) * 10])
    ## Se generan los clusters mediante el metodo de KMeans.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(Data.to_numpy())
    ## Se obtiene el coeficiente de la silueta, se imprimen los valores en pantalla y en un pd.DataFrame.
        silhouette_avg = silhouette_score(Data.to_numpy(), cluster_labels)
        print("For n_clusters =", n_clusters, 
              "The average silhouette_score is :", silhouette_avg,)
        sample_silhouette_values = silhouette_samples(Data.to_numpy(), cluster_labels)
    ## Se crea el grafico de dispersion (Scatterplot) para visualizar los datos y los clusters. El grafico de dispersion
    ## se construye con las dos primeras caracteristicas (features) introducidas en los Datos
        y_lower = 10
        for i in range(n_clusters):
        
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7,)
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
        ## Se marca con una linea recta el maximo valor del coeficiente de silueta para cada grafica.
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ## Se etiquetan y dibujan los centros de los clusters en la grafica
        ax1.set_yticks([])
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(Data.to_numpy()[:, 0], Data.to_numpy()[:, 1], 
                    marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")
        centers = clusterer.cluster_centers_
        ax2.scatter(centers[:, 0], centers[:, 1], marker="o", c="white", alpha=1, s=200, edgecolor="k",)
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")
        ## Parametros miscelaneos 
        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")
        plt.suptitle("Silhouette analysis for KMeans clustering on sample data with n_clusters = %d" % n_clusters,
                     fontsize=14, fontweight="bold",)
    return(fig, (ax1, ax2))


####################################################################################################################################

def Silhouette(Data,ks=range(2, 30)):
    """
    Funcion para generar una grafica de coeficientes de silueta, utilizada durante el proceso de analisis
    del numero optimo de clusters

    Parameters
    ----------
    Data : pd.DataFrame
        Datos a partir de los cuales se generara la grafica.
    ks : list, optional
        Lista de numeros a los cuales se les estimara el coeficiente de silueta. The default is range(2, 30).

    Returns
    -------
    Grafica y tiempo de ejecucion.

    """
    start_time = time.time()
    silhouette=[]

    for i in ks:
        model = KMeans(n_clusters= i, n_init=1000, random_state=42)
        model.fit(Data)
        labels = model.labels_
        score = silhouette_score(Data, labels, metric='euclidean', sample_size=50000)
        silhouette.append(score)
    elapsed_time = time.time() - start_time
    df_silhouette = pd.DataFrame(data=zip(ks,silhouette), columns=['N_Clusters','Silhouette_score'])
    min_score1 = df_silhouette.sort_values(by='Silhouette_score',ascending=True,ignore_index=True).N_Clusters[0]
    min_score2 = df_silhouette.sort_values(by='Silhouette_score',ascending=True,ignore_index=True).N_Clusters[1]
    min_score3 = df_silhouette.sort_values(by='Silhouette_score',ascending=True,ignore_index=True).N_Clusters[2]

    print(f"Elapsed time to perform silhouette analysis (range of clusters tested: {ks}): {elapsed_time:.3f} seconds")

    plt.title('Método de silueta')
    plt.axvline(min_score1, linestyle='--', color='g', label=str(min_score1))
    plt.axvline(min_score2, linestyle=':', color='g', label=str(min_score2))
    plt.axvline(min_score3, linestyle='-.', color='g', label=str(min_score3))
    plt.plot(ks, silhouette, '*-', label='Silhouette')
    plt.xlabel('Número de clústeres')
    plt.ylabel('Coeficiente de silueta')
    plt.legend()
    return()


####################################################################################################################################

def CH_Index(Data,ks=range(2,30)):
    """
    Funcion para obtener y graficar el indice de Calinsky-Harabasz, empleada para el analisis del numero optimo de clusters.

    Parameters
    ----------
    Data : pd.DataFrame
        Datos con los cuales se realizara el proceso de agrupamiento (clustering).
    ks : list, optional
        Numeros de clusters para estimar el coeficiente de Calinsky-Harabasz. The default is range(2,30).

    Returns
    -------
    Grafica del coeficiente de Calinsky-Harabaz contra el numero de clusters.

    """
    start_time = time.time()
    vrc=[]
    for i in ks:
        model = KMeans(n_clusters= i, n_init=1000, random_state=42)
        model.fit(Data)
        labels = model.labels_
        score = calinski_harabasz_score(Data, labels)
        vrc.append(score)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to perform vrc analysis (range of clusters tested: {ks}): {elapsed_time:.3f} seconds")
    df_CH = pd.DataFrame(data=zip(ks,vrc), columns=['N_Clusters','Calinsky_Harabaz_score'])
    min_score1 = df_CH.sort_values(by='Calinsky_Harabaz_score',ascending=True,ignore_index=True).N_Clusters[0]
    min_score2 = df_CH.sort_values(by='Calinsky_Harabaz_score',ascending=True,ignore_index=True).N_Clusters[1]
    min_score3 = df_CH.sort_values(by='Calinsky_Harabaz_score',ascending=True,ignore_index=True).N_Clusters[2]

    plt.title('Índice de Calinski-Harabasz (Variance Ratio Criterion, VRC)')
    plt.axvline(min_score1, linestyle='--', color='g', label=str(min_score1))
    plt.axvline(min_score2, linestyle=':', color='g', label=str(min_score2))
    plt.axvline(min_score3, linestyle='-.', color='g', label=str(min_score3))
    plt.plot(ks, vrc, '*-', label='VRC')
    plt.xlabel('Número de clústeres')
    plt.ylabel('Índice de Calinski-Harabasz')
    plt.legend()
    plt.show()
    return()


####################################################################################################################################

def DB_Index(Data,ks=range(2, 30)):
    """
    Funcion para graficar el indice de Davies-Bouldin, empleado en el proceso de determinacion del numero optimo de clusters

    Parameters
    ----------
    Data : pd.DataFrame
        Datos a partir de los cuales se realizara el agrupamiento.
    ks : list, optional
        Numeros de clusters para los que se estimara el coeficiente de Davies-Bouldin. The default is range(2, 30).

    Returns
    -------
    Grafica con los coeficientes de Davies-Bouldin contra el numero de clusters.

    """
    start_time = time.time()
    dbi=[]

    for i in ks:
        model = KMeans(n_clusters= i, n_init=1000, random_state=42)
        model.fit(Data)
        labels = model.labels_
        score = davies_bouldin_score(Data, labels)
        dbi.append(score)
    elapsed_time = time.time() - start_time

    print(f"Elapsed time to perform dbi analysis (range of clusters tested: {ks}): {elapsed_time:.3f} seconds")
    df_DB = pd.DataFrame(data=zip(ks,dbi), columns=['N_Clusters','Davies_Bouldin_score'])
    min_score1 = df_DB.sort_values(by='Davies_Bouldin_score',ascending=False,ignore_index=True).N_Clusters[0]
    min_score2 = df_DB.sort_values(by='Davies_Bouldin_score',ascending=False,ignore_index=True).N_Clusters[1]
    min_score3 = df_DB.sort_values(by='Davies_Bouldin_score',ascending=False,ignore_index=True).N_Clusters[2]
    
    plt.title('Índice de Davies-Bouldin')
    plt.axvline(min_score1, linestyle='--', color='g', label=str(min_score1))
    plt.axvline(min_score2, linestyle=':', color='g', label=str(min_score2))
    plt.axvline(min_score3, linestyle='-.', color='g', label=str(min_score3))
    plt.plot(ks, dbi, '*-', label='Davies-Bouldin')
    plt.xlabel('Número de clústeres')
    plt.ylabel('Índice de Davies-Bouldin')
    plt.legend()
    plt.show()
    return()


#################################################################################################################################### 

def optimalK(data, nrefs=3, maxClusters=15):
    """
    Funcion para estimar la grafica de compactamiento por el metodo de "Gap Statistics". 
    El metodo de clustering de referencia es KMeans.

    Parameters
    ----------
    data : pd.DataFrame
        Datos de las caracteristicas (features) a las que se les calculara el coeficiente de la silueta.
    nrefs : int, optional
        Numero de sub conjuntos de datos de referencia para la estimacion del "Gap Statistics". The default is 3.
    maxClusters : int, optional
        Numero maximo de clusters para la implementacion del metodo. The default is 15.

    Returns
    -------
    pd.DataFrame con los pares numero de cluster - gap statistics.

    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
    for gap_index, k in enumerate(range(1, maxClusters)):
        refDisps = np.zeros(nrefs)
        for i in range(nrefs):
            randomReference = np.random.random_sample(size=data.shape)
            km = KMeans(k)
            km.fit(randomReference)
            refDisp = km.inertia_
            refDisps[i] = refDisp
        km = KMeans(k)
        km.fit(data)
        origDisp = km.inertia_
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)
        gaps[gap_index] = gap
        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)
    return (resultsdf)


#####################################################################################################################################



#####################################################################################################################################



#####################################################################################################################################



#####################################################################################################################################









