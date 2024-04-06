
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


SPARSITIES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95] 
#[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
OUTPUT_DIR  = 'results\\figures\\final'

def load_data(paths, sparsities):
    
    dfs = []

    for sparsity, path in zip(sparsities, paths):
        df = pd.read_csv(path)
        df = df.drop(columns='repeat').groupby('epoch').mean().reset_index()
        df['sparsity'] = sparsity
        dfs.append(df)
    
    return pd.concat(dfs).reset_index(drop=True)


def create_plots(df_1, df_2, sparsities, name_1, name_2, title='', savedir=None):
    
    epochs = 10

    fig_acc , axes_acc = plt.subplots(3, 4, figsize=(16, 12))
    fig_loss , axes_loss = plt.subplots(3, 4, figsize=(16, 12))

    title_acc = f'Accuracy: {title}'
    title_loss = f'Loss: {title}'

    st_acc = fig_acc.suptitle(title_acc, fontsize="large")
    st_loss = fig_loss.suptitle(title_loss, fontsize='large')

    

    lines_acc = []
    lines_loss = []

    for i, sparsity in enumerate(sparsities):
        row = i // 4
        col = i % 4

        x = list(range(epochs))
        y_sparse_train_acc = df_1.query(f'sparsity == {sparsity}')['train acc'].to_numpy()
        y_sparse_test_acc = df_1.query(f'sparsity == {sparsity}')['test acc'].to_numpy()
        y_sparse_train_loss = df_1.query(f'sparsity == {sparsity}')['train loss'].to_numpy()
        y_sparse_test_loss = df_1.query(f'sparsity == {sparsity}')['test loss'].to_numpy()

        y_conv_train_acc = df_2.query(f'sparsity == {sparsity}')['train acc'].to_numpy()
        y_conv_test_acc = df_2.query(f'sparsity == {sparsity}')['test acc'].to_numpy()
        y_conv_train_loss = df_2.query(f'sparsity == {sparsity}')['train loss'].to_numpy()
        y_conv_test_loss = df_2.query(f'sparsity == {sparsity}')['test loss'].to_numpy()

        l_sparse_train_acc, = axes_acc[row, col].plot(x, y_sparse_train_acc)
        l_sparse_test_acc, = axes_acc[row, col].plot(x, y_sparse_test_acc)
        l_conv_train_acc, = axes_acc[row, col].plot(x, y_conv_train_acc)
        l_conv_test_acc, = axes_acc[row, col].plot(x, y_conv_test_acc)
        axes_acc[row, col].set_title(f'Sparsity = {sparsity}', fontsize='small')

        l_sparse_train_loss, = axes_loss[row, col].plot(x, y_sparse_train_loss)
        l_sparse_test_loss, = axes_loss[row, col].plot(x, y_sparse_test_loss)
        l_conv_train_loss, = axes_loss[row, col].plot(x, y_conv_train_loss)
        l_conv_test_loss, = axes_loss[row, col].plot(x, y_conv_test_loss)
        axes_loss[row, col].set_title(f'Sparsity = {sparsity}', fontsize='small')

        if i == 0:
            lines_acc = [l_sparse_train_acc, l_sparse_test_acc, l_conv_train_acc, l_conv_test_acc]
            lines_loss = [l_sparse_train_loss, l_sparse_test_loss, l_conv_train_loss, l_conv_test_loss]

    axes_acc[2, 0].set_xlabel('epoch')
    axes_acc[2, 1].set_xlabel('epoch')  
    axes_acc[2, 2].set_xlabel('epoch')  
    axes_acc[2, 3].set_xlabel('epoch') 
    axes_acc[0, 0].set_ylabel('accuracy') 
    axes_acc[1, 0].set_ylabel('accuracy')
    axes_acc[2, 0].set_ylabel('accuracy')    

    axes_loss[2, 0].set_xlabel('epoch')
    axes_loss[2, 1].set_xlabel('epoch')  
    axes_loss[2, 2].set_xlabel('epoch')  
    axes_loss[2, 3].set_xlabel('epoch') 
    axes_loss[0, 0].set_ylabel('loss') 
    axes_loss[1, 0].set_ylabel('loss')
    axes_loss[2, 0].set_ylabel('loss')

    axes_acc.flat[-1].set_visible(False)
    axes_loss.flat[-1].set_visible(False)         
    
    labels = [f'{name_1} - Train', 
              f'{name_1} - Test', 
              f'{name_2} - Train', 
              f'{name_2} - Test']
    fig_acc.legend(lines_acc, labels, loc='lower right')
    fig_loss.legend(lines_loss, labels, loc='lower right')

    fig_acc.tight_layout()
    fig_loss.tight_layout()

    st_acc.set_y(0.975)
    fig_acc.subplots_adjust(top=0.9)
    st_loss.set_y(0.975)
    fig_loss.subplots_adjust(top=0.9)

    if savedir:
        
        filename_acc = title_acc.replace(':', '_').replace(',', '_').replace(' ', '_')
        filename_loss = title_loss.replace(':', '_').replace(',', '_').replace(' ', '_')

        path_acc = f'{savedir}\\{filename_acc}.png'
        path_loss = f'{savedir}\\{filename_loss}.png'

        fig_acc.savefig(path_acc)
        fig_loss.savefig(path_loss)

    else:
        plt.show()




# Load (1)
file_paths = ['results\\final\\SparseConvNetArchOriginalNormOriginal0.05_20240405181901.csv',
              'results\\final\\SparseConvNetArchOriginalNormOriginal0.1_20240405182506.csv',
              'results\\final\\SparseConvNetArchOriginalNormOriginal0.2_20240405183101.csv',
              'results\\final\\SparseConvNetArchOriginalNormOriginal0.3_20240405183708.csv',
              'results\\final\\SparseConvNetArchOriginalNormOriginal0.4_20240405184317.csv',
              'results\\final\\SparseConvNetArchOriginalNormOriginal0.5_20240405184921.csv',
              'results\\final\\SparseConvNetArchOriginalNormOriginal0.6_20240405185535.csv',
              'results\\final\\SparseConvNetArchOriginalNormOriginal0.7_20240405190156.csv',
              'results\\final\\SparseConvNetArchOriginalNormOriginal0.8_20240405190804.csv',
              'results\\final\\SparseConvNetArchOriginalNormOriginal0.9_20240405191419.csv',
              'results\\final\\SparseConvNetArchOriginalNormOriginal0.95_20240405192044.csv'
              ]

df_1 = load_data(file_paths, SPARSITIES)

# Load (2)
file_paths = ['results\\final\\ConvNetArchOriginal0.05_20240406105906.csv',
              'results\\final\\ConvNetArchOriginal0.1_20240406110520.csv',
              'results\\final\\ConvNetArchOriginal0.2_20240406111152.csv',
              'results\\final\\ConvNetArchOriginal0.3_20240406111810.csv',
              'results\\final\\ConvNetArchOriginal0.4_20240406112427.csv',
              'results\\final\\ConvNetArchOriginal0.5_20240406113008.csv',
              'results\\final\\ConvNetArchOriginal0.6_20240406113558.csv',
              'results\\final\\ConvNetArchOriginal0.7_20240406114144.csv',
              'results\\final\\ConvNetArchOriginal0.8_20240406114730.csv',
              'results\\final\\ConvNetArchOriginal0.9_20240406115302.csv',
              'results\\final\\ConvNetArchOriginal0.95_20240406115837.csv'
              ]

df_2 = load_data(file_paths, SPARSITIES)

# Load (3)
file_paths = ['results\\final\\SparseConvNetArchNewNormOriginal0.05_20240405195045.csv',
              'results\\final\\SparseConvNetArchNewNormOriginal0.1_20240405195825.csv',
              'results\\final\\SparseConvNetArchNewNormOriginal0.2_20240405200650.csv',
              'results\\final\\SparseConvNetArchNewNormOriginal0.3_20240405201425.csv',
              'results\\final\\SparseConvNetArchNewNormOriginal0.4_20240405202125.csv',
              'results\\final\\SparseConvNetArchNewNormOriginal0.5_20240405202830.csv',
              'results\\final\\SparseConvNetArchNewNormOriginal0.6_20240405203555.csv',
              'results\\final\\SparseConvNetArchNewNormOriginal0.7_20240405204321.csv',
              'results\\final\\SparseConvNetArchNewNormOriginal0.8_20240405205021.csv',
              'results\\final\\SparseConvNetArchNewNormOriginal0.9_20240405205908.csv',
              'results\\final\\SparseConvNetArchNewNormOriginal0.95_20240405210732.csv'
              ]

df_3 = load_data(file_paths, SPARSITIES)

# Load (4)
file_paths = ['results\\final\\ConvNetArchNew0.05_20240406011927.csv',
              'results\\final\\ConvNetArchNew0.1_20240406012601.csv',
              'results\\final\\ConvNetArchNew0.2_20240406013232.csv',
              'results\\final\\ConvNetArchNew0.3_20240406013902.csv',
              'results\\final\\ConvNetArchNew0.4_20240406014532.csv',
              'results\\final\\ConvNetArchNew0.5_20240406015202.csv',
              'results\\final\\ConvNetArchNew0.6_20240406015832.csv',
              'results\\final\\ConvNetArchNew0.7_20240406020502.csv',
              'results\\final\\ConvNetArchNew0.8_20240406021132.csv',
              'results\\final\\ConvNetArchNew0.9_20240406091125.csv',
              'results\\final\\ConvNetArchNew0.95_20240406091822.csv'
              ]

df_4 = load_data(file_paths, SPARSITIES)

# Load (5)
file_paths = ['results\\final\\SparseConvNetArchNewNormNone0.05_20240405234259.csv',
              'results\\final\\SparseConvNetArchNewNormNone0.1_20240405235135.csv',
              'results\\final\\SparseConvNetArchNewNormNone0.2_20240406000008.csv',
              'results\\final\\SparseConvNetArchNewNormNone0.3_20240406000857.csv',
              'results\\final\\SparseConvNetArchNewNormNone0.4_20240406001746.csv',
              'results\\final\\SparseConvNetArchNewNormNone0.5_20240406002634.csv',
              'results\\final\\SparseConvNetArchNewNormNone0.6_20240406003522.csv',
              'results\\final\\SparseConvNetArchNewNormNone0.7_20240406004411.csv',
              'results\\final\\SparseConvNetArchNewNormNone0.8_20240406005259.csv',
              'results\\final\\SparseConvNetArchNewNormNone0.9_20240406010148.csv',
              'results\\final\\SparseConvNetArchNewNormNone0.95_20240406011037.csv'
              ]

df_5 = load_data(file_paths, SPARSITIES)

# Load (6)
file_paths = ['results\\final\\SparseConvNetArchNewNormFix0.05_20240405211916.csv',
              'results\\final\\SparseConvNetArchNewNormFix0.1_20240405212753.csv',
              'results\\final\\SparseConvNetArchNewNormFix0.2_20240405213622.csv',
              'results\\final\\SparseConvNetArchNewNormFix0.3_20240405214453.csv',
              'results\\final\\SparseConvNetArchNewNormFix0.4_20240405215226.csv',
              'results\\final\\SparseConvNetArchNewNormFix0.5_20240405215929.csv',
              'results\\final\\SparseConvNetArchNewNormFix0.6_20240405220634.csv',
              'results\\final\\SparseConvNetArchNewNormFix0.7_20240405221337.csv',
              'results\\final\\SparseConvNetArchNewNormFix0.8_20240405222113.csv',
              'results\\final\\SparseConvNetArchNewNormFix0.9_20240405222945.csv',
              'results\\final\\SparseConvNetArchNewNormFix0.95_20240405223820.csv'
              ]

df_6 = load_data(file_paths, SPARSITIES)


create_plots(df_1, df_2, SPARSITIES, 
             'SparseConvNet',
             'ConvNet',
             'Original Architecture, Original Normalization',
             OUTPUT_DIR)

create_plots(df_3, df_4, SPARSITIES, 
             'SparseConvNet',
             'ConvNet',
             'New Architecture, Original Normalization',
             OUTPUT_DIR)

create_plots(df_5, df_4, SPARSITIES, 
             'SparseConvNet',
             'ConvNet',
             'New Architecture, No Normalization',
             OUTPUT_DIR)

create_plots(df_6, df_4, SPARSITIES, 
             'SparseConvNet',
             'ConvNet',
             'New Architecture, Fix Normalization',
             OUTPUT_DIR)



"""
###########################################################
# Original Architecture and Original Normalization method
###########################################################

# in order from 0.05 to 0.7
file_paths = ['results\\SparseConvNetOrig0.05_20240330153703.csv',
              'results\\SparseConvNetOrig0.1_20240330160925.csv',
              'results\\SparseConvNetOrig0.2_20240330163804.csv',
              'results\\SparseConvNetOrig0.3_20240330164424.csv',
              'results\\SparseConvNetOrig0.4_20240330165537.csv',
              'results\\SparseConvNetOrig0.5_20240330170150.csv',
              'results\\SparseConvNetOrig0.6_20240330170801.csv',
              'results\\SparseConvNetOrig0.7_20240330172420.csv'
              ]


df_1 = load_data(file_paths, SPARSITIES)
# print(df_sparseconvnet)

file_paths = ['results\\ConvNetOGArch0.05_20240331202836.csv',
              'results\\ConvNetOGArch0.1_20240331203330.csv',
              'results\\ConvNetOGArch0.2_20240331203816.csv',
              'results\\ConvNetOGArch0.3_20240331204335.csv',
              'results\\ConvNetOGArch0.4_20240331204854.csv',
              'results\\ConvNetOGArch0.5_20240331205422.csv',
              'results\\ConvNetOGArch0.6_20240331205954.csv',
              'results\\ConvNetOGArch0.7_20240331210527.csv'
              ]

df_2 = load_data(file_paths, SPARSITIES)

create_plots(df_1, df_2, SPARSITIES, 
             'SparseConvNet',
             'ConvNet',
             'Original Architecture, Original Normalization',
             OUTPUT_DIR)


###########################################################
# Original Architecture and New Normalization method
###########################################################

#import SpareConvNet experiments

# in order from 0.05 to 0.7
file_paths = ['results\\SparseConvNetFix0.05_20240330153734.csv',
              'results\\SparseConvNetFix0.1_20240330162145.csv',
              'results\\SparseConvNetFix0.2_20240330165621.csv',
              'results\\SparseConvNetFix0.3_20240330170220.csv',
              'results\\SparseConvNetFix0.4_20240330171359.csv',
              'results\\SparseConvNetFix0.5_20240330173059.csv',
              'results\\SparseConvNetFix0.6_20240330173925.csv',
              'results\\SparseConvNetFix0.7_20240331120720.csv'
              ]


df_1 = load_data(file_paths, SPARSITIES)
# print(df_sparseconvnet)

file_paths = ['results\\SparseConvNetOrig0.05_20240330153703.csv',
              'results\\SparseConvNetOrig0.1_20240330160925.csv',
              'results\\SparseConvNetOrig0.2_20240330163804.csv',
              'results\\SparseConvNetOrig0.3_20240330164424.csv',
              'results\\SparseConvNetOrig0.4_20240330165537.csv',
              'results\\SparseConvNetOrig0.5_20240330170150.csv',
              'results\\SparseConvNetOrig0.6_20240330170801.csv',
              'results\\SparseConvNetOrig0.7_20240330172420.csv'
              ]

df_2 = load_data(file_paths, SPARSITIES)

create_plots(df_1, df_2, SPARSITIES, 
             'SparseConvNet (new norm.)',
             'SparseConvNet (orig.)',
             'Original Architecture, New Normalization',
             OUTPUT_DIR)


###########################################################
# Original Architecture vs New Architecture
###########################################################

#import SpareConvNet experiments

# in order from 0.05 to 0.7
file_paths = ['results\\ConvNetOGArch0.05_20240331202836.csv',
              'results\\ConvNetOGArch0.1_20240331203330.csv',
              'results\\ConvNetOGArch0.2_20240331203816.csv',
              'results\\ConvNetOGArch0.3_20240331204335.csv',
              'results\\ConvNetOGArch0.4_20240331204854.csv',
              'results\\ConvNetOGArch0.5_20240331205422.csv',
              'results\\ConvNetOGArch0.6_20240331205954.csv',
              'results\\ConvNetOGArch0.7_20240331210527.csv'
              ]


df_1 = load_data(file_paths, SPARSITIES)
# print(df_sparseconvnet)

file_paths = ['results\\ConvNetNewArch0.05_20240331171621.csv',
              'results\\ConvNetNewArch0.1_20240331172259.csv',
              'results\\ConvNetNewArch0.2_20240331172936.csv',
              'results\\ConvNetNewArch0.3_20240331173612.csv',
              'results\\ConvNetNewArch0.4_20240331174324.csv',
              'results\\ConvNetNewArch0.5_20240331175107.csv',
              'results\\ConvNetNewArch0.6_20240331192451.csv',
              'results\\ConvNetNewArch0.7_20240331193248.csv'
              ]

df_2 = load_data(file_paths, SPARSITIES)

create_plots(df_1, df_2, SPARSITIES, 
             'ConvNet (orig.)',
             'ConvNet (new arch.)',
             'New ConvNet Architecture',
             OUTPUT_DIR)

###########################################################
# New Architecture, New Normalization
###########################################################

#import SpareConvNet experiments

# in order from 0.05 to 0.7
file_paths = ['results\SparseConvNetFixNewArch0.05_20240331141154.csv',
              'results\SparseConvNetFixNewArch0.1_20240331141953.csv',
              'results\SparseConvNetFixNewArch0.2_20240331142754.csv',
              'results\SparseConvNetFixNewArch0.3_20240331143554.csv',
              'results\SparseConvNetFixNewArch0.4_20240331144356.csv',
              'results\SparseConvNetFixNewArch0.5_20240331145157.csv',
              'results\SparseConvNetFixNewArch0.6_20240331145959.csv',
              'results\SparseConvNetFixNewArch0.7_20240331150800.csv'
              ]


df_1 = load_data(file_paths, SPARSITIES)
# print(df_sparseconvnet)

file_paths = ['results\\ConvNetNewArch0.05_20240331171621.csv',
              'results\\ConvNetNewArch0.1_20240331172259.csv',
              'results\\ConvNetNewArch0.2_20240331172936.csv',
              'results\\ConvNetNewArch0.3_20240331173612.csv',
              'results\\ConvNetNewArch0.4_20240331174324.csv',
              'results\\ConvNetNewArch0.5_20240331175107.csv',
              'results\\ConvNetNewArch0.6_20240331192451.csv',
              'results\\ConvNetNewArch0.7_20240331193248.csv'
              ]

df_2 = load_data(file_paths, SPARSITIES)

create_plots(df_1, df_2, SPARSITIES, 
             'SparseConvNet',
             'ConvNet',
             'New Architecture, New Normalization',
             OUTPUT_DIR)



###########################################################
# Old Architecture, No Normalization vs New Normalization
###########################################################

#import SpareConvNet experiments

# in order from 0.05 to 0.7
file_paths = ['results\\SparseConvNetFix0.05_20240330153734.csv',
              'results\\SparseConvNetFix0.1_20240330162145.csv',
              'results\\SparseConvNetFix0.2_20240330165621.csv',
              'results\\SparseConvNetFix0.3_20240330170220.csv',
              'results\\SparseConvNetFix0.4_20240330171359.csv',
              'results\\SparseConvNetFix0.5_20240330173059.csv',
              'results\\SparseConvNetFix0.6_20240330173925.csv',
              'results\\SparseConvNetFix0.7_20240331120720.csv'
              ]


df_1 = load_data(file_paths, SPARSITIES)
# print(df_sparseconvnet)

file_paths = ['results\\SparseConvNetNoNormOrigArch0.05_20240403172539.csv',
              'results\\SparseConvNetNoNormOrigArch0.1_20240403173702.csv',
              'results\\SparseConvNetNoNormOrigArch0.2_20240403174818.csv',
              'results\\SparseConvNetNoNormOrigArch0.3_20240403175936.csv',
              'results\\SparseConvNetNoNormOrigArch0.4_20240403181052.csv',
              'results\\SparseConvNetNoNormOrigArch0.5_20240403182216.csv',
              'results\\SparseConvNetNoNormOrigArch0.6_20240403183339.csv',
              'results\\SparseConvNetNoNormOrigArch0.7_20240403184459.csv'
              ]

df_2 = load_data(file_paths, SPARSITIES)

create_plots(df_1, df_2, SPARSITIES, 
             'SparseConvNet (new norm.)',
             'SparseConvNet (no norm.)',
             'Original Architecture, New Normalization vs No Normalization',
             OUTPUT_DIR)

"""