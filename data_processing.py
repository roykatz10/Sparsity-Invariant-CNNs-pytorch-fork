import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


SPARSITIES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

def load_data(paths, sparsities):
    
    dfs = []

    for sparsity, path in zip(sparsities, paths):
        df = pd.read_csv(path)
        df = df.drop(columns='repeat').groupby('epoch').mean().reset_index()
        df['sparsity'] = sparsity
        dfs.append(df)
    
    return pd.concat(dfs).reset_index(drop=True)


def create_plots(df_sparse, df_conv, sparsities, title=''):
    
    epochs = 10

    fig_acc , axes_acc = plt.subplots(2, 4, figsize=(10, 5))
    fig_loss , axes_loss = plt.subplots(2, 4, figsize=(10, 5))

    title_acc = f'Accuracy: {title}'
    title_loss = f'Loss: {title}'

    st_acc = fig_acc.suptitle(title_acc, fontsize="large")
    st_loss = fig_loss.suptitle(title_loss, fontsize='large')

    st_acc.set_y(0.95)
    fig_acc.subplots_adjust(top=0.85)

    st_loss.set_y(0.95)
    fig_loss.subplots_adjust(top=0.85)

    lines_acc = []
    lines_loss = []

    for i, sparsity in enumerate(sparsities):
        row = i // 4
        col = i % 4

        x = list(range(epochs))
        y_sparse_train_acc = df_sparse.query(f'sparsity == {sparsity}')['train acc'].to_numpy()
        y_sparse_test_acc = df_sparse.query(f'sparsity == {sparsity}')['test acc'].to_numpy()
        y_sparse_train_loss = df_sparse.query(f'sparsity == {sparsity}')['train loss'].to_numpy()
        y_sparse_test_loss = df_sparse.query(f'sparsity == {sparsity}')['test loss'].to_numpy()

        y_conv_train_acc = df_conv.query(f'sparsity == {sparsity}')['train acc'].to_numpy()
        y_conv_test_acc = df_conv.query(f'sparsity == {sparsity}')['test acc'].to_numpy()
        y_conv_train_loss = df_conv.query(f'sparsity == {sparsity}')['train loss'].to_numpy()
        y_conv_test_loss = df_conv.query(f'sparsity == {sparsity}')['test loss'].to_numpy()

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

    axes_acc[1, 0].set_xlabel('epoch')
    axes_acc[1, 1].set_xlabel('epoch')  
    axes_acc[1, 2].set_xlabel('epoch')  
    axes_acc[1, 3].set_xlabel('epoch') 
    axes_acc[0, 0].set_ylabel('accuracy') 
    axes_acc[1, 0].set_ylabel('accuracy')   

    axes_loss[1, 0].set_xlabel('epoch')
    axes_loss[1, 1].set_xlabel('epoch')  
    axes_loss[1, 2].set_xlabel('epoch')  
    axes_loss[1, 3].set_xlabel('epoch') 
    axes_loss[0, 0].set_ylabel('loss') 
    axes_loss[1, 0].set_ylabel('loss')         
    
    labels = ['SparseConvNet - Train', 'SparseConvNet - Test', 'ConvNet - Train', 'ConvNet - Test']
    fig_acc.legend(lines_acc, labels, loc='upper right', ncols=2)
    fig_loss.legend(lines_loss, labels, loc='upper right', ncol=2)

    fig_acc.tight_layout()
    fig_loss.tight_layout()

    plt.show()

###########################################################
# Original Architecture and Original Normalization method
###########################################################

#import SpareConvNet experiments

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


df_sparseconvnet = load_data(file_paths, SPARSITIES)
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

df_convnet = load_data(file_paths, SPARSITIES)
# print(df_convnet)
# a = 0.05
# print(df_convnet.query(f'sparsity == {a}')['train acc'].to_numpy())


create_plots(df_sparseconvnet, df_convnet, SPARSITIES, 
             'Original Architecture, Original Normalization')



