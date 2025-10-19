import numpy as np
import pandas as pd


def uniform_bin(x, num_bins, alpha=0, mini=None, maxi=None):
    """
    将连续型数据 x 分箱成分类型数据
    输出范围为 1, 2, ..., num_bins
    """
    mini = x.min() if mini==None else mini
    maxi = x.max() if maxi==None else maxi
    # print(mini,maxi)

    bins = [mini]
    bin_width = (maxi - mini) / num_bins

    while len(bins) < num_bins:
        new_bin_edge = bins[-1] + bin_width
        
        # 如果新的边界值大于最大值，则停止添加边界
        if new_bin_edge > maxi:
            break
        # 查找下一个边界值使得该箱子至少包含一个数据点
        while np.sum((x >= bins[-1]) & (x < new_bin_edge)) == 0:
            new_bin_edge += bin_width
            if abs(new_bin_edge - maxi)<1e-5:
                break
        # 添加边界值
        if abs(new_bin_edge - maxi)<1e-5:
            bins.append(maxi)
        else: 
            bins.append(new_bin_edge)
        
    if maxi not in bins:
        bins.append(maxi)  # 添加最后一个边界点
    
    bins = np.array(bins)
    # print(bins)

    # 分配数据到箱子中，并使用左端点作为 bin_values
    bin_values = np.zeros(len(x))
    for i in range(len(bins) - 1):
        bin_values[(x >= bins[i]) & (x < bins[i+1])] = (bins[i]+bins[i+1])/2
     # 处理最后一个区间
    bin_values[x == maxi] = (bins[-2]+bins[-1])/2
    return bin_values,bins



def quantile_bin(x,num_bins,eps=1e-5):
    '''
    Bin a comtinous variable by sample quantiles, such that each bin has equal sample points
    Output range: 1,2,...,num_bins
    note: numpy.digitze bin[i-1]<=x<bin[i], we enlarge the last bin a little to include x.max() to the *num_bins*-th bin
    '''
    _, bins = pd.qcut(x,num_bins,retbins=True,duplicates='drop')
    bins[-1] += eps
    tx = np.digitize(x,bins)
    return tx