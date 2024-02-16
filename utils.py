'''
Author: dkl
Date: 2024-02-12 09:32:32
Description: 工具性函数
'''


def get_nearby_date(date, datelst, flag_latest=True):
    """
    返回在日期列表datelst中，距离指定日期date最近的日期.
    若date在datelst中，则历史最近的日期为date的上个日期，未来最近的日期为date

    Parameters
    ----------
    date : str
        指定日期
    datelst : list
        日期列表
    flag_latest : bool
        当date在datelst的两个元素之间时，
        若为True, 最近日期为date的历史相邻日期,
        若为False, 最近日期为未来相邻日期

    Returns
    -------
    str
        距离指定日期date最近的日期
    """
    datelst = sorted(list(set(datelst)))
    if date in datelst:
        return date
    elif date < datelst[0]:
        return datelst[0]
    elif date > datelst[-1]:
        return datelst[-1]
    # 以下为date在datelst两个相邻元素之间的情况
    for idx in range(len(datelst)-1):
        if (datelst[idx] < date) and (date < datelst[idx+1]):
            # 历史最近
            if flag_latest:
                return datelst[idx]
            # 未来最近
            else:
                return datelst[idx+1]


def stackdf(df, var_name, date_name='trade_date', code_name='stock_code'):
    '''
    Description
    ----------
    对输入数据进行堆栈，每行为截面数据，每列为时间序列数据

    Parameters
    ----------
    df: pandas.DataFrame.
        输出数据为堆栈后的数据
    date_name: str. 日期名称, 默认为trade_date
    code_name: str. 代码名称, 默认为stock_code

    Return
    ----------
    pandas.DataFrame.
    堆栈后的数据,列为trade_date, stock_code和var_name
    '''
    df = df.copy()
    df = df.stack().reset_index()
    df.columns = [date_name, code_name, var_name]
    return df


def unstackdf(df, date_name='trade_date', code_name='stock_code'):
    '''
    Description
    ----------
    反堆栈函数

    Parameters
    ----------
    df: pandas.DataFrame.
        输入列必须为三列且必须有date_name和code_name
    date_name: str. 日期名称, 默认为trade_date
    code_name: str. 代码名称, 默认为stock_code

    Return
    ----------
    pandas.DataFrame. 反堆栈后的数据
    '''
    _check_sub_columns(df, [date_name, code_name])
    if not (len(df.columns) == 3):
        error_message = 'length of df.columns must be 3'
        raise ValueError(error_message)
    df = df.copy()
    df = df.set_index([date_name, code_name]).unstack()
    df.columns = df.columns.get_level_values(1).tolist()
    df.index = df.index.tolist()
    return df


# 检查df的列的部分
def _check_sub_columns(df, var_lst):
    '''
    Description
    ----------
    检查var_lst是否是df.columns的列的子集（不考虑排序）

    Parameters
    ----------
    df: pandas.DataFrame. 输入数据
    var_lst: List[str]. 变量名列表

    Return
    ----------
    Bool
    '''
    if not set(var_lst).issubset(df.columns):
        var_name = ','.join(var_lst)
        raise ValueError(f'{var_name} must be in the columns of df.')


def get_resample_ret(ret, trade_date_lst):
    """
    以指定的日期计算对应的股票收益率. 如果期间股票没有交易则设定为0

    Parameters
    ----------
    ret : pandas.DataFrame.
        较高频率(如日频)的股票收益率，列名为trade_date, stock_code, ret
    trade_date_lst : List
        交易日期列表

    Returns
    -------
    pandas.DataFrame.
        从上个交易日期到本交易日期下的股票收益率
    """
    unstackret = unstackdf(ret)
    unstackret = unstackret.fillna(0)
    price = (unstackret+1).cumprod()
    price=price.loc[trade_date_lst,:].copy()
    res=price/price.shift(1)-1
    res=res.fillna(0)
    res=stackdf(res, 'ret')
    return res


def check_columns(df, var_lst):
    '''
    Description
    ----------
    检查var_lst是否是df.columns的列（不考虑顺序）

    Parameters
    ----------
    df: pandas.DataFrame. 输入数据
    var_lst: List[str]. 变量名列表

    Return
    ----------
    Bool
    '''
    lst1 = list(var_lst)
    lst2 = df.columns.tolist()
    if not sorted(lst1) == sorted(lst2):
        var_str = ', '.join(var_lst)
        err = 'The columns of df must be var_lst:{}'.format(var_str)
        raise ValueError(err)