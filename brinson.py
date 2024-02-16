'''
Author: dkl
Date: 2024-02-12 09:16:26
Description: brinson绩效归因框架
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils


class Brinson(object):
    def __init__(self, single_method='BF', multi_method='GRAP'):
        """
        Brinson绩效归因框架

        Parameters
        ----------
        single_method : str, optional
            单期归因方法，BF或者BHB, by default 'BF'
        multi_method : str, optional
            多期归因方法, by default 'GRAP'
        """
        # Brinson归因需要的数据
        self.data = None
        # Brinson归因方法
        if single_method not in ['BF', 'BHB']:
            raise ValueError('single_method must be BF or BHB.')
        if multi_method not in ['GRAP']:
            raise ValueError('multi_method must be GRAP.')
        self.single_method = single_method
        self.multi_method = multi_method
        self.trade_date_lst = None
        self._calc_data_flag = False

    def load_data(self, data):
        columns = ['trade_date', 'industry', 'wp', 'rp', 'wb', 'rb']
        utils.check_columns(data, columns)
        self.trade_date_lst = set(sorted(data['trade_date'].tolist()))
        self.data = data[columns].copy()

    def _calc_data(self):
        if self._calc_data_flag:
            return
        if self.data is None:
            raise ValueError('self.data is None.')
        # 计算Brinson绩效归因需要的SR, ER
        wp = self.data['wp']
        wb = self.data['wb']
        rp = self.data['rp']
        rb = self.data['rb']
        self.data['er'] = wp * rp - wb * rb
        if self.single_method == 'BF':
            # 配置收益
            market_ret = sum(wb*rb)
            self.data['ar'] = (wp-wb)*(rb-market_ret)
            # 选择收益
            self.data['sr'] = wp*(rp-rb)
            self._calc_data_flag = True
            return
        elif self.single_method == 'BHB':
            # 配置收益
            self.data['ar'] = (wp-wb)*rb
            # 选择收益
            self.data['sr'] = wb*(rp-rb)
            # 交互收益
            self.data['ir'] = (wp-wb) * (rp-rb)
            self._calc_data_flag = True
            return

    def brinson_single_term(self, trade_date):
        self._calc_data()
        trade_date = utils.get_nearby_date(trade_date, self.trade_date_lst)
        df = self.data.copy()
        if self.single_method == 'BF':
            col_lst = ['trade_date', 'industry', 'er', 'ar', 'sr']
        elif self.single_method == 'BHB':
            col_lst = ['trade_date', 'industry', 'er', 'ar', 'sr', 'ir']
        cond1 = (df['trade_date'] == trade_date)
        # 去除收益率为0的数据
        cond2 = (abs(df['rp']) > 1e-10) & (abs(df['rb']) > 1e-10)
        df = df.loc[cond1 & cond2, col_lst].copy()
        df = df.reset_index(drop=True)
        fig = self.plot_single_term(df)
        return df, fig

    def plot_single_term(self, df):
        date = df.loc[0, 'trade_date']
        labels = df['industry'].tolist()
        x = np.arange(len(labels))
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(f'{date}单期归因结果')
        if self.single_method == 'BF':
            width = 0.35
            offset = 0.2
            ax.bar(x - offset, df['ar'], width, label='配置收益')
            ax.bar(x + offset, df['sr'], width, label='选择收益')
        elif self.single_method == 'BHB':
            width = 0.2
            offset = 0.22
            ax.bar(x - offset, df['ar'], width, label='配置收益')
            ax.bar(x, df['sr'], width, label='选择收益')
            ax.bar(x + offset, df['sr'], width, label='交互收益')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend(loc=2)
        return fig

    def brinson_multi_term(self, start_date, end_date):
        self._calc_data()
        start_date = utils.get_nearby_date(start_date, self.trade_date_lst, flag_latest=True)
        end_date = utils.get_nearby_date(end_date, self.trade_date_lst, flag_latest=False)
        df = self.data.copy()
        df['wp_rp'] = df['wp']*df['rp']
        df['wb_rb'] = df['wb']*df['rb']
        if self.single_method == 'BF':
            ret_name_lst = ['er', 'ar', 'sr']
        elif self.single_method == 'BHB':
            ret_name_lst = ['er', 'ar', 'sr', 'ir']
        col_lst = ['trade_date', 'industry', 'wp_rp', 'wb_rb'] + ret_name_lst
        cond1 = (df['trade_date'] >= start_date)
        cond2 = (df['trade_date'] <= end_date)
        df = df.loc[cond1 & cond2, col_lst].copy()
        # 聚合计算总的ER, AR, SR, IR
        groupby_lst = ['wp_rp', 'wb_rb']+ret_name_lst
        df = df.groupby('trade_date')[groupby_lst].sum()
        df = df.reset_index()
        df.columns = ['trade_date', 'str_ret', 'ben_ret'] + ret_name_lst
        for ret_name in ret_name_lst:
            arr = df[ret_name].values.reshape(-1)
            str_arr = df['str_ret'].values.reshape(-1)
            ben_arr = df['ben_ret'].values.reshape(-1)
            df['adj_'+ret_name] = self.grap_adjust(arr, str_arr, ben_arr)
        fig = self.plot_multi_term(df)
        return df, fig

    def grap_adjust(self, arr, str_arr, ben_arr):
        """
        arr: 原来的指标
        str_arr: 策略收益率
        ben_arr:基准收益率
        """
        arr = arr.reshape(-1)
        str_arr = str_arr.reshape(-1)
        ben_arr = ben_arr.reshape(-1)
        adj_arr = np.zeros_like(arr)
        n = arr.shape[0]
        for t in range(n):
            # 计算之前的策略净值
            if t == 0:
                str_factor = 1
            else:
                str_factor = np.prod(1 + str_arr[0:t])
            # 计算之后基准指数的再投资收益
            if t + 1 == n:
                ben_factor = 1
            else:
                ben_factor = np.prod(1 + ben_arr[t+1:n])
            adj_arr[t] = arr[t] * str_factor * ben_factor
        return adj_arr

    def plot_multi_term(self, df):
        trade_date_lst = sorted(list(set(df['trade_date'])))
        start_date = trade_date_lst[0]
        end_date = trade_date_lst[-1]
        x = [pd.to_datetime(i) for i in trade_date_lst]
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(f'{start_date}-{end_date}多期归因结果')
        if self.single_method == 'BF':
            ax = fig.add_subplot(3, 1, 1)
            ax.set_title(f'超额收益累加')
            ax.bar(x, df['adj_er'].cumsum(), width=8, fc='blue')
            ax.set_xticklabels([])
            ax = fig.add_subplot(3, 1, 2)
            ax.set_title(f'配置收益累加')
            ax.bar(x, df['adj_ar'].cumsum(), width=8, fc='green')
            ax.set_xticklabels([])
            ax = fig.add_subplot(3, 1, 3)
            ax.set_title(f'选择收益累加')
            ax.bar(x, df['adj_sr'].cumsum(), width=8, fc='red')
        elif self.single_method == 'BHB':
            ax = fig.add_subplot(4, 1, 1)
            ax.set_title(f'超额收益累加')
            ax.bar(x, df['adj_er'].cumsum(), width=8, fc='blue')
            ax.set_xticklabels([])
            ax = fig.add_subplot(4, 1, 2)
            ax.set_title(f'配置收益累加')
            ax.bar(x, df['adj_ar'].cumsum(), width=8, fc='green')
            ax.set_xticklabels([])
            ax = fig.add_subplot(4, 1, 3)
            ax.set_title(f'选择收益累加')
            ax.bar(x, df['adj_sr'].cumsum(), width=8, fc='red')
            ax.set_xticklabels([])
            ax = fig.add_subplot(4, 1, 4)
            ax.set_title(f'交互收益累加')
            ax.bar(x, df['adj_ir'].cumsum(), width=8, fc='grey')
        return fig


if __name__ == '__main__':
    brinson_data = pd.read_csv('brinson_data.csv')
    brinson_data['trade_date'] = brinson_data['trade_date'].apply(str)
    brinson = Brinson(single_method='BF')
    brinson.load_data(brinson_data)
    single_df, fig = brinson.brinson_single_term(trade_date='20050501')
    multi_df, _ = brinson.brinson_multi_term('20140601', '20251031')
    multi_df = multi_df[['trade_date','adj_er','adj_ar','adj_sr']].copy()
    print(multi_df)
    plt.show()
