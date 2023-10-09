import pandas as pd
import qstock as qs
from Neutralization.neutralization import Industry
from dateutil.parser import parse

def cvt_date(date_str):
    date = parse(date_str)
    formatted_date = date.strftime("%Y-%m-%d")
    return formatted_date

class Hist_data:
    def __init__(self, stock_list, start='20100101', end='20200101'):
        self.start = cvt_date(start)
        self.end = cvt_date(end)
        self.stock_list = stock_list

    def simple_close(self):
        df = qs.get_price(self.stock_list)
        # return df[self.start, self.end]
        # print(df.index)
        return df[self.start: self.end]

    def get_index(self, type='sh'):
        code_list = ['sh', 'sz', 'cyb', 'zxb', 'hs300', 'sz50', 'zz500']
        assert type in code_list
        df = qs.get_data(type)
        return df[self.start: self.end]
if __name__ == '__main__':

    i = Industry()
    name = i.get_industries()[1]
    lst = i.specified(name)
    lst = [str_[2:] for str_ in lst]
    print()
    # quit()
    h = Hist_data(stock_list=lst)
    print(h.simple_close())
    print(h.get_index())

