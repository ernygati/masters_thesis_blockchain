import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(columns = ['Date', 'Info'])

print('\nВас приветствует менеджер дедлайнов!\n')
print('\n ПРОВЕРИТЬ или ДОБАВИТЬ дедлайн?\n')

while True:
    try:
        check_or_add = str(input()).strip()
        if check_or_add == 'ДОБАВИТЬ' or check_or_add == 'добавить':
            if_all = ' '
            while if_all == ' ':
                print('\nПожалуйста, введите дату дедлайна в формате дд/мм/год:\n')
                date = str(input())
                print('\nПожалуйста, введите информацию о дедлайне:\n')
                info = str(input())
                df.loc[len(df1.index)] = [date, info]
                df.Date = pd.to_datetime(df.Date)
                print('\nНажмите ПРОБЕЛ, если нужно добавить дедлайн, \n или введите нет:\n')
                if_all = input()
                print(type(if_all))
        else:
            raise ValueError
    except Exception:
        print('Введи ПРОВЕРИТЬ или ДОБАВИТЬ')