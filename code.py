import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
import seaborn as sns


df = pd.read_csv('tmp.csv')
df = df[536:]
df = df.drop_duplicates('timestamp', keep='first')
df = df.reset_index(drop = True)

df['b_a_spread'] = df['BINANCE_BTC-USDT_ask_1'] - df['BINANCE_BTC-USDT_bid_1']
df = df.loc[df['b_a_spread']>0]


Time = df['timestamp']
Time = np.array(Time)
#Date_Time = Time
#Date_Time = pd.to_numeric(Date_Time, errors='coerce')
#Date_Time = pd.to_datetime(Date_Time, unit = 'ms')

#------------------------------------------------------------------------------


df['b12'] = df['BINANCE_BTC-USDT_bid_1'] - df['BINANCE_BTC-USDT_bid_2']
df['b13'] = df['BINANCE_BTC-USDT_bid_1'] - df['BINANCE_BTC-USDT_bid_3']
df['b14'] = df['BINANCE_BTC-USDT_bid_1'] - df['BINANCE_BTC-USDT_bid_4']

df['aten1'] = df['BINANCE_BTC-USDT_ask_1'] - df['BINANCE_BTC-USDT_ask_2']
df['aten2'] = df['BINANCE_BTC-USDT_ask_1'] - df['BINANCE_BTC-USDT_ask_3']
df['aten3'] = df['BINANCE_BTC-USDT_ask_1'] - df['BINANCE_BTC-USDT_ask_4']

Total_P_ask = df['BINANCE_BTC-USDT_ask_1']+df['BINANCE_BTC-USDT_ask_10']+df['BINANCE_BTC-USDT_ask_2']+df['BINANCE_BTC-USDT_ask_3']+df['BINANCE_BTC-USDT_ask_4']+df['BINANCE_BTC-USDT_ask_5']+df['BINANCE_BTC-USDT_ask_6']+df['BINANCE_BTC-USDT_ask_7']+df['BINANCE_BTC-USDT_ask_8']+df['BINANCE_BTC-USDT_ask_9']
Total_P_bid = df['BINANCE_BTC-USDT_bid_1']+df['BINANCE_BTC-USDT_bid_10']+df['BINANCE_BTC-USDT_bid_2']+df['BINANCE_BTC-USDT_bid_3']+df['BINANCE_BTC-USDT_bid_4']+df['BINANCE_BTC-USDT_bid_5']+df['BINANCE_BTC-USDT_bid_6']+df['BINANCE_BTC-USDT_bid_7']+df['BINANCE_BTC-USDT_bid_8']+df['BINANCE_BTC-USDT_bid_9']

Total_V_ask = df['BINANCE_BTC-USDT_askq_1']+df['BINANCE_BTC-USDT_askq_10']+df['BINANCE_BTC-USDT_askq_2']+df['BINANCE_BTC-USDT_askq_3']+df['BINANCE_BTC-USDT_askq_4']+df['BINANCE_BTC-USDT_askq_5']+df['BINANCE_BTC-USDT_askq_6']+df['BINANCE_BTC-USDT_askq_7']+df['BINANCE_BTC-USDT_askq_8']+df['BINANCE_BTC-USDT_askq_9']
Total_V_bid = df['BINANCE_BTC-USDT_bidq_1']+df['BINANCE_BTC-USDT_bidq_10']+df['BINANCE_BTC-USDT_bidq_2']+df['BINANCE_BTC-USDT_bidq_3']+df['BINANCE_BTC-USDT_bidq_4']+df['BINANCE_BTC-USDT_bidq_5']+df['BINANCE_BTC-USDT_bidq_6']+df['BINANCE_BTC-USDT_bidq_7']+df['BINANCE_BTC-USDT_bidq_8']+df['BINANCE_BTC-USDT_bidq_9']

df['mid_p_Y'] = (df['BINANCE_BTC-USDT_ask_1'] + df['BINANCE_BTC-USDT_bid_1']) / 2

df['acc_diff_P'] = Total_P_ask - Total_P_bid
df['acc_diff_v'] = Total_V_ask - Total_V_bid
df['mean_P_ask'] = Total_P_ask / 10
df['mean_P_bid'] = Total_P_bid / 10
df['mean_V_ask'] = Total_V_ask / 10
df['mean_V_bid'] = Total_V_bid / 10

#------------------------------------------------------------------------------

Past_ticks = 1000
Fut_ticks = 100

# Return calculated w.r.t. future values
df['mid_P_future'] = df['mid_p_Y'].shift(-Fut_ticks)
df = df[:-Fut_ticks]
df['Ret'] = (df['mid_P_future'] / df['mid_p_Y']) - 1
df['Ret'] = df['Ret'] * 10000

# Slope calculated w.r.t. past values
df['mid_P_past'] = df['mid_p_Y'].shift(Past_ticks)
df['BINANCE_BTC-USDT_bid_1_past'] = df['BINANCE_BTC-USDT_bid_1'].shift(Past_ticks)
df['BINANCE_BTC-USDT_ask_1_past'] = df['BINANCE_BTC-USDT_ask_1'].shift(Past_ticks)
df['timestamp_past'] = df['timestamp'].shift(Past_ticks)
df = df[Past_ticks:]
df = df.reset_index(drop=True)
df['time_gap'] = df['timestamp'] - df['timestamp_past'] 
df['BINANCE_BTC-USDT_bid_1_slope'] = (df['BINANCE_BTC-USDT_bid_1'] - df['BINANCE_BTC-USDT_bid_1_past']) / df['time_gap']
df['BINANCE_BTC-USDT_ask_1_slope'] = (df['BINANCE_BTC-USDT_ask_1'] - df['BINANCE_BTC-USDT_ask_1_past']) / df['time_gap']
df['mid_P_slope'] = (df['mid_p_Y'] - df['mid_P_past']) / df['time_gap']


df_ask = df[['BINANCE_BTC-USDT_ask_1', 'BINANCE_BTC-USDT_ask_2', 'BINANCE_BTC-USDT_ask_3', 'BINANCE_BTC-USDT_ask_4', 'BINANCE_BTC-USDT_ask_5', 'BINANCE_BTC-USDT_ask_6', 'BINANCE_BTC-USDT_ask_7', 'BINANCE_BTC-USDT_ask_8', 'BINANCE_BTC-USDT_ask_9', 'BINANCE_BTC-USDT_ask_10']].values
df_bid = df[['BINANCE_BTC-USDT_bid_1', 'BINANCE_BTC-USDT_bid_2', 'BINANCE_BTC-USDT_bid_3', 'BINANCE_BTC-USDT_bid_4', 'BINANCE_BTC-USDT_bid_5', 'BINANCE_BTC-USDT_bid_6', 'BINANCE_BTC-USDT_bid_7', 'BINANCE_BTC-USDT_bid_8', 'BINANCE_BTC-USDT_bid_9', 'BINANCE_BTC-USDT_bid_10']].values
df_askq = df[['BINANCE_BTC-USDT_askq_1', 'BINANCE_BTC-USDT_askq_2', 'BINANCE_BTC-USDT_askq_3', 'BINANCE_BTC-USDT_askq_4', 'BINANCE_BTC-USDT_askq_5', 'BINANCE_BTC-USDT_askq_6', 'BINANCE_BTC-USDT_askq_7', 'BINANCE_BTC-USDT_askq_8', 'BINANCE_BTC-USDT_askq_9', 'BINANCE_BTC-USDT_askq_10']].values
df_bidq = df[['BINANCE_BTC-USDT_bidq_1', 'BINANCE_BTC-USDT_bidq_2', 'BINANCE_BTC-USDT_bidq_3', 'BINANCE_BTC-USDT_bidq_4', 'BINANCE_BTC-USDT_bidq_5', 'BINANCE_BTC-USDT_bidq_6', 'BINANCE_BTC-USDT_bidq_7', 'BINANCE_BTC-USDT_bidq_8', 'BINANCE_BTC-USDT_bidq_9', 'BINANCE_BTC-USDT_bidq_10']].values
Mean = []
for i in range(len(df)):
    bq = 0
    sq = 0
    Bsum = 0
    Ssum = 0
    for j in range(10):
        if bq < 500:
            Bsum += df_bid[i][j] * df_bidq[i][j]
            bq += df_bidq[i][j]
        if sq < 500:
            Ssum += df_ask[i][j] * df_askq[i][j]
            sq += df_askq[i][j]
    Mean.append((Bsum+Ssum)/2)

Mean = np.array(Mean)
df['MEAN'] = Mean


df = df[:-40]
col = list(df)

"""
import seaborn as sns
from pylab import savefig
corr = df.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(40, 20))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
svm = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
figure = svm.get_figure()    
figure.savefig('corr1.png')"""

#X_col_old = ['BINANCE_BTC-USDT_ask_1', 'BINANCE_BTC-USDT_bid_1',
#             'New_Buy_Orders_No', 'Sum_Buy_Order_Prices',
#             'Total_Qty_Buy_Orders', 'Buy_Orders_No_Cancelled',
#             'LTP', 'b12', 'aten1', 'mid_p_Y', 'acc_diff_v',
#             'mean_P_ask', 'mean_P_bid', 'mean_V_ask',
#             'mean_V_bid', 'mid_P_future']



X_col = ['BINANCE_BTC-USDT_askq_1', 'Sum_Buy_Order_Prices', 'Total_Qty_Buy_Orders',
         'Trade_Price_Sum', 'b_a_spread', 'b12', 'aten1', 'mid_p_Y', 'acc_diff_P',
         'acc_diff_v', 'mid_P_slope', 'MEAN']

Feat = df[X_col]
Feat['MEAN'] = Feat['MEAN'] / 100
Feat['Trade_Price_Sum'] = Feat['Trade_Price_Sum'] / 100
Lab = df[['Ret']]



Lab_cod = pd.DataFrame(columns = Lab.columns, index = Lab.index)
Lab_cod['Ret'] = 0
percentile = np.percentile(Lab[:-5000].values, 66)
mask1 = Lab['Ret'] >= percentile 
mask2 = Lab['Ret'] <= -percentile
column_name = 'Ret'
Lab_cod.loc[mask1, column_name] = 2
Lab_cod.loc[mask2, column_name] = 1
print(Lab_cod['Ret'].value_counts())
print("****percentile = ", percentile)


X = Feat.values
y = Lab_cod.values
y = y.reshape(-1,1)

X_train = X[:-5000, :]
y_train = y[:-5000, :]
X_test = X[-5000:, :]
y_test = y[-5000:, :]


sc_X = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
dummy_y = np_utils.to_categorical(y_train)

#model = Sequential()
#model.add(Dense(output_dim = 8, input_dim=15, activation='relu'))
#model.add(Dense(output_dim = 4, activation='relu'))
#model.add(Dense(output_dim = 3, activation='softmax'))
#model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

model = Sequential()
model.add(Dense(output_dim = 6, input_dim=12, activation='relu'))
model.add(Dense(output_dim = 6, activation='relu'))
model.add(Dense(output_dim = 3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
history = model.fit(X_train_scaled, dummy_y, batch_size=16, validation_split = 0.2, epochs=20)

fig = plt.figure(figsize=(35, 10))
ax1 = plt.subplot2grid((2,2),(0,0))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('model accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.subplot2grid((2,2),(1,1))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
fig.show()




predictions = model.predict(X_test_scaled)
result = np.argmax(predictions,axis=1)

"""result = []

for i in predictions:
    if i[1] > 0.8:
        result.append(1)
    elif i[2] > 0.8:
        result.append(2)
    else:
        result.append(0)"""

#result = np.array(result)

ResDF = df[['BINANCE_BTC-USDT_bid_1', 'BINANCE_BTC-USDT_ask_1']][-35500:]
ResDF = ResDF.reset_index(drop=True)
ResDF['Predicted'] = result
ResDF['Real'] = y_test

print(ResDF['Predicted'].value_counts())

cm = confusion_matrix(y_test, ResDF['Predicted'])



#===============================================================================

Infer = []
for i in range(len(ResDF)):
    if ResDF['Predicted'][i] == 2 and ResDF['Real'][i] == 2:
        Infer.append("True Positive")
    elif ResDF['Predicted'][i] == 1 and ResDF['Real'][i] == 1:
        Infer.append("True Negative")
    else:
        Infer.append("Other")

ResDF["Infer"] = Infer

Tru_pos = ResDF[ResDF['Infer'] == 'True Positive']
Tru_neg = ResDF[ResDF['Infer'] == 'True Negative']


Real_time_feat = df[['timestamp','BINANCE_BTC-USDT_ask_1','BINANCE_BTC-USDT_bid_1']][-35500:]
Real_time_feat = Real_time_feat.reset_index(drop = True)
Record2 = pd.DataFrame(columns = ['T_bPrice', 'T_aPrice','Action', 'Net_Profit',
                                  'Position', 'TradQ', 'Cash', 'Shar_Amt', 'Cu_Credit', 
                                  'Cu_Debit', 'Trade', 'Turnover', 'P&L', 'P&LperTrade'])

B_LTP = []
S_LTP = []
LTP = df['LTP'][-35500:]
LTP = LTP.reset_index(drop = True)


Bid = []
Ask = []

bp = 0
ap = 0

Action = []
Profit = []
Shares = []
time = []
cur_pro = 0
bought = False
pos = []
p_ = 0
r=0
tradq = []
q1=0
q2=0

mon = 0
shr = 0

Cash = []
Share_Amt = []
fir = 0

credit = 0
debit = 0

CC = []
CD = []

pnl = 0
PNL = []

cur_trade = 0
pre_trade = 0
prev = 0

trade = []

turn = []

for var in range(len(Real_time_feat)):

    # Buying
    if result[var] == 2 and bought == False:
        B_LTP.append(LTP[var])
        
        bp = Real_time_feat['BINANCE_BTC-USDT_bid_1'][var]
        ap = Real_time_feat['BINANCE_BTC-USDT_ask_1'][var]
        Bid.append(bp)
        Ask.append(ap)
        Action.append('Bought')
        cur_pro -= (Real_time_feat['BINANCE_BTC-USDT_ask_1'][var])
        Profit.append(cur_pro)
        time.append(Real_time_feat['timestamp'][var])
        bought = True
        p_ = 1
        pos.append(p_)
        if r==0:
            shr = 1
            r=1
            q2=1
            #prev = -1
        else:
            q1 = q2
            r=r-1
            shr += 1
            q2 = p_
        tradq.append(q1-q2)

        Share_Amt.append(shr*Real_time_feat['BINANCE_BTC-USDT_ask_1'][var])

        mon -= Real_time_feat['BINANCE_BTC-USDT_ask_1'][var]
        Cash.append(mon)

        ref = q1-q2
        debit += Real_time_feat['BINANCE_BTC-USDT_ask_1'][var]*ref
        CD.append(-debit)
        CC.append(credit)

        #print(-debit, credit, p_, (ap+bp)/2)
        pnl = credit+debit + p_* (ap+bp)/2
        PNL.append(pnl)

        cur_trade = -ref * Real_time_feat['BINANCE_BTC-USDT_ask_1'][var]
        trade.append(cur_trade)
        sum_ = cur_trade+pre_trade
        turn.append(sum_)
        pre_trade = sum_



    # Selling
    elif result[var] == 1 and bought == True:
        
        S_LTP.append(LTP[var])
        bp = Real_time_feat['BINANCE_BTC-USDT_bid_1'][var]
        ap = Real_time_feat['BINANCE_BTC-USDT_ask_1'][var]
        Bid.append(bp)
        Ask.append(ap)
        Action.append('Sold')
        cur_pro += (Real_time_feat['BINANCE_BTC-USDT_bid_1'][var])
        Profit.append(cur_pro)
        time.append(Real_time_feat['timestamp'][var])
        bought = False

        p_ = -1
        q1 = q2
        q2 = p_
        pos.append(p_)

        r=r+1
        tradq.append(q1-q2)
        mon += Real_time_feat['BINANCE_BTC-USDT_bid_1'][var]
        Cash.append(mon)
        shr -= 1
        Share_Amt.append(shr*Real_time_feat['BINANCE_BTC-USDT_ask_1'][var])

        ref = q1-q2
        credit += Real_time_feat['BINANCE_BTC-USDT_bid_1'][var]*ref
        CC.append(credit)
        CD.append(-debit)
        #print(-debit, credit, p_, (ap+bp)/2)
        pnl = credit+debit + p_* (ap+bp)/2
        PNL.append(pnl)

        cur_trade = ref * Real_time_feat['BINANCE_BTC-USDT_bid_1'][var]
        trade.append(cur_trade)
        sum_ = cur_trade+pre_trade
        turn.append(sum_)
        pre_trade = sum_


Record2['timestamp'] = time
Record2['T_bPrice'] = Bid
Record2['T_aPrice'] = Ask
Record2['Action'] = Action
Record2['Net_Profit'] = Profit
Record2['Position'] = pos
Record2['TradQ'] = tradq

Record2['Cash'] = Cash
Record2['Shar_Amt'] = Share_Amt

Record2['Cu_Credit'] = CC
Record2['Cu_Debit'] = CD

Record2['Trade'] = trade
Record2['P&L'] = PNL
Record2['Turnover'] = turn

Record2['P&LperTrade'] = Record2['P&L'] / Record2['Turnover']
Record2['trans_cost'] = Record2['Turnover'] * (0.03/100)
Record2['PnL_after_com'] = Record2['P&L'] - Record2['trans_cost']

plt.figure(figsize = (10, 5))
plt.plot(Record2['Cu_Credit'])
plt.plot(-Record2['Cu_Debit'])
plt.legend()
plt.show()

plt.figure(figsize = (10, 5))
plt.plot(Record2['PnL_after_com'])
plt.plot(Record2['P&L'])
plt.legend()
plt.show()


Time = df['timestamp'][-35500:]
LTP = df['LTP'][-35500:]
Time = Time.reset_index(drop = True)
LTP = LTP.reset_index(drop = True)    

#return Record2, Time, LTP




ltp_b = []
x_b = []
ltp_s = []
x_s = []
star = 0
for i in range(len(Record2)):
    for j in range(star, len(Time)):
        if Record2['timestamp'][i] == Time[j]:
            if Record2['Action'][i] == 'Bought':
                ltp_b.append(LTP[j])
                x_b.append(j)
                star = j
            else:
                ltp_s.append(LTP[j])
                x_s.append(j)
                star = j


ltp_b = np.array(ltp_b)
x_b = np.array(x_b)
ltp_s = np.array(ltp_s)
x_s = np.array(x_s)

#return ltp_b, x_b, ltp_s, x_s


plt.figure(figsize = (60,30))
ax0 = plt.plot(LTP, color = 'black', linewidth=0.2)
ax1 = plt.scatter(x_b, ltp_b, color = 'b')
ax2 = plt.scatter(x_s, ltp_s, color = 'r')
plt.legend((ax0, ax1, ax2), ('LTP', 'Bought', 'Sold'))
plt.savefig('myfig.png')
plt.show()

latency = Record2['timestamp'].shift(-1)[:-1] - Record2['timestamp'][:-1]
ax = sns.distplot(Record2['PnL_after_com'])

