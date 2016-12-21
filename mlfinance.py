import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ML for finance

df = pd.read_csv('spy.csv', index_col="Date")
df = df.iloc[::-1]

aapl_df = pd.read_csv('aapl.csv', index_col="Date")
aapl_df = aapl_df.iloc[::-1]

# merge data set on dates using suffix on to mrege same column names
big_df = df.join(aapl_df, lsuffix="SPY", rsuffix="AAPL")
close_df = big_df[['CloseSPY', 'CloseAAPL']].copy()
# mean 20 days
rm = pd.rolling_mean(df["Close"], window=20)
# std 20 days
sd = pd.rolling_std(df["Close"], window=20)
cp = df["Close"]
cp_copy = cp.copy()
# daily rerurns
cp_copy[1:] = (cp[1:] / cp[:-1].values) - 1
cp_copy[0] = 0
# cumulative returns
cr = cp.copy()
cr[1:] = (cp[:-1].values / cp[0]) - 1
cr[0] = 0
print (cr)

# bollinger bands
lw = rm - 2 * sd
up = rm + 2 * sd
plt.plot(cp, color="b")
plt.plot(lw, color="r")
plt.plot(up, color="y")
plt.xlim(0, len(cp))
plt.title("Close price SPY")
# mult charts at same time
plt.figure()
# histgram
plt.hist(cp_copy, bins=20)
plt.axvline(cp_copy.mean(), color="w", linestyle="dashed")
plt.axvline(-cp_copy.std(), color="r", linestyle="dashed")
plt.axvline(cp_copy.std(), color="r", linestyle="dashed")
plt.figure()
# daily returns
plt.plot(cp_copy, color="k")
plt.ylim(cp_copy.min(), cp_copy.max())
plt.xlim(0, len(cp_copy))
plt.title("Daily returns SPY")
plt.figure()
# cum returns
plt.plot(cr, color="m")
plt.title("Cumulative returns SPY")
plt.ylim(cr.min(), cr.max())
plt.xlim(0, len(cp))
plt.figure()

plt.scatter(x=close_df["CloseSPY"], y=close_df["CloseAAPL"])

plt.show()
