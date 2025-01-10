import datetime as dt
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.options.mode.chained_assignment = None

df_ = pd.read_csv(r'C:\Users\ap\Desktop\miuul\CRM\FLOMusteriSegmentasyonu\flo_data_20k.csv')
df = df_.copy()
df.head()
df.describe().T
def outlier_thresholds(DataFrame, variable):
    quartile1 = DataFrame[variable].quantile(0.01)
    quartile3 = DataFrame[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(DataFrame, variable):
    low_limit, up_limit = outlier_thresholds(DataFrame, variable)
    low_limit = min(low_limit, 0)
    DataFrame.loc[(DataFrame[variable] < low_limit), variable] = round(low_limit, 0)
    DataFrame.loc[(DataFrame[variable] > up_limit), variable] = round(up_limit, 0)

columns =["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online"]
for col in columns:
    replace_with_thresholds(df, col)

df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_price"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df.info()
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)

rfm = pd.DataFrame()
rfm["cust_id"] = df["master_id"]
rfm["recency"] = today_date - df["last_order_date"]
rfm["frequency"] = df["total_order"]
rfm["monetary"] = df["total_price"]
rfm["T"] = df["last_order_date"] - df["first_order_date"]
rfm.head()
rfm.info()

cltv_df = pd.DataFrame()
cltv_df.head()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_weekly"] = ((df["last_order_date"] - df["first_order_date"])) / 7
cltv_df["T_weekly"] = (today_date - df["first_order_date"]) / 7
cltv_df["frequency"] = df["total_order"]
cltv_df["monetary_avg"] = df["total_price"] / df["total_order"]
cltv_df.info()
cltv_df["recency_weekly"] = cltv_df["recency_weekly"].dt.days
cltv_df["T_weekly"] = cltv_df["T_weekly"].dt.days


bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"],
        cltv_df["recency_weekly"],
        cltv_df["T_weekly"])

cltv_df["exp_sales_3_month"] = bgf.predict(4*3,
                                           cltv_df["frequency"],
                                           cltv_df["recency_weekly"],
                                           cltv_df["T_weekly"])

cltv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                           cltv_df["frequency"],
                                           cltv_df["recency_weekly"],
                                           cltv_df["T_weekly"])

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"], cltv_df["monetary_avg"])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                       cltv_df["monetary_avg"])

cltv_df["CLTV"] = ggf.customer_lifetime_value(bgf,
                                              cltv_df["frequency"],
                                              cltv_df["recency_weekly"],
                                              cltv_df["T_weekly"],
                                              cltv_df["monetary_avg"],
                                              time=6,
                                              freq='W',
                                              discount_rate=0.01)
cltv_df.head()
cltv_df.sort_values("CLTV", ascending=False).head(20)

cltv_df["segment"] = pd.qcut(cltv_df["CLTV"], 4, labels=["D", "C", "B", "A"])