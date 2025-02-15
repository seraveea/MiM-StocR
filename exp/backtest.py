import qlib
import pandas as pd
from qlib.utils.time import Freq
from qlib.utils import flatten_dict
from qlib.backtest import backtest, executor
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy


def backtest_helper(data, start_date, end_date, topk, n_drop, model_name):
    # Benchmark is for calculating the excess return of your strategy.
    # Its data format will be like **ONE normal instrument**.
    # For example, you can query its data with the code below
    # `D.features(["SH000300"], ["$close"], start_time='2010-01-01', end_time='2017-12-31', freq='day')`
    # It is different from the argument `market`, which indicates a universe of stocks (e.g. **A SET** of stocks like csi300)
    # For example, you can query all data from a stock market with the code below.
    # ` D.features(D.instruments(market='csi300'), ["$close"], start_time='2010-01-01', end_time='2017-12-31', freq='day')`

    qlib.init(provider_uri="../../qlib_data/cn_data")
    CSI100_BENCH = "SH000903"
    CSI300_BENCH = "SH000300"
    FREQ = "day"
    STRATEGY_CONFIG = {
        "topk": topk,
        "n_drop": n_drop,
        "signal": data,
        "only_tradable": True,
        "risk_degree": 0.95
    }

    EXECUTOR_CONFIG = {
        "time_per_step": "day",
        "generate_portfolio_metrics": True,
    }

    backtest_config = {
        "start_time": start_date,
        "end_time": end_date,
        "account": 100000000,
        "benchmark": CSI300_BENCH,  # "benchmark": NASDAQ_BENCH,
        "exchange_kwargs": {
            "freq": FREQ,
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.00005,
            "close_cost": 0.0003,
            "min_cost": 5,
        },
    }

    # strategy object
    strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
    # executor object
    executor_obj = executor.SimulatorExecutor(**EXECUTOR_CONFIG)
    # backtest
    portfolio_metric_dict, indicator_dict = backtest(executor=executor_obj, strategy=strategy_obj, **backtest_config)
    analysis_freq = "{0}{1}".format(*Freq.parse(FREQ))
    # backtest info
    report_normal, positions_normal = portfolio_metric_dict.get(analysis_freq)
    #     pf.create_simple_tear_sheet(returns=report_normal["return"] - report_normal["cost"],benchmark_rets=report_normal['bench'])
    #     pf.create_simple_tear_sheet(returns=report_normal["return"] - report_normal["bench"] - report_normal["cost"],benchmark_rets=report_normal['bench'])

    # analysis
    analysis = dict()
    #     analysis["excess_return_without_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"], freq=analysis_freq)
    #     analysis["excess_return_with_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq=analysis_freq)
    analysis["excess_return_with_cost"] = risk_analysis(
        report_normal["return"] - report_normal["cost"] - report_normal["bench"], freq=analysis_freq)
    #
    analysis_df = pd.concat(analysis)  # type: pd.DataFrame
    # log metrics
    analysis_dict = flatten_dict(analysis_df["risk"].unstack().T.to_dict())
    analysis["excess_return_with_cost"].rename(columns={'risk': model_name + '_' + start_date[:4] + 'top' + str(topk)},
                                               inplace=True)
    df = analysis["excess_return_with_cost"].transpose()
    return df, report_normal
    # a df here, give it date, topk and model, train method, return a df column then

    # qcr.analysis_position.report_graph(report_normal)


def backtest_module(symbol_file, start_date, end_date, topk, drop_n, model_name):
    data = pd.read_pickle(symbol_file)

    def helper(x):
        return x['pred_score']

    if 'pred_class' in data.columns:
        data['symbol'] = data.apply(lambda x: helper(x), axis=1)
        data = data[['symbol']]
        data.columns = [['score']]
    else:
        data = data[['score']]
        data.columns = [['score']]
    slc = slice(pd.Timestamp(start_date), pd.Timestamp(end_date))
    data = data[slc]
    return backtest_helper(data, start_date, end_date, topk, drop_n, model_name)


return_analysis, report_normal = backtest_module(
    '../pred_output/LSTM_ourmethod.pkl','2020-01-01', '2020-12-31', 50, 5, 'placeholder'
)
print(return_analysis)
