"""
A股ETF期权双买策略 - 回测框架

基于A股真实市场特征的历史事件回测 + 蒙特卡洛模拟
"""

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from cn_config import RISK_FREE_RATE, UNDERLYING_MAP
from cn_straddle import (
    BSModel,
    CnMarketSnapshot,
    CnStraddleStrategy,
    SignalType,
)


# ============================================================
# A股历史事件数据
# ============================================================

@dataclass
class CnHistoricalEvent:
    """A股真实历史事件"""
    event_date: date
    event_type: str
    description: str
    underlying: str              # 标的
    price_before: float          # 事件前ETF价格
    price_after: float           # 事件后ETF价格
    iv_before: float             # 事件前IV（小数）
    iv_after: float              # 事件后IV峰值（小数）
    iv_percentile: float         # 事件前IV历史分位
    dte_typical: int             # 最近月合约剩余天数


# 2024-2025年A股关键事件
# 基于50ETF和300ETF公开行情数据特征
CN_HISTORICAL_EVENTS = [
    # ---- 2024年 ----
    CnHistoricalEvent(
        event_date=date(2024, 1, 22),
        event_type="RRR_CUT",
        description="2024年1月央行宣布降准50bp，超预期",
        underlying="50ETF",
        price_before=2.280,
        price_after=2.350,
        iv_before=0.155,
        iv_after=0.225,
        iv_percentile=15,
        dte_typical=28,
    ),
    CnHistoricalEvent(
        event_date=date(2024, 2, 6),
        event_type="POLICY_MEETING",
        description="2024年2月国务院专题会议，稳市场一揽子政策",
        underlying="50ETF",
        price_before=2.180,
        price_after=2.360,
        iv_before=0.320,
        iv_after=0.480,
        iv_percentile=85,  # 高波环境，不应入场
        dte_typical=21,
    ),
    CnHistoricalEvent(
        event_date=date(2024, 3, 5),
        event_type="POLICY_MEETING",
        description="2024年两会开幕，GDP目标5%左右",
        underlying="300ETF_SH",
        price_before=3.450,
        price_after=3.520,
        iv_before=0.178,
        iv_after=0.210,
        iv_percentile=22,
        dte_typical=18,
    ),
    CnHistoricalEvent(
        event_date=date(2024, 4, 30),
        event_type="POLITBURO",
        description="2024年4月政治局会议，提出灵活运用利率汇率工具",
        underlying="50ETF",
        price_before=2.520,
        price_after=2.580,
        iv_before=0.142,
        iv_after=0.188,
        iv_percentile=10,
        dte_typical=25,
    ),
    CnHistoricalEvent(
        event_date=date(2024, 7, 22),
        event_type="RATE_CUT",
        description="2024年7月LPR超预期下调10bp",
        underlying="50ETF",
        price_before=2.410,
        price_after=2.365,
        iv_before=0.138,
        iv_after=0.195,
        iv_percentile=8,
        dte_typical=14,
    ),
    CnHistoricalEvent(
        event_date=date(2024, 9, 24),
        event_type="POLICY_MEETING",
        description="2024年924新政：央行+金融监管+证监会联合发布会",
        underlying="50ETF",
        price_before=2.385,
        price_after=2.780,
        iv_before=0.155,
        iv_after=0.650,
        iv_percentile=18,
        dte_typical=21,
    ),
    CnHistoricalEvent(
        event_date=date(2024, 10, 8),
        event_type="POLICY_MEETING",
        description="2024年国庆后开盘+发改委发布会（预期落空）",
        underlying="50ETF",
        price_before=2.980,
        price_after=2.720,
        iv_before=0.580,
        iv_after=0.420,
        iv_percentile=95,  # 极高波，不应入场
        dte_typical=14,
    ),
    CnHistoricalEvent(
        event_date=date(2024, 12, 9),
        event_type="POLITBURO",
        description="2024年12月政治局会议，罕见提出'适度宽松'货币政策",
        underlying="50ETF",
        price_before=2.680,
        price_after=2.750,
        iv_before=0.195,
        iv_after=0.280,
        iv_percentile=28,
        dte_typical=18,
    ),

    # ---- 2025年 ----
    CnHistoricalEvent(
        event_date=date(2025, 1, 20),
        event_type="LPR",
        description="2025年1月LPR维持不变",
        underlying="50ETF",
        price_before=2.620,
        price_after=2.635,
        iv_before=0.168,
        iv_after=0.185,
        iv_percentile=20,
        dte_typical=28,
    ),
    CnHistoricalEvent(
        event_date=date(2025, 2, 17),
        event_type="POLICY_MEETING",
        description="2025年2月民营企业座谈会，提振信心",
        underlying="300ETF_SH",
        price_before=3.820,
        price_after=3.920,
        iv_before=0.175,
        iv_after=0.235,
        iv_percentile=22,
        dte_typical=21,
    ),
    CnHistoricalEvent(
        event_date=date(2025, 3, 5),
        event_type="POLICY_MEETING",
        description="2025年两会开幕，GDP目标及赤字率关注",
        underlying="50ETF",
        price_before=2.750,
        price_after=2.790,
        iv_before=0.162,
        iv_after=0.210,
        iv_percentile=16,
        dte_typical=18,
    ),
    CnHistoricalEvent(
        event_date=date(2025, 3, 20),
        event_type="LPR",
        description="2025年3月LPR维持不变，市场平稳",
        underlying="50ETF",
        price_before=2.770,
        price_after=2.760,
        iv_before=0.148,
        iv_after=0.165,
        iv_percentile=12,
        dte_typical=10,
    ),
]


# ============================================================
# 回测引擎
# ============================================================

def _generate_cn_price_path(price_before: float, price_after: float,
                            days_before: int, days_after: int) -> list[float]:
    """生成A股ETF价格路径"""
    path = []
    total_move = (price_after - price_before) / price_before

    for day in range(days_before + 1 + days_after):
        if day < days_before:
            # 事件前随机波动
            noise = np.random.normal(0, 0.005)
            price = price_before * (1 + noise)
        elif day == days_before:
            # 事件当日
            price = price_before * (1 + total_move * 0.70)
        elif day == days_before + 1:
            price = price_before * (1 + total_move * 0.90)
        else:
            # 后续回归
            price = price_after * (1 + np.random.normal(0, 0.008))
        path.append(price)

    return path


def _generate_cn_iv_path(iv_before: float, iv_after: float,
                         days_before: int, days_after: int) -> list[float]:
    """生成A股IV路径"""
    path = []
    total_days = days_before + 1 + days_after

    for day in range(total_days):
        if day < days_before:
            frac = day / max(days_before, 1)
            iv = iv_before + (iv_after - iv_before) * 0.15 * frac
        elif day <= days_before + 1:
            iv = iv_after
        else:
            decay = day - days_before - 1
            iv = iv_before + (iv_after - iv_before) * math.exp(-0.4 * decay)
        path.append(iv)

    return path


def backtest_single_event(event: CnHistoricalEvent,
                          account_balance: float = 200000.0) -> dict:
    """
    回测单个A股历史事件

    默认账户20万元（A股期权常见账户规模）
    """
    np.random.seed(hash(event.event_date.isoformat()) % 2**31)

    underlying_key = event.underlying
    strategy = CnStraddleStrategy(account_balance, underlying_key)

    # 行权价取最近的0.05整数
    strike_interval = UNDERLYING_MAP[underlying_key]["strike_interval"]
    strike = round(event.price_before / strike_interval) * strike_interval

    days_before = min(event.dte_typical // 4, 5)
    days_after = 3

    prices = _generate_cn_price_path(event.price_before, event.price_after,
                                     days_before, days_after)
    ivs = _generate_cn_iv_path(event.iv_before, event.iv_after,
                               days_before, days_after)

    result = {
        "event": event.description,
        "date": event.event_date.isoformat(),
        "type": event.event_type,
        "underlying": event.underlying,
        "entered": False,
        "trade": None,
        "skip_reason": None,
        "daily_log": [],
    }

    for day_idx in range(len(prices)):
        price = prices[day_idx]
        iv = ivs[day_idx]
        dte = max(event.dte_typical - day_idx, 1)
        T = dte / 365
        event_offset = day_idx - days_before  # 负=事件前

        call_px = BSModel.call_price(price, strike, T, RISK_FREE_RATE, iv)
        put_px = BSModel.put_price(price, strike, T, RISK_FREE_RATE, iv)

        # IV分位调整
        if event_offset <= 0:
            iv_pct = event.iv_percentile
        else:
            iv_pct = min(event.iv_percentile + event_offset * 20, 95)

        snapshot = CnMarketSnapshot(
            timestamp=f"{event.event_date} T{event_offset:+d}",
            underlying=event.underlying,
            underlying_price=price,
            atm_call_price=call_px,
            atm_put_price=put_px,
            atm_strike=strike,
            iv_call=iv,
            iv_put=iv * 1.05,  # A股认沽通常有更大的skew
            dte=dte,
            days_to_next_event=max(-event_offset, 0),
            next_event_type=event.event_type,
            next_event_desc=event.description,
            iv_percentile=iv_pct,
        )

        log_entry = {
            "time": snapshot.timestamp,
            "price": f"{price:.3f}",
            "iv": f"{iv:.1%}",
            "call": f"{call_px:.4f}",
            "put": f"{put_px:.4f}",
            "straddle": f"{call_px + put_px:.4f}",
        }

        if strategy.position is None:
            signal = strategy.check_entry_signal(snapshot)
            if signal == SignalType.ENTRY:
                pos = strategy.open_straddle(snapshot)
                result["entered"] = True
                log_entry["action"] = f"开仓 {pos.contracts}张 行权价={strike:.2f}"
            elif event_offset == 0 and not result["entered"]:
                reasons = []
                avg_iv = (snapshot.iv_call + snapshot.iv_put) / 2
                if iv_pct > 30:
                    reasons.append(f"IV分位{iv_pct}%>30%")
                if avg_iv > 0.25:
                    reasons.append(f"IV={avg_iv:.1%}>25%")
                if avg_iv < 0.08:
                    reasons.append(f"IV={avg_iv:.1%}<8%")
                if snapshot.days_to_next_event > 5:
                    reasons.append(f"距事件{snapshot.days_to_next_event}天>5天")
                result["skip_reason"] = "; ".join(reasons) if reasons else "其他条件不满足"
        else:
            days_since = event_offset if event_offset > 0 else -1
            signal = strategy.check_exit_signal(snapshot, days_since)
            if signal != SignalType.NO_SIGNAL:
                trade = strategy.close_straddle(snapshot, signal.value)
                result["trade"] = trade
                log_entry["action"] = (
                    f"平仓({signal.value}) "
                    f"盈亏={trade.total_pnl:+,.0f}元 ({trade.return_pct:+.1f}%)"
                )
            else:
                pnl_info = strategy.get_current_pnl(snapshot)
                log_entry["action"] = f"持仓中 浮盈={pnl_info['total_pnl']:+,.0f}元"

        result["daily_log"].append(log_entry)

    # 未平仓则强制平仓
    if strategy.position is not None and prices:
        last_dte = max(event.dte_typical - len(prices) + 1, 1)
        T = last_dte / 365
        final_snapshot = CnMarketSnapshot(
            timestamp=f"{event.event_date} FORCE_CLOSE",
            underlying=event.underlying,
            underlying_price=prices[-1],
            atm_call_price=BSModel.call_price(prices[-1], strike, T, RISK_FREE_RATE, ivs[-1]),
            atm_put_price=BSModel.put_price(prices[-1], strike, T, RISK_FREE_RATE, ivs[-1]),
            atm_strike=strike,
            iv_call=ivs[-1],
            iv_put=ivs[-1] * 1.05,
            dte=last_dte,
            days_to_next_event=0,
            next_event_type=event.event_type,
            next_event_desc=event.description,
            iv_percentile=50,
        )
        trade = strategy.close_straddle(final_snapshot, "强制平仓")
        result["trade"] = trade

    return result


def run_cn_backtest() -> pd.DataFrame:
    """运行A股全部历史事件回测"""
    rows = []

    for event in CN_HISTORICAL_EVENTS:
        result = backtest_single_event(event)
        trade = result["trade"]

        row = {
            "日期": event.event_date.strftime("%Y-%m-%d"),
            "事件": event.event_type,
            "描述": event.description[:25],
            "标的": event.underlying,
            "价格": f"{event.price_before:.3f}→{event.price_after:.3f}",
            "IV": f"{event.iv_before:.1%}→{event.iv_after:.1%}",
            "IV分位": f"{event.iv_percentile}%",
            "入场": "是" if result["entered"] else "否",
        }

        if trade:
            row["收益率"] = f"{trade.return_pct:+.1f}%"
            row["盈亏(元)"] = f"{trade.total_pnl:+,.0f}"
            row["认购PnL"] = f"{trade.call_pnl:+,.0f}"
            row["认沽PnL"] = f"{trade.put_pnl:+,.0f}"
            row["退出原因"] = trade.exit_reason
        else:
            row["收益率"] = "N/A"
            row["盈亏(元)"] = "N/A"
            row["认购PnL"] = "N/A"
            row["认沽PnL"] = "N/A"
            row["退出原因"] = result.get("skip_reason", "未入场")

        rows.append(row)

    return pd.DataFrame(rows)


def cn_monte_carlo(n_simulations: int = 1000,
                   underlying: str = "50ETF",
                   account_balance: float = 200000.0) -> dict:
    """
    A股期权蒙特卡洛模拟

    基于A股ETF期权市场特征随机生成场景
    """
    np.random.seed(42)
    pnl_list = []
    return_list = []

    # A股ETF典型价格范围
    price_range = {
        "50ETF": (2.2, 3.2),
        "300ETF_SH": (3.2, 4.5),
        "300ETF_SZ": (3.2, 4.5),
        "500ETF": (4.5, 7.0),
        "1000ETF": (1.5, 2.5),
    }
    lo, hi = price_range.get(underlying, (2.0, 3.5))

    for _ in range(n_simulations):
        base_price = np.random.uniform(lo, hi)

        # A股IV特征：常态12-22%，极端可到60%+
        iv_before = np.random.uniform(0.10, 0.22)
        iv_jump = np.random.lognormal(mean=0.2, sigma=0.7)
        iv_after = min(iv_before * (1 + iv_jump), 0.80)

        # 价格变动
        price_move = np.random.choice([-1, 1]) * np.random.lognormal(mean=-3.5, sigma=1.0)
        price_move = max(min(price_move, 0.15), -0.15)  # A股ETF单日波动通常<10%
        price_after = base_price * (1 + price_move)

        event = CnHistoricalEvent(
            event_date=date(2024, 6, 1),
            event_type="MC_SIM",
            description=f"蒙特卡洛模拟#{_}",
            underlying=underlying,
            price_before=base_price,
            price_after=price_after,
            iv_before=iv_before,
            iv_after=iv_after,
            iv_percentile=np.random.uniform(5, 30),
            dte_typical=np.random.randint(7, 30),
        )

        result = backtest_single_event(event, account_balance)
        if result["trade"]:
            pnl_list.append(result["trade"].total_pnl)
            return_list.append(result["trade"].return_pct)

    pnl_arr = np.array(pnl_list)
    ret_arr = np.array(return_list)

    if len(pnl_list) == 0:
        return {
            "标的": underlying,
            "模拟次数": n_simulations,
            "有效交易": 0,
            "入场率": "0.0%",
            "说明": "无交易触发，请检查入场条件参数",
        }

    return {
        "标的": underlying,
        "模拟次数": n_simulations,
        "有效交易": len(pnl_list),
        "入场率": f"{len(pnl_list)/n_simulations:.1%}",
        "---盈亏统计(元)---": "",
        "平均盈亏": f"{np.mean(pnl_arr):+,.0f}",
        "中位数盈亏": f"{np.median(pnl_arr):+,.0f}",
        "最大盈利": f"{np.max(pnl_arr):+,.0f}",
        "最大亏损": f"{np.min(pnl_arr):+,.0f}",
        "---收益率统计---": "",
        "平均收益率": f"{np.mean(ret_arr):+.1f}%",
        "中位数收益率": f"{np.median(ret_arr):+.1f}%",
        "盈利概率": f"{np.mean(ret_arr > 0):.1%}",
        "翻倍概率(>100%)": f"{np.mean(ret_arr > 100):.1%}",
        "3倍以上概率": f"{np.mean(ret_arr > 200):.1%}",
        "亏损>50%概率": f"{np.mean(ret_arr < -50):.1%}",
        "---收益分位数---": "",
        "5%分位": f"{np.percentile(ret_arr, 5):+.1f}%",
        "25%分位": f"{np.percentile(ret_arr, 25):+.1f}%",
        "50%分位": f"{np.percentile(ret_arr, 50):+.1f}%",
        "75%分位": f"{np.percentile(ret_arr, 75):+.1f}%",
        "95%分位": f"{np.percentile(ret_arr, 95):+.1f}%",
    }


def compute_cn_stats(df: pd.DataFrame) -> dict:
    """计算A股回测统计"""
    entered = df[df["入场"] == "是"]
    if len(entered) == 0:
        return {"交易次数": 0}

    pnls = []
    returns = []
    for _, row in entered.iterrows():
        try:
            p = float(str(row["盈亏(元)"]).replace(",", "").replace("+", ""))
            r = float(str(row["收益率"]).replace("%", "").replace("+", ""))
            pnls.append(p)
            returns.append(r)
        except (ValueError, AttributeError):
            pass

    if not pnls:
        return {"交易次数": len(entered), "有效数据": 0}

    pnl_arr = np.array(pnls)
    ret_arr = np.array(returns)
    wins = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr <= 0]

    return {
        "总事件数": len(CN_HISTORICAL_EVENTS),
        "入场次数": len(entered),
        "入场率": f"{len(entered)/len(CN_HISTORICAL_EVENTS):.0%}",
        "盈利次数": len(wins),
        "亏损次数": len(losses),
        "胜率": f"{len(wins)/len(pnl_arr):.0%}",
        "累计盈亏(元)": f"{np.sum(pnl_arr):+,.0f}",
        "平均盈亏(元)": f"{np.mean(pnl_arr):+,.0f}",
        "平均收益率": f"{np.mean(ret_arr):+.1f}%",
        "中位数收益率": f"{np.median(ret_arr):+.1f}%",
        "最大收益率": f"{np.max(ret_arr):+.1f}%",
        "最大亏损率": f"{np.min(ret_arr):+.1f}%",
        "盈亏比": (
            f"{np.mean(wins)/abs(np.mean(losses)):.1f}:1"
            if len(losses) > 0 and np.mean(losses) != 0 else "N/A"
        ),
    }


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 72)
    print("A股ETF期权双买策略（跨式/Straddle）回测报告")
    print("适用标的：50ETF / 300ETF / 500ETF / 1000ETF 期权")
    print(f"回测事件数：{len(CN_HISTORICAL_EVENTS)} 个")
    print("=" * 72)

    # 1. 历史事件回测
    print("\n[1] 历史事件回测结果:")
    print("-" * 72)
    df = run_cn_backtest()
    for _, row in df.iterrows():
        entered_mark = ">>>" if row["入场"] == "是" else "   "
        print(
            f"  {entered_mark} {row['日期']} | {row['事件']:>15s} | "
            f"{row['标的']:>8s} | {row['价格']:>16s} | "
            f"IV {row['IV']:>12s} ({row['IV分位']:>4s}) | "
            f"入场={row['入场']} | {row['收益率']:>10s} | "
            f"盈亏={row['盈亏(元)']:>10s} | {row['退出原因']}"
        )

    # 2. 统计汇总
    print("\n\n[2] 策略统计汇总:")
    print("-" * 72)
    stats = compute_cn_stats(df)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # 3. 重点事件详细日志
    print("\n\n[3] 重点事件逐日记录:")
    key_indices = [0, 4, 5, 10]  # 降准、降息、924、两会
    for idx in key_indices:
        if idx < len(CN_HISTORICAL_EVENTS):
            event = CN_HISTORICAL_EVENTS[idx]
            result = backtest_single_event(event)
            print(f"\n  {'='*60}")
            print(f"  事件: {event.description}")
            print(f"  日期: {event.event_date} | 标的: {event.underlying}")
            print(f"  {'='*60}")
            for entry in result["daily_log"]:
                action = entry.get("action", "")
                print(
                    f"    {entry['time']:>25s} | ETF={entry['price']:>7s} | "
                    f"IV={entry['iv']:>6s} | C={entry['call']:>8s} P={entry['put']:>8s} | "
                    f"跨式={entry['straddle']:>8s} | {action}"
                )
            if result["trade"]:
                t = result["trade"]
                print(
                    f"\n    结果: 收益率={t.return_pct:+.1f}% | "
                    f"盈亏={t.total_pnl:+,.0f}元 "
                    f"(认购={t.call_pnl:+,.0f}, 认沽={t.put_pnl:+,.0f}) | "
                    f"IV: {t.entry_iv:.1%}->{t.exit_iv:.1%} | "
                    f"ETF: {t.entry_price:.3f}->{t.exit_price:.3f}"
                )

    # 4. 蒙特卡洛模拟
    print("\n\n[4] 蒙特卡洛模拟 (50ETF, 1000次):")
    print("-" * 72)
    mc = cn_monte_carlo(1000, "50ETF")
    for k, v in mc.items():
        if v == "":
            print(f"\n  {k}")
        else:
            print(f"  {k}: {v}")

    # 5. 策略结论
    print("\n\n" + "=" * 72)
    print("A股期权双买策略 - 适用性结论")
    print("=" * 72)
    print("""
    一、策略核心逻辑适用A股：
    -------------------------------------------------
    1. A股ETF期权支持T+0，买方可当日开平仓，止盈更灵活
    2. A股政策驱动特征明显，重大会议/降息等事件前后IV变化显著
    3. 50ETF/300ETF期权流动性充足，跨式策略可执行
    4. 买方最大亏损=权利金，风险完全可控

    二、A股 vs 黄金期权的关键差异：
    -------------------------------------------------
    | 维度       | A股ETF期权              | 黄金期权(COMEX)        |
    |-----------|------------------------|----------------------|
    | 合约乘数   | 10000                  | 100盎司               |
    | 行权价间距 | 0.05元                  | $5-$10               |
    | IV中枢    | 15-20%（常态）           | 12-18%               |
    | 极端IV    | 可达60%+（如924行情）     | 30-40%               |
    | 交易时间   | T+0                    | 近24小时              |
    | 事件类型   | 政策会议/LPR/PMI        | FOMC/NFP/CPI         |
    | 手续费    | ~4.3元/张               | ~$5/手                |
    | 流动性    | 50ETF/300ETF好，其他一般  | GLD期权非常好          |

    三、关键成功要素：
    -------------------------------------------------
    1. 入场时机：IV分位<30% + 距重大事件<=5天
    2. 标的选择：优先50ETF/300ETF（流动性最好）
    3. 合约选择：当月或下月，DTE 7-35天
    4. 仓位控制：单次不超过账户5%（如20万账户，单次<=1万权利金）
    5. T+0止盈：事件爆发当日可即时止盈，不必等到次日

    四、风险提示：
    -------------------------------------------------
    1. 大部分交易小亏（权利金归零）——需要足够的交易次数
    2. A股期权滑点较大（尤其500ETF/1000ETF），实际成本比回测高
    3. 临近到期的虚值期权流动性极差，避免持有到最后3天
    4. 高波环境（IV分位>30%）严禁入场——如2024年2月和10月
    5. 政策预期被市场充分定价时效果差——需关注预期差
    """)
