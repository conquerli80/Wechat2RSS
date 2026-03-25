"""
A股ETF期权双买策略 (China A-Share ETF Options Long Straddle)
============================================================

适用场景：
- 50ETF期权、300ETF期权、500ETF期权、1000ETF期权
- 事件驱动 + 低隐波入场的跨式买入策略

核心逻辑：
1. 在隐含波动率历史低位 + 重大事件（LPR/PMI/政治局会议等）前入场
2. 同时买入平值看涨和看跌期权（跨式组合/Straddle）
3. 通过 Vega（波动率上升）和 Gamma（方向爆发）获利
4. 最大亏损 = 总权利金，风险完全可控
5. 支持T+0日内止盈

A股期权特点：
- T+0交易（买方可当日开平仓）
- 合约乘数10000
- 行权价间距通常0.05元
- 当月/下月/季月合约
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from scipy.stats import norm

from cn_config import (
    COMMISSION_PER_CONTRACT,
    DEFAULT_UNDERLYING,
    EXCHANGE_FEE,
    IV_ABSOLUTE_MAX,
    IV_ABSOLUTE_MIN,
    IV_PERCENTILE_THRESHOLD,
    IV_PULLBACK_THRESHOLD,
    LEG_TAKE_PROFIT_MULTIPLIER,
    MAX_ABS_DELTA,
    MAX_CONCURRENT_STRADDLES,
    MAX_DAYS_TO_EVENT,
    MAX_DTE,
    MAX_HOLD_DAYS_AFTER_EVENT,
    MAX_LEG_RATIO,
    MAX_LOSS_RATIO,
    MAX_POSITION_RATIO,
    MIN_DTE,
    MIN_GAMMA_PER_CONTRACT,
    MIN_VEGA_PER_CONTRACT,
    RISK_FREE_RATE,
    SLIPPAGE_PCT,
    TAKE_PROFIT_MULTIPLIER,
    TIME_STOP_DTE,
    UNDERLYING_MAP,
)


# ============================================================
# Black-Scholes 定价模型（与黄金版相同，BS模型是通用的）
# ============================================================

class BSModel:
    """Black-Scholes 期权定价模型"""

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return 0.0
        return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return 0.0
        return BSModel.d1(S, K, T, r, sigma) - sigma * math.sqrt(T)

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return max(S - K, 0)
        d1 = BSModel.d1(S, K, T, r, sigma)
        d2 = BSModel.d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return max(K - S, 0)
        d1 = BSModel.d1(S, K, T, r, sigma)
        d2 = BSModel.d2(S, K, T, r, sigma)
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def implied_vol(market_price: float, S: float, K: float, T: float,
                    r: float, is_call: bool, tol: float = 1e-6,
                    max_iter: int = 100) -> Optional[float]:
        """牛顿法求解隐含波动率"""
        sigma = 0.2
        for _ in range(max_iter):
            price_func = BSModel.call_price if is_call else BSModel.put_price
            price = price_func(S, K, T, r, sigma)
            vega = BSModel.vega(S, K, T, r, sigma)
            if vega < 1e-10:
                return None
            diff = price - market_price
            if abs(diff) < tol:
                return sigma
            sigma -= diff / vega
            if sigma <= 0.001:
                return None
        return sigma

    @staticmethod
    def delta_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return 1.0 if S > K else 0.0
        return norm.cdf(BSModel.d1(S, K, T, r, sigma))

    @staticmethod
    def delta_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return -1.0 if S < K else 0.0
        return norm.cdf(BSModel.d1(S, K, T, r, sigma)) - 1

    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = BSModel.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * math.sqrt(T))

    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = BSModel.d1(S, K, T, r, sigma)
        return S * norm.pdf(d1) * math.sqrt(T) / 100

    @staticmethod
    def theta_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = BSModel.d1(S, K, T, r, sigma)
        d2 = BSModel.d2(S, K, T, r, sigma)
        term1 = -S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
        term2 = -r * K * math.exp(-r * T) * norm.cdf(d2)
        return (term1 + term2) / 365

    @staticmethod
    def theta_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = BSModel.d1(S, K, T, r, sigma)
        d2 = BSModel.d2(S, K, T, r, sigma)
        term1 = -S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
        term2 = r * K * math.exp(-r * T) * norm.cdf(-d2)
        return (term1 + term2) / 365


# ============================================================
# 数据结构
# ============================================================

class OptionType(Enum):
    CALL = "认购"
    PUT = "认沽"


class SignalType(Enum):
    NO_SIGNAL = "无信号"
    ENTRY = "开仓"
    EXIT_TAKE_PROFIT = "止盈平仓"
    EXIT_LEG_PROFIT = "单腿止盈"
    EXIT_TIME_STOP = "时间止损"
    EXIT_IV_PULLBACK = "波动率回落平仓"
    EXIT_EVENT_PASSED = "事件结束平仓"
    EXIT_MAX_LOSS = "最大亏损止损"


@dataclass
class CnMarketSnapshot:
    """A股市场快照"""
    timestamp: str                  # 时间戳 (YYYY-MM-DD HH:MM)
    underlying: str                 # 标的代码 (如 "50ETF")
    underlying_price: float         # ETF现价
    atm_call_price: float           # 平值认购权利金
    atm_put_price: float            # 平值认沽权利金
    atm_strike: float               # 平值行权价
    iv_call: float                  # 认购隐含波动率
    iv_put: float                   # 认沽隐含波动率
    dte: int                        # 剩余到期天数
    days_to_next_event: int         # 距下一重大事件天数
    next_event_type: str            # 事件类型
    next_event_desc: str            # 事件描述
    iv_percentile: float            # 隐波历史分位数 (0-100)
    # A股特有字段
    call_volume: int = 0            # 认购成交量
    put_volume: int = 0             # 认沽成交量
    pcr_volume: float = 0.0         # 成交量PCR（Put/Call Ratio）
    pcr_oi: float = 0.0             # 持仓量PCR


@dataclass
class CnStraddlePosition:
    """A股跨式持仓"""
    entry_time: str
    underlying: str
    strike: float
    call_premium: float             # 买入认购权利金（元/份）
    put_premium: float              # 买入认沽权利金（元/份）
    contracts: int                  # 合约张数
    multiplier: int                 # 合约乘数
    total_cost: float               # 总成本（权利金*乘数*张数 + 手续费）
    entry_iv: float                 # 入场隐波均值
    entry_price: float              # 入场ETF价格
    peak_iv: float = 0.0            # 持仓期间最高隐波
    event_type: str = ""
    event_desc: str = ""


@dataclass
class CnTradeResult:
    """A股交易结果"""
    entry_time: str
    exit_time: str
    underlying: str
    strike: float
    contracts: int
    total_cost: float               # 总投入（元）
    call_pnl: float                 # 认购盈亏（元）
    put_pnl: float                  # 认沽盈亏（元）
    total_pnl: float                # 总盈亏（元）
    return_pct: float               # 收益率（百分比）
    return_multiple: float          # 收益倍数
    exit_reason: str
    entry_iv: float
    exit_iv: float
    entry_price: float
    exit_price: float


# ============================================================
# A股跨式策略引擎
# ============================================================

class CnStraddleStrategy:
    """
    A股ETF期权双买（跨式）策略引擎

    工作流程:
    1. check_entry_signal()  - 检查入场条件
    2. open_straddle()       - 开仓
    3. check_exit_signal()   - 检查出场条件（支持T+0日内）
    4. close_straddle()      - 平仓
    """

    def __init__(self, account_balance: float, underlying: str = DEFAULT_UNDERLYING):
        self.account_balance = account_balance
        self.initial_balance = account_balance
        self.underlying = underlying
        self.underlying_info = UNDERLYING_MAP[underlying]
        self.multiplier = self.underlying_info["multiplier"]
        self.position: Optional[CnStraddlePosition] = None
        self.trade_history: list[CnTradeResult] = []
        self.r = RISK_FREE_RATE

    def _total_fee_per_contract(self) -> float:
        """单张合约单边手续费（开仓或平仓）"""
        return COMMISSION_PER_CONTRACT + EXCHANGE_FEE

    def check_entry_signal(self, snapshot: CnMarketSnapshot) -> SignalType:
        """
        检查入场信号

        入场条件（全部满足）：
        1. 无当前持仓
        2. IV分位数低于阈值（低波环境）
        3. IV绝对值在合理范围
        4. 距重大事件 <= MAX_DAYS_TO_EVENT
        5. DTE在合理范围
        6. Greeks满足最低要求
        7. 成交量PCR无极端值（可选）
        """
        if self.position is not None:
            return SignalType.NO_SIGNAL

        # 条件1: IV分位数
        if snapshot.iv_percentile > IV_PERCENTILE_THRESHOLD:
            return SignalType.NO_SIGNAL

        # 条件2: IV绝对值范围
        avg_iv = (snapshot.iv_call + snapshot.iv_put) / 2
        if avg_iv > IV_ABSOLUTE_MAX or avg_iv < IV_ABSOLUTE_MIN:
            return SignalType.NO_SIGNAL

        # 条件3: 事件临近
        if snapshot.days_to_next_event > MAX_DAYS_TO_EVENT:
            return SignalType.NO_SIGNAL

        # 条件4: DTE范围
        if not (MIN_DTE <= snapshot.dte <= MAX_DTE):
            return SignalType.NO_SIGNAL

        # 条件5: Greeks
        T = snapshot.dte / 365
        vega = BSModel.vega(snapshot.underlying_price, snapshot.atm_strike, T, self.r, avg_iv)
        gamma = BSModel.gamma(snapshot.underlying_price, snapshot.atm_strike, T, self.r, avg_iv)

        if vega < MIN_VEGA_PER_CONTRACT:
            return SignalType.NO_SIGNAL
        if gamma < MIN_GAMMA_PER_CONTRACT:
            return SignalType.NO_SIGNAL

        return SignalType.ENTRY

    def calculate_position_size(self, snapshot: CnMarketSnapshot) -> int:
        """
        计算开仓张数

        A股期权：每张 = 权利金 * 合约乘数(10000) + 手续费
        """
        straddle_premium = snapshot.atm_call_price + snapshot.atm_put_price
        max_capital = self.account_balance * MAX_POSITION_RATIO

        # 每张成本 = (认购权利金 + 认沽权利金) * 乘数 * (1+滑点) + 2张手续费
        cost_per_contract = (
            straddle_premium * self.multiplier * (1 + SLIPPAGE_PCT)
            + 2 * self._total_fee_per_contract()
        )

        if cost_per_contract <= 0:
            return 1

        contracts = int(max_capital / cost_per_contract)
        return max(contracts, 1)

    def open_straddle(self, snapshot: CnMarketSnapshot) -> Optional[CnStraddlePosition]:
        """开仓：同时买入平值认购 + 平值认沽"""
        if self.position is not None:
            return None

        contracts = self.calculate_position_size(snapshot)
        straddle_premium = snapshot.atm_call_price + snapshot.atm_put_price

        total_cost = (
            straddle_premium * self.multiplier * contracts * (1 + SLIPPAGE_PCT)
            + 2 * contracts * self._total_fee_per_contract()
        )

        avg_iv = (snapshot.iv_call + snapshot.iv_put) / 2

        self.position = CnStraddlePosition(
            entry_time=snapshot.timestamp,
            underlying=snapshot.underlying,
            strike=snapshot.atm_strike,
            call_premium=snapshot.atm_call_price,
            put_premium=snapshot.atm_put_price,
            contracts=contracts,
            multiplier=self.multiplier,
            total_cost=total_cost,
            entry_iv=avg_iv,
            entry_price=snapshot.underlying_price,
            peak_iv=avg_iv,
            event_type=snapshot.next_event_type,
            event_desc=snapshot.next_event_desc,
        )
        return self.position

    def check_exit_signal(self, snapshot: CnMarketSnapshot,
                          days_since_event: int = -1) -> SignalType:
        """
        检查出场信号（支持T+0日内检查）

        出场条件（满足任一）：
        1. 组合止盈
        2. 单腿暴利止盈
        3. 时间止损
        4. IV回落止损
        5. 事件结束
        6. 最大亏损止损
        """
        if self.position is None:
            return SignalType.NO_SIGNAL

        pos = self.position
        T = snapshot.dte / 365
        avg_iv = (snapshot.iv_call + snapshot.iv_put) / 2

        # 更新峰值IV
        pos.peak_iv = max(pos.peak_iv, avg_iv)

        # 计算当前组合价值
        current_call = BSModel.call_price(
            snapshot.underlying_price, pos.strike, T, self.r, snapshot.iv_call
        )
        current_put = BSModel.put_price(
            snapshot.underlying_price, pos.strike, T, self.r, snapshot.iv_put
        )

        current_value = (current_call + current_put) * self.multiplier * pos.contracts
        pnl = current_value - pos.total_cost

        # 条件1: 组合止盈
        if pnl >= pos.total_cost * TAKE_PROFIT_MULTIPLIER:
            return SignalType.EXIT_TAKE_PROFIT

        # 条件2: 单腿止盈（认沽暴利）
        put_value = current_put * self.multiplier * pos.contracts
        put_cost = pos.put_premium * self.multiplier * pos.contracts
        if put_cost > 0 and put_value / put_cost >= LEG_TAKE_PROFIT_MULTIPLIER:
            return SignalType.EXIT_LEG_PROFIT

        # 条件3: 单腿止盈（认购暴利）
        call_value = current_call * self.multiplier * pos.contracts
        call_cost = pos.call_premium * self.multiplier * pos.contracts
        if call_cost > 0 and call_value / call_cost >= LEG_TAKE_PROFIT_MULTIPLIER:
            return SignalType.EXIT_LEG_PROFIT

        # 条件4: 时间止损
        if snapshot.dte <= TIME_STOP_DTE:
            return SignalType.EXIT_TIME_STOP

        # 条件5: IV回落止损
        if pos.peak_iv > pos.entry_iv * 1.3:
            pullback = (pos.peak_iv - avg_iv) / pos.peak_iv
            if pullback >= IV_PULLBACK_THRESHOLD:
                return SignalType.EXIT_IV_PULLBACK

        # 条件6: 事件结束
        if days_since_event >= MAX_HOLD_DAYS_AFTER_EVENT:
            return SignalType.EXIT_EVENT_PASSED

        # 条件7: 最大亏损止损
        if pos.total_cost > 0 and pnl / pos.total_cost <= MAX_LOSS_RATIO:
            return SignalType.EXIT_MAX_LOSS

        return SignalType.NO_SIGNAL

    def close_straddle(self, snapshot: CnMarketSnapshot,
                       exit_reason: str) -> Optional[CnTradeResult]:
        """平仓"""
        if self.position is None:
            return None

        pos = self.position
        T = snapshot.dte / 365
        avg_iv = (snapshot.iv_call + snapshot.iv_put) / 2

        current_call = BSModel.call_price(
            snapshot.underlying_price, pos.strike, T, self.r, snapshot.iv_call
        )
        current_put = BSModel.put_price(
            snapshot.underlying_price, pos.strike, T, self.r, snapshot.iv_put
        )

        # 扣除平仓滑点和手续费
        call_proceeds_per = current_call * (1 - SLIPPAGE_PCT)
        put_proceeds_per = current_put * (1 - SLIPPAGE_PCT)
        close_fees = 2 * pos.contracts * self._total_fee_per_contract()

        call_pnl = (call_proceeds_per - pos.call_premium) * self.multiplier * pos.contracts
        put_pnl = (put_proceeds_per - pos.put_premium) * self.multiplier * pos.contracts
        total_pnl = call_pnl + put_pnl - close_fees

        return_pct = (total_pnl / pos.total_cost * 100) if pos.total_cost > 0 else 0
        return_multiple = total_pnl / pos.total_cost if pos.total_cost > 0 else 0

        result = CnTradeResult(
            entry_time=pos.entry_time,
            exit_time=snapshot.timestamp,
            underlying=pos.underlying,
            strike=pos.strike,
            contracts=pos.contracts,
            total_cost=pos.total_cost,
            call_pnl=call_pnl,
            put_pnl=put_pnl,
            total_pnl=total_pnl,
            return_pct=return_pct,
            return_multiple=return_multiple,
            exit_reason=exit_reason,
            entry_iv=pos.entry_iv,
            exit_iv=avg_iv,
            entry_price=pos.entry_price,
            exit_price=snapshot.underlying_price,
        )

        self.trade_history.append(result)
        self.account_balance += total_pnl
        self.position = None
        return result

    def get_current_greeks(self, snapshot: CnMarketSnapshot) -> dict:
        """获取当前持仓Greeks（每张 * 乘数 * 张数）"""
        if self.position is None:
            return {}

        pos = self.position
        T = snapshot.dte / 365
        avg_iv = (snapshot.iv_call + snapshot.iv_put) / 2

        call_delta = BSModel.delta_call(snapshot.underlying_price, pos.strike, T, self.r, snapshot.iv_call)
        put_delta = BSModel.delta_put(snapshot.underlying_price, pos.strike, T, self.r, snapshot.iv_put)
        gamma = BSModel.gamma(snapshot.underlying_price, pos.strike, T, self.r, avg_iv)
        vega = BSModel.vega(snapshot.underlying_price, pos.strike, T, self.r, avg_iv)
        theta_c = BSModel.theta_call(snapshot.underlying_price, pos.strike, T, self.r, snapshot.iv_call)
        theta_p = BSModel.theta_put(snapshot.underlying_price, pos.strike, T, self.r, snapshot.iv_put)

        n = pos.contracts * self.multiplier
        return {
            "net_delta": (call_delta + put_delta) * n,
            "gamma": gamma * 2 * n,
            "vega": vega * 2 * n,
            "theta": (theta_c + theta_p) * n,
            "delta_warning": abs(call_delta + put_delta) > MAX_ABS_DELTA,
            # 盈亏平衡点
            "upper_be": pos.strike + (pos.call_premium + pos.put_premium),
            "lower_be": pos.strike - (pos.call_premium + pos.put_premium),
        }

    def get_current_pnl(self, snapshot: CnMarketSnapshot) -> dict:
        """获取当前持仓盈亏"""
        if self.position is None:
            return {}

        pos = self.position
        T = snapshot.dte / 365

        current_call = BSModel.call_price(
            snapshot.underlying_price, pos.strike, T, self.r, snapshot.iv_call
        )
        current_put = BSModel.put_price(
            snapshot.underlying_price, pos.strike, T, self.r, snapshot.iv_put
        )

        n = pos.contracts * self.multiplier
        call_value = current_call * n
        put_value = current_put * n
        total_value = call_value + put_value
        total_pnl = total_value - pos.total_cost

        return {
            "call_value": call_value,
            "put_value": put_value,
            "total_value": total_value,
            "total_pnl": total_pnl,
            "return_pct": total_pnl / pos.total_cost * 100 if pos.total_cost > 0 else 0,
            "return_multiple": total_pnl / pos.total_cost if pos.total_cost > 0 else 0,
        }

    def summary(self) -> dict:
        """策略汇总"""
        if not self.trade_history:
            return {"total_trades": 0}

        wins = [t for t in self.trade_history if t.total_pnl > 0]
        losses = [t for t in self.trade_history if t.total_pnl <= 0]

        total_pnl = sum(t.total_pnl for t in self.trade_history)
        returns = [t.return_multiple for t in self.trade_history]
        avg_return = np.mean(returns)
        max_return = max(returns)
        max_loss = min(returns)

        return {
            "标的": self.underlying,
            "初始资金": f"{self.initial_balance:,.0f}",
            "当前资金": f"{self.account_balance:,.0f}",
            "总交易次数": len(self.trade_history),
            "盈利次数": len(wins),
            "亏损次数": len(losses),
            "胜率": f"{len(wins) / len(self.trade_history):.1%}",
            "累计盈亏": f"{total_pnl:+,.0f}",
            "累计收益率": f"{total_pnl / self.initial_balance:+.1%}",
            "平均收益倍数": f"{avg_return:+.2f}x",
            "最大单笔收益": f"{max_return:+.2f}x",
            "最大单笔亏损": f"{max_loss:+.2f}x",
            "盈亏比": (
                f"{np.mean([t.total_pnl for t in wins]) / abs(np.mean([t.total_pnl for t in losses])):.1f}:1"
                if losses and np.mean([t.total_pnl for t in losses]) != 0 else "N/A"
            ),
        }
