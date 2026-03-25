"""
黄金期权双买策略 (Gold Options Long Straddle)
=============================================

核心逻辑：
1. 在隐含波动率历史低位 + 重大事件前入场
2. 同时买入平值看涨和看跌期权
3. 通过 Vega（波动率上升）和 Gamma（方向爆发）获利
4. 最大亏损 = 总权利金支出，风险完全可控

收益来源：
- 60% 隐波拉升带来的估值修复（Vega收益）
- 40% 方向性爆发带来的Gamma暴击
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from scipy.stats import norm

from config import (
    COMMISSION_PER_CONTRACT,
    IV_ABSOLUTE_MAX,
    IV_PERCENTILE_THRESHOLD,
    IV_PULLBACK_THRESHOLD,
    LEG_TAKE_PROFIT_MULTIPLIER,
    MAX_ABS_DELTA,
    MAX_DAYS_TO_EVENT,
    MAX_DTE,
    MAX_HOLD_DAYS_AFTER_EVENT,
    MAX_LEG_RATIO,
    MAX_POSITION_RATIO,
    MIN_DTE,
    MIN_GAMMA_PER_CONTRACT,
    MIN_VEGA_PER_CONTRACT,
    RISK_FREE_RATE,
    SLIPPAGE_PCT,
    TAKE_PROFIT_MULTIPLIER,
    TIME_STOP_DTE,
)


# ============================================================
# Black-Scholes 定价模型
# ============================================================

class BSModel:
    """Black-Scholes 期权定价模型"""

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        return BSModel.d1(S, K, T, r, sigma) - sigma * math.sqrt(T)

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """计算看涨期权价格"""
        if T <= 0:
            return max(S - K, 0)
        d1 = BSModel.d1(S, K, T, r, sigma)
        d2 = BSModel.d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """计算看跌期权价格"""
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
        sigma = 0.3  # 初始猜测
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
            if sigma <= 0:
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
        if T <= 0:
            return 0.0
        d1 = BSModel.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * math.sqrt(T))

    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return 0.0
        d1 = BSModel.d1(S, K, T, r, sigma)
        return S * norm.pdf(d1) * math.sqrt(T) / 100  # 除以100，表示IV变动1%的影响

    @staticmethod
    def theta_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return 0.0
        d1 = BSModel.d1(S, K, T, r, sigma)
        d2 = BSModel.d2(S, K, T, r, sigma)
        term1 = -S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
        term2 = -r * K * math.exp(-r * T) * norm.cdf(d2)
        return (term1 + term2) / 365  # 每日Theta

    @staticmethod
    def theta_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
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
    CALL = "CALL"
    PUT = "PUT"


class SignalType(Enum):
    NO_SIGNAL = "NO_SIGNAL"
    ENTRY = "ENTRY"
    EXIT_TAKE_PROFIT = "EXIT_TP"
    EXIT_TIME_STOP = "EXIT_TIME"
    EXIT_IV_PULLBACK = "EXIT_IV"
    EXIT_EVENT_PASSED = "EXIT_EVENT"


@dataclass
class MarketSnapshot:
    """市场快照"""
    timestamp: str              # 时间戳
    gold_price: float           # 黄金现货/期货价格
    atm_call_price: float       # 平值看涨期权价格
    atm_put_price: float        # 平值看跌期权价格
    atm_strike: float           # 平值行权价
    iv_call: float              # 看涨隐含波动率
    iv_put: float               # 看跌隐含波动率
    dte: int                    # 剩余到期天数
    days_to_next_event: int     # 距下一重大事件天数
    next_event_type: str        # 下一重大事件类型
    iv_percentile: float        # 隐波历史分位数 (0-100)


@dataclass
class StraddlePosition:
    """双买持仓"""
    entry_time: str
    strike: float
    call_premium: float         # 买入时看涨权利金
    put_premium: float          # 买入时看跌权利金
    total_cost: float           # 总成本（含手续费）
    contracts: int              # 合约手数
    entry_iv: float             # 入场时隐波均值
    entry_gold_price: float     # 入场时金价
    peak_iv: float = 0.0        # 持仓期间最高隐波
    event_date: str = ""        # 对应事件日期
    event_type: str = ""        # 对应事件类型


@dataclass
class TradeResult:
    """交易结果"""
    entry_time: str
    exit_time: str
    strike: float
    contracts: int
    total_cost: float           # 总投入
    call_pnl: float             # 看涨盈亏
    put_pnl: float              # 看跌盈亏
    total_pnl: float            # 总盈亏
    return_multiple: float      # 收益倍数
    exit_reason: str            # 退出原因
    entry_iv: float
    exit_iv: float
    entry_price: float
    exit_price: float


# ============================================================
# 策略引擎
# ============================================================

class GoldStraddleStrategy:
    """
    黄金期权双买策略引擎

    工作流程:
    1. check_entry_signal() - 检查是否满足入场条件
    2. open_straddle()      - 开仓
    3. check_exit_signal()  - 检查是否满足出场条件
    4. close_straddle()     - 平仓
    """

    def __init__(self, account_balance: float):
        self.account_balance = account_balance
        self.position: Optional[StraddlePosition] = None
        self.trade_history: list[TradeResult] = []
        self.r = RISK_FREE_RATE

    def check_entry_signal(self, snapshot: MarketSnapshot) -> SignalType:
        """
        检查入场信号

        入场条件（必须全部满足）：
        1. 无当前持仓
        2. 隐波处于历史低位（分位数 < 阈值）
        3. 隐波绝对值低于上限
        4. 距离重大事件 <= MAX_DAYS_TO_EVENT 天
        5. 期权DTE在合理范围内
        6. Greeks满足最低要求
        """
        if self.position is not None:
            return SignalType.NO_SIGNAL

        # 条件1: 隐波分位数
        if snapshot.iv_percentile > IV_PERCENTILE_THRESHOLD:
            return SignalType.NO_SIGNAL

        # 条件2: 隐波绝对值
        avg_iv = (snapshot.iv_call + snapshot.iv_put) / 2
        if avg_iv > IV_ABSOLUTE_MAX:
            return SignalType.NO_SIGNAL

        # 条件3: 事件临近
        if snapshot.days_to_next_event > MAX_DAYS_TO_EVENT:
            return SignalType.NO_SIGNAL

        # 条件4: DTE范围
        if not (MIN_DTE <= snapshot.dte <= MAX_DTE):
            return SignalType.NO_SIGNAL

        # 条件5: Greeks检查
        T = snapshot.dte / 365
        vega = BSModel.vega(snapshot.gold_price, snapshot.atm_strike, T, self.r, avg_iv)
        gamma = BSModel.gamma(snapshot.gold_price, snapshot.atm_strike, T, self.r, avg_iv)

        if vega < MIN_VEGA_PER_CONTRACT:
            return SignalType.NO_SIGNAL
        if gamma < MIN_GAMMA_PER_CONTRACT:
            return SignalType.NO_SIGNAL

        return SignalType.ENTRY

    def calculate_position_size(self, snapshot: MarketSnapshot) -> int:
        """计算开仓手数"""
        straddle_cost = snapshot.atm_call_price + snapshot.atm_put_price
        max_capital = self.account_balance * MAX_POSITION_RATIO
        # 每手成本 = 权利金 + 手续费 + 滑点
        cost_per_contract = straddle_cost * (1 + SLIPPAGE_PCT) + 2 * COMMISSION_PER_CONTRACT
        contracts = int(max_capital / cost_per_contract)
        return max(contracts, 1)  # 至少1手

    def open_straddle(self, snapshot: MarketSnapshot) -> Optional[StraddlePosition]:
        """开仓：同时买入平值看涨和看跌"""
        if self.position is not None:
            return None

        contracts = self.calculate_position_size(snapshot)
        straddle_cost = snapshot.atm_call_price + snapshot.atm_put_price
        total_cost = contracts * (straddle_cost * (1 + SLIPPAGE_PCT) + 2 * COMMISSION_PER_CONTRACT)
        avg_iv = (snapshot.iv_call + snapshot.iv_put) / 2

        self.position = StraddlePosition(
            entry_time=snapshot.timestamp,
            strike=snapshot.atm_strike,
            call_premium=snapshot.atm_call_price,
            put_premium=snapshot.atm_put_price,
            total_cost=total_cost,
            contracts=contracts,
            entry_iv=avg_iv,
            entry_gold_price=snapshot.gold_price,
            peak_iv=avg_iv,
            event_date="",
            event_type=snapshot.next_event_type,
        )
        return self.position

    def check_exit_signal(self, snapshot: MarketSnapshot,
                          days_since_event: int = -1) -> SignalType:
        """
        检查出场信号

        出场条件（满足任一即可）：
        1. 组合盈利达到止盈倍数
        2. DTE低于时间止损阈值
        3. 隐波从高点回落超过阈值
        4. 事件已过且超过最大持仓天数
        """
        if self.position is None:
            return SignalType.NO_SIGNAL

        pos = self.position
        T = snapshot.dte / 365
        avg_iv = (snapshot.iv_call + snapshot.iv_put) / 2

        # 更新峰值隐波
        pos.peak_iv = max(pos.peak_iv, avg_iv)

        # 计算当前组合价值
        current_call = BSModel.call_price(
            snapshot.gold_price, pos.strike, T, self.r, snapshot.iv_call
        )
        current_put = BSModel.put_price(
            snapshot.gold_price, pos.strike, T, self.r, snapshot.iv_put
        )
        current_value = (current_call + current_put) * pos.contracts
        pnl = current_value - pos.total_cost

        # 条件1: 组合止盈
        if pnl >= pos.total_cost * TAKE_PROFIT_MULTIPLIER:
            return SignalType.EXIT_TAKE_PROFIT

        # 条件2: 单腿止盈（Put暴利场景）
        put_value = current_put * pos.contracts
        put_cost = pos.put_premium * pos.contracts
        if put_cost > 0 and put_value / put_cost >= LEG_TAKE_PROFIT_MULTIPLIER:
            return SignalType.EXIT_TAKE_PROFIT

        # 条件3: 单腿止盈（Call暴利场景）
        call_value = current_call * pos.contracts
        call_cost = pos.call_premium * pos.contracts
        if call_cost > 0 and call_value / call_cost >= LEG_TAKE_PROFIT_MULTIPLIER:
            return SignalType.EXIT_TAKE_PROFIT

        # 条件4: 时间止损
        if snapshot.dte <= TIME_STOP_DTE:
            return SignalType.EXIT_TIME_STOP

        # 条件5: 隐波回落止损
        if pos.peak_iv > pos.entry_iv * 1.5:  # 隐波曾经大幅上升过
            pullback = (pos.peak_iv - avg_iv) / pos.peak_iv
            if pullback >= IV_PULLBACK_THRESHOLD:
                return SignalType.EXIT_IV_PULLBACK

        # 条件6: 事件已过
        if days_since_event >= MAX_HOLD_DAYS_AFTER_EVENT:
            return SignalType.EXIT_EVENT_PASSED

        return SignalType.NO_SIGNAL

    def close_straddle(self, snapshot: MarketSnapshot,
                       exit_reason: str) -> Optional[TradeResult]:
        """平仓"""
        if self.position is None:
            return None

        pos = self.position
        T = snapshot.dte / 365
        avg_iv = (snapshot.iv_call + snapshot.iv_put) / 2

        # 计算平仓价值
        current_call = BSModel.call_price(
            snapshot.gold_price, pos.strike, T, self.r, snapshot.iv_call
        )
        current_put = BSModel.put_price(
            snapshot.gold_price, pos.strike, T, self.r, snapshot.iv_put
        )

        # 扣除滑点和手续费
        call_proceeds = current_call * (1 - SLIPPAGE_PCT) - COMMISSION_PER_CONTRACT
        put_proceeds = current_put * (1 - SLIPPAGE_PCT) - COMMISSION_PER_CONTRACT

        call_pnl = (call_proceeds - pos.call_premium) * pos.contracts
        put_pnl = (put_proceeds - pos.put_premium) * pos.contracts
        total_pnl = call_pnl + put_pnl

        result = TradeResult(
            entry_time=pos.entry_time,
            exit_time=snapshot.timestamp,
            strike=pos.strike,
            contracts=pos.contracts,
            total_cost=pos.total_cost,
            call_pnl=call_pnl,
            put_pnl=put_pnl,
            total_pnl=total_pnl,
            return_multiple=total_pnl / pos.total_cost if pos.total_cost > 0 else 0,
            exit_reason=exit_reason,
            entry_iv=pos.entry_iv,
            exit_iv=avg_iv,
            entry_price=pos.entry_gold_price,
            exit_price=snapshot.gold_price,
        )

        self.trade_history.append(result)
        self.account_balance += total_pnl
        self.position = None
        return result

    def get_current_greeks(self, snapshot: MarketSnapshot) -> dict:
        """获取当前持仓的希腊字母"""
        if self.position is None:
            return {}

        pos = self.position
        T = snapshot.dte / 365
        avg_iv = (snapshot.iv_call + snapshot.iv_put) / 2

        call_delta = BSModel.delta_call(snapshot.gold_price, pos.strike, T, self.r, snapshot.iv_call)
        put_delta = BSModel.delta_put(snapshot.gold_price, pos.strike, T, self.r, snapshot.iv_put)
        gamma = BSModel.gamma(snapshot.gold_price, pos.strike, T, self.r, avg_iv)
        vega = BSModel.vega(snapshot.gold_price, pos.strike, T, self.r, avg_iv)
        theta_call = BSModel.theta_call(snapshot.gold_price, pos.strike, T, self.r, snapshot.iv_call)
        theta_put = BSModel.theta_put(snapshot.gold_price, pos.strike, T, self.r, snapshot.iv_put)

        return {
            "net_delta": (call_delta + put_delta) * pos.contracts,
            "gamma": gamma * 2 * pos.contracts,  # Call和Put的Gamma相同
            "vega": vega * 2 * pos.contracts,
            "theta": (theta_call + theta_put) * pos.contracts,
            "delta_warning": abs(call_delta + put_delta) > MAX_ABS_DELTA,
        }

    def summary(self) -> dict:
        """策略汇总统计"""
        if not self.trade_history:
            return {"total_trades": 0}

        wins = [t for t in self.trade_history if t.total_pnl > 0]
        losses = [t for t in self.trade_history if t.total_pnl <= 0]

        total_pnl = sum(t.total_pnl for t in self.trade_history)
        avg_return = np.mean([t.return_multiple for t in self.trade_history])
        max_return = max(t.return_multiple for t in self.trade_history)
        max_loss = min(t.return_multiple for t in self.trade_history)

        return {
            "total_trades": len(self.trade_history),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self.trade_history),
            "total_pnl": total_pnl,
            "avg_return_multiple": avg_return,
            "max_return_multiple": max_return,
            "max_loss_multiple": max_loss,
            "profit_factor": (
                sum(t.total_pnl for t in wins) / abs(sum(t.total_pnl for t in losses))
                if losses else float("inf")
            ),
        }
