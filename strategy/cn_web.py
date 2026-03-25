"""
A股ETF期权双买策略 - Web实盘看板
=================================
功能：
1. 实时市场状态面板
2. 入场/出场信号判断 + 操作指示
3. 持仓盈亏实时监控
4. 事件日历预警
5. 历史交易记录
6. 声音+弹窗预警
"""

import json
import math
import threading
import time
from datetime import date, datetime, timedelta

import numpy as np
from flask import Flask, jsonify, render_template, request

from cn_config import (
    DEFAULT_UNDERLYING,
    EVENT_TYPES,
    IV_ABSOLUTE_MAX,
    IV_PERCENTILE_THRESHOLD,
    MAX_DAYS_TO_EVENT,
    MAX_DTE,
    MAX_POSITION_RATIO,
    MIN_DTE,
    RISK_FREE_RATE,
    UNDERLYING_MAP,
)
from cn_straddle import (
    BSModel,
    CnMarketSnapshot,
    CnStraddleStrategy,
    CnStraddlePosition,
    SignalType,
)

app = Flask(__name__)

# ============================================================
# 全局状态
# ============================================================

class AppState:
    """应用全局状态"""
    def __init__(self):
        self.strategy = CnStraddleStrategy(200000.0, DEFAULT_UNDERLYING)
        self.latest_snapshot = None
        self.alerts = []          # 预警消息列表
        self.trade_log = []       # 交易日志
        self.market_history = []  # 市场数据历史
        self.auto_refresh = True
        self.refresh_interval = 5  # 秒
        # 模拟市场数据（实盘时替换为真实数据源）
        self.sim_day = 0
        self.sim_running = False

state = AppState()


# ============================================================
# 事件日历（2025年已知事件）
# ============================================================

UPCOMING_EVENTS = [
    {"date": "2025-03-27", "type": "MLF", "desc": "3月MLF操作结果公布", "importance": "中"},
    {"date": "2025-03-31", "type": "PMI", "desc": "3月官方制造业PMI", "importance": "高"},
    {"date": "2025-04-01", "type": "PMI", "desc": "3月财新制造业PMI", "importance": "中"},
    {"date": "2025-04-10", "type": "CPI_CN", "desc": "3月CPI/PPI数据", "importance": "高"},
    {"date": "2025-04-14", "type": "TRADE_DATA", "desc": "3月进出口数据", "importance": "中"},
    {"date": "2025-04-16", "type": "GDP_CN", "desc": "Q1 GDP数据", "importance": "高"},
    {"date": "2025-04-21", "type": "LPR", "desc": "4月LPR报价", "importance": "高"},
    {"date": "2025-04-25", "type": "POLITBURO", "desc": "4月政治局会议（预计）", "importance": "高"},
    {"date": "2025-05-07", "type": "FOMC", "desc": "5月FOMC会议", "importance": "高"},
    {"date": "2025-05-12", "type": "CPI_CN", "desc": "4月CPI/PPI数据", "importance": "中"},
    {"date": "2025-05-20", "type": "LPR", "desc": "5月LPR报价", "importance": "高"},
    {"date": "2025-05-30", "type": "PMI", "desc": "5月官方制造业PMI", "importance": "高"},
    {"date": "2025-06-18", "type": "FOMC", "desc": "6月FOMC会议", "importance": "高"},
    {"date": "2025-06-20", "type": "LPR", "desc": "6月LPR报价", "importance": "高"},
    {"date": "2025-07-15", "type": "GDP_CN", "desc": "Q2 GDP数据", "importance": "高"},
    {"date": "2025-07-21", "type": "LPR", "desc": "7月LPR报价", "importance": "高"},
    {"date": "2025-07-25", "type": "POLITBURO", "desc": "7月政治局会议（预计）", "importance": "高"},
]


def get_upcoming_events(days_ahead=30):
    """获取未来N天内的事件"""
    today = date.today()
    upcoming = []
    for evt in UPCOMING_EVENTS:
        evt_date = date.fromisoformat(evt["date"])
        days_until = (evt_date - today).days
        if 0 <= days_until <= days_ahead:
            upcoming.append({
                **evt,
                "days_until": days_until,
                "is_imminent": days_until <= MAX_DAYS_TO_EVENT,
            })
    return sorted(upcoming, key=lambda x: x["days_until"])


def get_nearest_event():
    """获取最近的事件"""
    events = get_upcoming_events(60)
    if events:
        return events[0]
    return None


# ============================================================
# 模拟市场数据生成（实盘替换此部分）
# ============================================================

def generate_simulated_snapshot(underlying="50ETF"):
    """
    生成模拟市场快照
    实盘时替换为：
    - 从券商API获取ETF实时价格
    - 从期权行情获取认购/认沽实时价格和IV
    - 计算IV历史分位数
    """
    np.random.seed(int(time.time()) % 10000)

    info = UNDERLYING_MAP[underlying]
    base_prices = {"50ETF": 2.75, "300ETF_SH": 3.85, "300ETF_SZ": 3.85,
                   "500ETF": 5.80, "1000ETF": 1.95}
    base = base_prices.get(underlying, 2.75)

    # 模拟价格波动
    price = base * (1 + np.random.normal(0, 0.008))
    price = round(price, 3)

    # 模拟IV（在低波区间波动）
    iv_base = np.random.uniform(0.12, 0.22)
    iv_call = round(iv_base + np.random.normal(0, 0.005), 4)
    iv_put = round(iv_call * 1.05, 4)  # 认沽偏高

    # ATM行权价
    strike_interval = info["strike_interval"]
    strike = round(price / strike_interval) * strike_interval

    # DTE（模拟当月合约）
    today = date.today()
    # 期权到期日大约是每月第四个周三
    month_end = today.replace(day=28)
    dte = max((month_end - today).days, 5)
    if dte < 3:
        dte += 28  # 切换到下月

    T = dte / 365
    call_px = BSModel.call_price(price, strike, T, RISK_FREE_RATE, iv_call)
    put_px = BSModel.put_price(price, strike, T, RISK_FREE_RATE, iv_put)

    # IV分位（模拟）
    iv_pct = np.random.uniform(8, 40)

    # 最近事件
    nearest = get_nearest_event()
    days_to_event = nearest["days_until"] if nearest else 99
    event_type = nearest["type"] if nearest else "NONE"
    event_desc = nearest["desc"] if nearest else "无近期事件"

    snapshot = CnMarketSnapshot(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        underlying=underlying,
        underlying_price=price,
        atm_call_price=round(call_px, 4),
        atm_put_price=round(put_px, 4),
        atm_strike=strike,
        iv_call=iv_call,
        iv_put=iv_put,
        dte=dte,
        days_to_next_event=days_to_event,
        next_event_type=event_type,
        next_event_desc=event_desc,
        iv_percentile=round(iv_pct, 1),
    )

    return snapshot


def check_and_alert(snapshot):
    """检查预警条件"""
    alerts = []
    avg_iv = (snapshot.iv_call + snapshot.iv_put) / 2

    # 1. 低IV预警
    if snapshot.iv_percentile <= IV_PERCENTILE_THRESHOLD:
        alerts.append({
            "level": "info",
            "title": "低波动率环境",
            "msg": f"IV分位={snapshot.iv_percentile:.0f}% (阈值{IV_PERCENTILE_THRESHOLD}%)，关注入场机会",
            "time": snapshot.timestamp,
        })

    # 2. 事件临近预警
    if snapshot.days_to_next_event <= MAX_DAYS_TO_EVENT:
        alerts.append({
            "level": "warning",
            "title": f"事件临近：{snapshot.next_event_type}",
            "msg": f"{snapshot.next_event_desc}，距今{snapshot.days_to_next_event}天",
            "time": snapshot.timestamp,
        })

    # 3. 入场信号
    signal = state.strategy.check_entry_signal(snapshot)
    if signal == SignalType.ENTRY:
        contracts = state.strategy.calculate_position_size(snapshot)
        cost = (snapshot.atm_call_price + snapshot.atm_put_price) * 10000 * contracts
        alerts.append({
            "level": "critical",
            "title": "入场信号触发！",
            "msg": (
                f"建议买入 {snapshot.underlying} 跨式组合\n"
                f"行权价: {snapshot.atm_strike:.2f}\n"
                f"认购: {snapshot.atm_call_price:.4f} + 认沽: {snapshot.atm_put_price:.4f}\n"
                f"建议张数: {contracts}张\n"
                f"预计成本: {cost:,.0f}元\n"
                f"触发事件: {snapshot.next_event_desc}"
            ),
            "time": snapshot.timestamp,
        })

    # 4. 出场信号（有持仓时）
    if state.strategy.position is not None:
        exit_signal = state.strategy.check_exit_signal(snapshot)
        if exit_signal != SignalType.NO_SIGNAL:
            pnl_info = state.strategy.get_current_pnl(snapshot)
            alerts.append({
                "level": "critical",
                "title": f"出场信号：{exit_signal.value}",
                "msg": (
                    f"建议立即平仓！\n"
                    f"当前浮盈: {pnl_info['total_pnl']:+,.0f}元 ({pnl_info['return_pct']:+.1f}%)\n"
                    f"退出原因: {exit_signal.value}"
                ),
                "time": snapshot.timestamp,
            })

    # 5. 持仓Delta偏移预警
    if state.strategy.position is not None:
        greeks = state.strategy.get_current_greeks(snapshot)
        if greeks.get("delta_warning"):
            alerts.append({
                "level": "warning",
                "title": "Delta偏移预警",
                "msg": f"组合Net Delta={greeks['net_delta']:.2f}，方向敞口过大",
                "time": snapshot.timestamp,
            })

    return alerts


# ============================================================
# Flask 路由
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/market")
def api_market():
    """获取当前市场数据"""
    underlying = request.args.get("underlying", DEFAULT_UNDERLYING)
    snapshot = generate_simulated_snapshot(underlying)
    state.latest_snapshot = snapshot

    # 检查预警
    new_alerts = check_and_alert(snapshot)
    state.alerts = new_alerts + state.alerts[:50]  # 保留最近50条

    # 入场信号判断
    entry_signal = state.strategy.check_entry_signal(snapshot)
    entry_conditions = {
        "iv_percentile_ok": snapshot.iv_percentile <= IV_PERCENTILE_THRESHOLD,
        "iv_absolute_ok": (snapshot.iv_call + snapshot.iv_put) / 2 <= IV_ABSOLUTE_MAX,
        "event_near_ok": snapshot.days_to_next_event <= MAX_DAYS_TO_EVENT,
        "dte_ok": MIN_DTE <= snapshot.dte <= MAX_DTE,
        "signal": entry_signal.value,
        "can_enter": entry_signal == SignalType.ENTRY,
    }

    # 持仓信息
    position_info = None
    exit_info = None
    if state.strategy.position is not None:
        pnl = state.strategy.get_current_pnl(snapshot)
        greeks = state.strategy.get_current_greeks(snapshot)
        pos = state.strategy.position
        position_info = {
            "entry_time": pos.entry_time,
            "strike": pos.strike,
            "contracts": pos.contracts,
            "total_cost": pos.total_cost,
            "call_premium": pos.call_premium,
            "put_premium": pos.put_premium,
            "entry_iv": pos.entry_iv,
            "entry_price": pos.entry_price,
            "event_type": pos.event_type,
            **pnl,
            **greeks,
        }
        exit_signal = state.strategy.check_exit_signal(snapshot)
        exit_info = {
            "signal": exit_signal.value,
            "should_exit": exit_signal != SignalType.NO_SIGNAL,
        }

    return jsonify({
        "snapshot": {
            "timestamp": snapshot.timestamp,
            "underlying": snapshot.underlying,
            "underlying_price": snapshot.underlying_price,
            "atm_strike": snapshot.atm_strike,
            "atm_call_price": snapshot.atm_call_price,
            "atm_put_price": snapshot.atm_put_price,
            "straddle_price": round(snapshot.atm_call_price + snapshot.atm_put_price, 4),
            "iv_call": snapshot.iv_call,
            "iv_put": snapshot.iv_put,
            "iv_avg": round((snapshot.iv_call + snapshot.iv_put) / 2, 4),
            "dte": snapshot.dte,
            "days_to_event": snapshot.days_to_next_event,
            "event_type": snapshot.next_event_type,
            "event_desc": snapshot.next_event_desc,
            "iv_percentile": snapshot.iv_percentile,
        },
        "entry_conditions": entry_conditions,
        "position": position_info,
        "exit_info": exit_info,
        "account": {
            "balance": state.strategy.account_balance,
            "initial": state.strategy.initial_balance,
            "pnl": state.strategy.account_balance - state.strategy.initial_balance,
            "total_trades": len(state.strategy.trade_history),
        },
    })


@app.route("/api/alerts")
def api_alerts():
    """获取预警列表"""
    return jsonify({"alerts": state.alerts[:20]})


@app.route("/api/events")
def api_events():
    """获取事件日历"""
    return jsonify({"events": get_upcoming_events(60)})


@app.route("/api/open", methods=["POST"])
def api_open():
    """手动开仓"""
    if state.latest_snapshot is None:
        return jsonify({"error": "无市场数据，请先刷新"}), 400
    if state.strategy.position is not None:
        return jsonify({"error": "已有持仓，不可重复开仓"}), 400

    pos = state.strategy.open_straddle(state.latest_snapshot)
    if pos is None:
        return jsonify({"error": "开仓失败"}), 400

    log_entry = {
        "time": pos.entry_time,
        "action": "开仓",
        "underlying": pos.underlying,
        "strike": pos.strike,
        "contracts": pos.contracts,
        "cost": pos.total_cost,
        "iv": pos.entry_iv,
        "price": pos.entry_price,
    }
    state.trade_log.append(log_entry)

    return jsonify({
        "success": True,
        "msg": f"已开仓 {pos.contracts}张 行权价={pos.strike:.2f}",
        "position": log_entry,
    })


@app.route("/api/close", methods=["POST"])
def api_close():
    """手动平仓"""
    if state.latest_snapshot is None:
        return jsonify({"error": "无市场数据"}), 400
    if state.strategy.position is None:
        return jsonify({"error": "无持仓"}), 400

    reason = request.json.get("reason", "手动平仓") if request.is_json else "手动平仓"
    trade = state.strategy.close_straddle(state.latest_snapshot, reason)
    if trade is None:
        return jsonify({"error": "平仓失败"}), 400

    log_entry = {
        "time": trade.exit_time,
        "action": "平仓",
        "underlying": trade.underlying,
        "strike": trade.strike,
        "contracts": trade.contracts,
        "pnl": trade.total_pnl,
        "return_pct": trade.return_pct,
        "reason": trade.exit_reason,
    }
    state.trade_log.append(log_entry)

    return jsonify({
        "success": True,
        "msg": f"已平仓 盈亏={trade.total_pnl:+,.0f}元 ({trade.return_pct:+.1f}%)",
        "trade": log_entry,
    })


@app.route("/api/trades")
def api_trades():
    """获取交易历史"""
    history = []
    for t in state.strategy.trade_history:
        history.append({
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "underlying": t.underlying,
            "strike": t.strike,
            "contracts": t.contracts,
            "total_cost": t.total_cost,
            "call_pnl": t.call_pnl,
            "put_pnl": t.put_pnl,
            "total_pnl": t.total_pnl,
            "return_pct": t.return_pct,
            "exit_reason": t.exit_reason,
            "entry_iv": t.entry_iv,
            "exit_iv": t.exit_iv,
        })
    summary = state.strategy.summary()
    return jsonify({"trades": history, "summary": summary, "log": state.trade_log[-30:]})


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    """查看/修改策略参数"""
    if request.method == "GET":
        return jsonify({
            "account_balance": state.strategy.account_balance,
            "underlying": state.strategy.underlying,
            "iv_percentile_threshold": IV_PERCENTILE_THRESHOLD,
            "iv_absolute_max": IV_ABSOLUTE_MAX,
            "max_days_to_event": MAX_DAYS_TO_EVENT,
            "min_dte": MIN_DTE,
            "max_dte": MAX_DTE,
            "max_position_ratio": MAX_POSITION_RATIO,
            "available_underlyings": list(UNDERLYING_MAP.keys()),
        })

    if request.method == "POST":
        data = request.json or {}
        if "underlying" in data and data["underlying"] in UNDERLYING_MAP:
            state.strategy = CnStraddleStrategy(
                state.strategy.account_balance, data["underlying"]
            )
        if "account_balance" in data:
            state.strategy.account_balance = float(data["account_balance"])
            state.strategy.initial_balance = float(data["account_balance"])
        return jsonify({"success": True})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """重置策略状态"""
    underlying = state.strategy.underlying
    balance = state.strategy.initial_balance
    state.strategy = CnStraddleStrategy(balance, underlying)
    state.alerts = []
    state.trade_log = []
    return jsonify({"success": True, "msg": "策略已重置"})


# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("A股ETF期权双买策略 - 实盘看板")
    print("=" * 50)
    print("启动中...")
    print("浏览器访问: http://localhost:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False)
