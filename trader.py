"""
Prosperity 4 — Trader V3
=========================

Minimal improvements over V1:
  1. vwap_mid instead of simple mid (better FV when book is asymmetric)
  2. Multi-level take: walk the book as long as levels are beyond FV ± edge
  3. Pepper INV_SKEW = 0.08 (best sharpe from prior tuning)

No autocorrelation, no regime detection, no median anchors. Stays simple.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import json
import math


# =============================================================================
# Helpers
# =============================================================================

def best_bid(depth: OrderDepth):
    if not depth.buy_orders:
        return None, 0
    px = max(depth.buy_orders.keys())
    return px, depth.buy_orders[px]

def best_ask(depth: OrderDepth):
    if not depth.sell_orders:
        return None, 0
    px = min(depth.sell_orders.keys())
    return px, -depth.sell_orders[px]

def simple_mid(depth: OrderDepth):
    bb, _ = best_bid(depth)
    ba, _ = best_ask(depth)
    if bb is None or ba is None:
        return None
    return (bb + ba) / 2.0

def vwap_mid(depth: OrderDepth):
    bb, bv = best_bid(depth)
    ba, av = best_ask(depth)
    if bb is None or ba is None:
        return None
    if bv + av == 0:
        return (bb + ba) / 2.0
    return (bb * av + ba * bv) / (bv + av)


# =============================================================================
# Parameters
# =============================================================================

class PepperParams:
    SYMBOL         = "INTARIAN_PEPPER_ROOT"
    LIMIT          = 80
    EWMA_ALPHA     = 0.15
    TAKE_EDGE      = 1.0
    MAKE_EDGE      = 1.0
    INV_SKEW       = 0.08      # <-- tuned from 0.05
    MAX_ORDER      = 25
    MIN_WARMUP     = 10
    SPREAD_FLOOR   = 1

class OsmiumParams:
    SYMBOL         = "ASH_COATED_OSMIUM"
    LIMIT          = 80
    EWMA_ALPHA     = 0.25
    TAKE_EDGE_K    = 0.9       # <-- tuned from 1.2
    MAKE_EDGE_K    = 1.6
    MIN_EDGE       = 1.5
    INV_SKEW_K     = 0.04
    MAX_ORDER      = 20
    VOL_WINDOW     = 30
    MOMENTUM_WIN   = 5
    MOMENTUM_BLOCK = 2.0
    MIN_WARMUP     = 15
    SPREAD_FLOOR   = 1


# =============================================================================
# Trader
# =============================================================================

class Trader:
    def __init__(self):
        self.state = {
            PepperParams.SYMBOL: {"fv": None, "ticks": 0, "mid_hist": []},
            OsmiumParams.SYMBOL: {"fv": None, "ticks": 0, "mid_hist": []},
        }

    def _load_state(self, traderData: str):
        if not traderData:
            return
        try:
            loaded = json.loads(traderData)
            for k, v in loaded.items():
                if k in self.state:
                    self.state[k].update(v)
        except Exception:
            pass

    def _dump_state(self) -> str:
        for k in self.state:
            hist = self.state[k].get("mid_hist", [])
            if len(hist) > 200:
                self.state[k]["mid_hist"] = hist[-200:]
        try:
            return json.dumps(self.state)
        except Exception:
            return ""

    def _update_fv(self, symbol: str, mid: float, alpha: float):
        st = self.state[symbol]
        if st["fv"] is None:
            st["fv"] = mid
        else:
            st["fv"] = alpha * mid + (1 - alpha) * st["fv"]
        st["ticks"] += 1
        st["mid_hist"].append(mid)

    def _rolling_sigma(self, symbol: str, window: int) -> float:
        hist = self.state[symbol]["mid_hist"][-window:]
        if len(hist) < 5:
            return 1.0
        mean = sum(hist) / len(hist)
        var = sum((x - mean) ** 2 for x in hist) / len(hist)
        return max(math.sqrt(var), 0.5)

    def _max_buy(self, position: int, limit: int, cap: int) -> int:
        room = limit - position
        if room <= 0:
            return 0
        scale = room / limit
        return max(1, int(round(min(cap, cap * (0.4 + 0.6 * scale)))))

    def _max_sell(self, position: int, limit: int, cap: int) -> int:
        room = limit + position
        if room <= 0:
            return 0
        scale = room / limit
        return max(1, int(round(min(cap, cap * (0.4 + 0.6 * scale)))))

    # --- multi-level takes ---
    def _take_asks(self, symbol, depth, fv, edge, cap):
        """Buy every ask level whose price is <= fv - edge, up to cap."""
        orders = []
        bought = 0
        # sorted ascending
        for px in sorted(depth.sell_orders.keys()):
            if px > fv - edge:
                break
            if bought >= cap:
                break
            avail = -depth.sell_orders[px]
            qty = min(avail, cap - bought)
            if qty > 0:
                orders.append(Order(symbol, px, qty))
                bought += qty
        return orders, bought

    def _take_bids(self, symbol, depth, fv, edge, cap):
        """Sell every bid level whose price is >= fv + edge, up to cap."""
        orders = []
        sold = 0
        # sorted descending
        for px in sorted(depth.buy_orders.keys(), reverse=True):
            if px < fv + edge:
                break
            if sold >= cap:
                break
            avail = depth.buy_orders[px]
            qty = min(avail, cap - sold)
            if qty > 0:
                orders.append(Order(symbol, px, -qty))
                sold += qty
        return orders, sold

    # =========================================================================
    # PEPPER
    # =========================================================================
    def _trade_pepper(self, depth: OrderDepth, position: int) -> List[Order]:
        P = PepperParams
        orders: List[Order] = []

        bb, _ = best_bid(depth)
        ba, _ = best_ask(depth)
        m = vwap_mid(depth)
        if m is None:
            return orders

        self._update_fv(P.SYMBOL, m, P.EWMA_ALPHA)
        st = self.state[P.SYMBOL]
        if st["ticks"] < P.MIN_WARMUP:
            return orders

        fv = st["fv"] - P.INV_SKEW * position

        buy_cap  = self._max_buy(position, P.LIMIT, P.MAX_ORDER)
        sell_cap = self._max_sell(position, P.LIMIT, P.MAX_ORDER)

        # Multi-level takes
        take_buys, bought = self._take_asks(P.SYMBOL, depth, fv, P.TAKE_EDGE, buy_cap)
        orders.extend(take_buys)
        take_sells, sold = self._take_bids(P.SYMBOL, depth, fv, P.TAKE_EDGE, sell_cap)
        orders.extend(take_sells)

        # Passive quotes around FV
        if bb is not None and ba is not None and (ba - bb) >= P.SPREAD_FLOOR:
            make_bid = int(math.floor(fv - P.MAKE_EDGE))
            make_ask = int(math.ceil(fv + P.MAKE_EDGE))
            make_bid = min(make_bid, bb + 1) if bb + 1 < ba else bb
            make_ask = max(make_ask, ba - 1) if ba - 1 > bb else ba

            rem_buy  = max(0, buy_cap - bought)
            rem_sell = max(0, sell_cap - sold)

            if rem_buy > 0 and make_bid < fv:
                orders.append(Order(P.SYMBOL, make_bid, rem_buy))
            if rem_sell > 0 and make_ask > fv:
                orders.append(Order(P.SYMBOL, make_ask, -rem_sell))

        return orders

    # =========================================================================
    # OSMIUM
    # =========================================================================
    def _trade_osmium(self, depth: OrderDepth, position: int) -> List[Order]:
        O = OsmiumParams
        orders: List[Order] = []

        bb, bb_vol = best_bid(depth)
        ba, ba_vol = best_ask(depth)
        m = simple_mid(depth)
        if m is None:
            return orders

        self._update_fv(O.SYMBOL, m, O.EWMA_ALPHA)
        st = self.state[O.SYMBOL]
        if st["ticks"] < O.MIN_WARMUP:
            return orders

        sigma = self._rolling_sigma(O.SYMBOL, O.VOL_WINDOW)
        take_edge = max(O.MIN_EDGE, O.TAKE_EDGE_K * sigma)
        make_edge = max(O.MIN_EDGE, O.MAKE_EDGE_K * sigma)

        hist = st["mid_hist"]
        momentum = 0.0
        if len(hist) > O.MOMENTUM_WIN:
            momentum = hist[-1] - hist[-1 - O.MOMENTUM_WIN]

        fv = st["fv"] - O.INV_SKEW_K * position * sigma

        buy_cap  = self._max_buy(position, O.LIMIT, O.MAX_ORDER)
        sell_cap = self._max_sell(position, O.LIMIT, O.MAX_ORDER)

        block_buy  = momentum < -O.MOMENTUM_BLOCK * sigma
        block_sell = momentum >  O.MOMENTUM_BLOCK * sigma

        bought = 0
        sold = 0

        # Single-level take (more defensive on volatile asset)
        if (not block_buy) and ba is not None and ba <= fv - take_edge and buy_cap > 0:
            qty = min(buy_cap, ba_vol)
            if qty > 0:
                orders.append(Order(O.SYMBOL, ba, qty))
                bought += qty

        if (not block_sell) and bb is not None and bb >= fv + take_edge and sell_cap > 0:
            qty = min(sell_cap, bb_vol)
            if qty > 0:
                orders.append(Order(O.SYMBOL, bb, -qty))
                sold += qty

        if bb is not None and ba is not None and (ba - bb) >= O.SPREAD_FLOOR:
            make_bid = int(math.floor(fv - make_edge))
            make_ask = int(math.ceil(fv + make_edge))
            make_bid = min(make_bid, bb)
            make_ask = max(make_ask, ba)

            rem_buy  = max(0, buy_cap - bought) // 2
            rem_sell = max(0, sell_cap - sold) // 2

            if (not block_buy) and rem_buy > 0 and make_bid < fv - 0.5:
                orders.append(Order(O.SYMBOL, make_bid, rem_buy))
            if (not block_sell) and rem_sell > 0 and make_ask > fv + 0.5:
                orders.append(Order(O.SYMBOL, make_ask, -rem_sell))

        return orders

    def run(self, state: TradingState):
        self._load_state(state.traderData)
        result: Dict[str, List[Order]] = {}

        for symbol, depth in state.order_depths.items():
            pos = state.position.get(symbol, 0)
            try:
                if symbol == PepperParams.SYMBOL:
                    result[symbol] = self._trade_pepper(depth, pos)
                elif symbol == OsmiumParams.SYMBOL:
                    result[symbol] = self._trade_osmium(depth, pos)
                else:
                    result[symbol] = []
            except Exception as e:
                print(f"[ERROR] {symbol}: {e}")
                result[symbol] = []

        traderData = self._dump_state()
        return result, 0, traderData
