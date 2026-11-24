// Pump.fun Replay Sprint — paper trading bot (agent-fun/replay/v2).
//
// Strategy: momentum breakout + dip-aware entry with fast exits.
//  - Track every bonding curve's price (virtual SOL reserves / virtual token
//    reserves) trade by trade, plus each coin's creation price.
//  - BUY a coin only after it proves itself: price at least 2x its creation
//    price within its first trades, driven by FEW distinct buyers (<= 6
//    wallets at trigger). A coin doubled by many small wallets is wash volume
//    that dumps on late buyers; a coin doubled by a handful of big buyers is
//    accumulation, and its survivors include the coins that complete.
//    Enter early, never chase a top that already faded from its peak.
//  - SELL on take-profit (+20% over cost basis — the grind after a pump
//    gives +10-30% before the dump), trailing stop (3% off the peak once
//    +10% up), hard stop (-5%), a big-sell exit (a single sell >= 0.3 SOL
//    in a slot means distribution has started — get out), or a time/stale
//    exit so dead coins never ride to zero.
//  - One position per coin, at most 5 open, fixed 0.15 SOL sizing (capped by
//    the harness 1%-of-real-reserve buy cap and by free cash). Every buy is at
//    least 0.01 SOL so it counts toward the 20-coin minimum bar.
//  - Fully deterministic: decisions use only the hello seed's batch data
//    (events, marks, fills, account). No clock, no randomness, no environment.
//
// The harness speaks agent-fun/replay/v2: hello once, then one batch per slot
// with that slot's events, and the bot answers each batch with its orders.

import { runBot, LAMPORTS_PER_SOL, type Batch, type BotState, type OrderInput } from "./vendor/makefun-bot.ts";

// ---- tunable parameters (swept against the dev window before submitting) ----
const BUY_SIZE_LAMPORTS = 150_000_000n; // 0.15 SOL per entry
const MIN_BUY_LAMPORTS = 10_000_000n; // 0.01 SOL: minimum counted round trip
const MAX_POSITIONS = 5;
// Slippage: buys tolerate 10% (a bigger jump means the entry already ran
// away — reject it). Sells tolerate anything: a stop that gets rejected in
// a crash leaves the position riding the dump to near-zero, which is
// strictly worse than filling at the crashed price, so never reject an exit.
const BUY_SLIPPAGE_BPS = 1000;
const SELL_SLIPPAGE_BPS = 10000;

// Entry: breakout from the creation price — the trigger the dev window
// rewards. A coin that doubles from creation did so on real buy flow.
const MIN_TRADES = 4; // trades seen on the coin before we trust the signal
const CREATE_MULT = 2.0; // price >= creation price * 2.0
const MAX_BUYERS = 6; // distinct buyer wallets at trigger (whale filter)
const BUY_SHARE_WINDOW = 5; // last N trades for buy dominance
const BUY_SHARE_MIN = 0.5; // >= 50% buys in that window
const MIN_REAL_SOL = 2_000_000_000n; // real SOL reserves >= 2 SOL (buy cap >= 0.02 SOL)

// Exits: the dump comes in a single slot, so no stop can catch it — take
// the grind's profit early and leave on the first sign of distribution.
const TAKE_PROFIT = 1.2; // sell at +20% over cost basis
const TRAIL_ARM = 1.1; // trailing stop arms once +10% up
const TRAIL_DROP = 0.97; // sell at 3% below the peak
const STOP_LOSS = 0.95; // sell at -5%
const BIG_SELL_LAMPORTS = 300_000_000n; // a single sell >= 0.3 SOL: exit
const MAX_HOLD_SLOTS = 600; // ~4 min: exit a flat position (the grind is short)
const MAX_HOLD_GAIN = 1.1; // ...unless it is already up 10%+
const HARD_HOLD_SLOTS = 28800; // ~3.2 h: exit no matter what
const STALE_SLOTS = 7200; // no trade for this long: exit

interface CoinStat {
  mint: string;
  firstSlot: number;
  lastTradeSlot: number;
  low: number;
  peak: number;
  lastPrice: number;
  trades: number;
  // Price at creation (virtual reserves on the CreateEvent); for a coin first
  // seen through a trade, the first trade's price stands in as the base.
  createPrice: number;
  // Distinct buyer wallets seen on this coin (whale filter: a pump driven
  // by a handful of wallets is accumulation; by many small ones, wash).
  users: Set<string>;
  // Ring of the last few trade sides (true = buy), capped at 40.
  buys: boolean[];
  realSol: bigint;
  complete: boolean;
  traded: boolean; // we already opened (and closed) a position here
}

interface OpenPos {
  entryUnitCost: number; // lamports per token paid (fees included)
  peakPrice: number; // highest price seen since entry
  entrySlot: number;
}

const coins = new Map<string, CoinStat>();
const open = new Map<string, OpenPos>();
let stats = { batches: 0, buys: 0, sells: 0, entries: 0, exits: 0, exitTp: 0, exitSl: 0, exitTrail: 0, exitTime: 0, exitStale: 0 };

function priceOf(vq: bigint, vt: bigint): number {
  if (vt === 0n) return 0;
  return Number(vq) / Number(vt);
}

function coinFor(mint: string, slot: number): CoinStat {
  let c = coins.get(mint);
  if (c === undefined) {
    c = {
      mint,
      firstSlot: slot,
      lastTradeSlot: slot,
      low: Number.POSITIVE_INFINITY,
      peak: 0,
      lastPrice: 0,
      trades: 0,
      createPrice: 0,
      users: new Set<string>(),
      buys: [],
      realSol: 0n,
      complete: false,
      traded: false,
    };
    coins.set(mint, c);
  }
  return c;
}

function recentBuyShare(c: CoinStat): number {
  const n = Math.min(BUY_SHARE_WINDOW, c.buys.length);
  if (n === 0) return 0;
  let b = 0;
  for (let i = c.buys.length - n; i < c.buys.length; i++) if (c.buys[i]) b++;
  return b / n;
}

function decideBuys(state: BotState, batch: Batch): OrderInput[] {
  const orders: OrderInput[] = [];
  if (state.committedPositionCount() >= MAX_POSITIONS) return orders;
  // Mints with a buy already in flight: never double-buy before the fill lands.
  const pendingBuys = new Set<string>();
  for (const p of state.pending.values()) {
    if (p.order.side === "buy") pendingBuys.add(p.order.mint);
  }
  // Cash left after the orders already pending plus the ones we add this batch.
  let cashLeft = state.freeCashLamports();
  for (const e of batch.events) {
    if (e.event !== "TradeEvent") continue;
    const c = coins.get(e.mint);
    if (c === undefined || c.complete || c.traded) continue;
    if (state.positions.has(e.mint) || pendingBuys.has(e.mint)) continue;
    if (state.committedPositionCount() + orders.length >= MAX_POSITIONS) break;
    if (c.trades < MIN_TRADES) continue;
    if (c.realSol < MIN_REAL_SOL) continue;
    if (c.createPrice <= 0 || c.lastPrice < c.createPrice * CREATE_MULT) continue;
    if (c.users.size > MAX_BUYERS) continue; // wash-volume pump: skip
    if (recentBuyShare(c) < BUY_SHARE_MIN) continue;
    const cap = state.maxBuyLamports(e.mint);
    let size = BUY_SIZE_LAMPORTS < cap ? BUY_SIZE_LAMPORTS : cap;
    const txFee = state.txFeeLamports();
    if (cashLeft < size + txFee) {
      size = cashLeft - txFee;
      if (size < MIN_BUY_LAMPORTS) continue;
    }
    orders.push({ side: "buy", mint: e.mint, quote_lamports: size, max_slippage_bps: BUY_SLIPPAGE_BPS });
    cashLeft -= size + txFee;
    pendingBuys.add(e.mint);
    stats.buys++;
  }
  return orders;
}

function decideSells(state: BotState, batch: Batch): OrderInput[] {
  const orders: OrderInput[] = [];
  // A sell already in flight for a mint: do not send another one for the
  // same tokens (the first fill closes it; duplicates only get rejected).
  const pendingSells = new Set<string>();
  for (const p of state.pending.values()) {
    if (p.order.side === "sell") pendingSells.add(p.order.mint);
  }
  // Mints with a big sell in this batch: distribution has started.
  const bigSell = new Set<string>();
  for (const e of batch.events) {
    if (e.event === "TradeEvent" && !e.is_buy && BigInt(e.sol_amount) >= BIG_SELL_LAMPORTS) {
      bigSell.add(e.mint);
    }
  }
  for (const [mint, holding] of state.positions) {
    if (pendingSells.has(mint)) continue;
    const c = coins.get(mint);
    const pos = open.get(mint);
    if (c === undefined || pos === undefined) continue;
    const price = c.lastPrice;
    if (price <= 0) continue;
    if (price > pos.peakPrice) pos.peakPrice = price;
    const gain = price / pos.entryUnitCost;
    let sell = false;
    if (gain >= TAKE_PROFIT) { sell = true; stats.exitTp++; } // take profit
    else if (gain <= STOP_LOSS) { sell = true; stats.exitSl++; } // hard stop
    else if (pos.peakPrice >= pos.entryUnitCost * TRAIL_ARM && price <= pos.peakPrice * TRAIL_DROP) { sell = true; stats.exitTrail++; } // trailing stop
    else if (bigSell.has(mint)) { sell = true; stats.exitTrail++; } // distribution: a big seller appeared
    else if (batch.slot - pos.entrySlot > HARD_HOLD_SLOTS) { sell = true; stats.exitTime++; } // hard time exit
    else if (batch.slot - pos.entrySlot > MAX_HOLD_SLOTS && gain < MAX_HOLD_GAIN) { sell = true; stats.exitTime++; } // flat for too long
    else if (batch.slot - c.lastTradeSlot > STALE_SLOTS) { sell = true; stats.exitStale++; } // coin went quiet
    if (sell) {
      orders.push({ side: "sell", mint, token_amount: holding.token_amount, max_slippage_bps: SELL_SLIPPAGE_BPS });
      stats.sells++;
    }
  }
  return orders;
}

await runBot({
  onHello(_state) {
    // hello arrives once before the first batch: nothing to precompute.
  },

  onBatch(state, batch) {
    stats.batches++;
    // Fold this batch's events into per-coin stats first.
    for (const e of batch.events) {
      if (e.event === "CreateEvent") {
        const c = coinFor(e.mint, e.slot);
        c.realSol = 0n;
        c.createPrice = priceOf(BigInt(e.virtual_quote_reserves), BigInt(e.virtual_token_reserves));
      } else if (e.event === "TradeEvent") {
        const c = coinFor(e.mint, e.slot);
        const price = priceOf(BigInt(e.virtual_quote_reserves), BigInt(e.virtual_token_reserves));
        if (e.is_buy) c.users.add(e.user);
        if (price > 0) {
          if (c.createPrice <= 0) c.createPrice = price; // first seen via trade
          if (price < c.low) c.low = price;
          if (price > c.peak) c.peak = price;
          c.lastPrice = price;
          c.buys.push(e.is_buy);
          if (c.buys.length > 40) c.buys.shift();
        }
        c.trades++;
        c.lastTradeSlot = e.slot;
        c.realSol = BigInt(e.real_sol_reserves);
      } else if (e.event === "CompleteEvent") {
        const c = coinFor(e.mint, e.slot);
        c.complete = true;
      }
    }
    // Marks refresh prices for held coins that had no trade in this batch.
    for (const m of batch.marks) {
      const c = coins.get(m.mint);
      if (c !== undefined) {
        const price = priceOf(BigInt(m.virtual_quote_reserves), BigInt(m.virtual_token_reserves));
        if (price > 0) c.lastPrice = price;
      }
    }
    // Register newly opened positions (their cost basis is the truth).
    // Units: cost basis is lamports per token, the same units as the curve
    // price (virtual quote lamports / virtual token units), so the gain
    // ratio price / entryUnitCost is a true multiple.
    for (const [mint, holding] of state.positions) {
      if (!open.has(mint)) {
        const c = coins.get(mint);
        const tokens = Number(holding.token_amount);
        const cost = Number(holding.cost_lamports);
        const entryUnitCost = tokens > 0 ? cost / tokens : 0;
        const price = c !== undefined && c.lastPrice > 0 ? c.lastPrice : entryUnitCost;
        open.set(mint, { entryUnitCost, peakPrice: price, entrySlot: batch.slot });
        stats.entries++;
      }
    }
    // Drop tracking for positions that are gone (sold, completed, flattened).
    for (const mint of [...open.keys()]) {
      if (!state.positions.has(mint)) {
        open.delete(mint);
        const c = coins.get(mint);
        if (c !== undefined) c.traded = true;
        stats.exits++;
      }
    }
    const sells = decideSells(state, batch);
    const buys = decideBuys(state, batch);
    return [...sells, ...buys];
  },

  onDone(state, _done) {
    const cash = Number(state.cash) / Number(LAMPORTS_PER_SOL);
    console.error(
      `[bot] done batches=${stats.batches} buys=${stats.buys} sells=${stats.sells} ` +
        `entries=${stats.entries} exits=${stats.exits} final_cash_sol=${cash.toFixed(4)} ` +
        `open_positions=${state.positions.size} exit_tp=${stats.exitTp} exit_sl=${stats.exitSl} ` +
        `exit_trail=${stats.exitTrail} exit_time=${stats.exitTime} exit_stale=${stats.exitStale}`,
    );
  },
});
