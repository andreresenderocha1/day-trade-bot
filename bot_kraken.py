import ccxt
import time
import os
import json
import logging
import datetime
import smtplib
import threading
import requests as _requests
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from http.server import HTTPServer, BaseHTTPRequestHandler
import pandas as pd
import pandas_ta_classic as ta
from requests.exceptions import RequestException

DATA_DIR = os.environ.get("DATA_DIR", ".")
STATE_FILE = os.path.join(DATA_DIR, "bot_state.json")


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'OK')

    def log_message(self, *args):
        pass


def _start_health_server():
    port = int(os.environ.get('PORT', 8080))
    HTTPServer(('0.0.0.0', port), _HealthHandler).serve_forever()


threading.Thread(target=_start_health_server, daemon=True).start()


class KrakenTradingBot:
    def __init__(self, symbol='BTC/CAD', timeframe='4h'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.quote_currency = symbol.split('/')[1]

        self.st_length = 10
        self.st_multiplier = 3.0

        self.trade_allocation = 0.25
        self.take_profit_pct = 0.025
        self.hard_stop_pct = 0.030
        self.rsi_max_buy = 65
        self.rsi_min_buy = 35
        self.volume_factor = 1.2
        self.max_hold_candles = 18
        self.loss_cooldown_candles = 2
        self.candles_since_loss = 999

        self.simulated_fiat = 1000.0
        self.initial_fiat = 1000.0
        self.in_position = False
        self.entry_price = 0.0
        self.position_amount = 0.0
        self.candles_in_position = 0
        self._st_dir_col = None
        self._st_val_col = None

        # Weekly reporting
        self.week_trades = []
        self.weekly_report_sent_date = ""

        self.setup_logging()
        self.load_state()
        self.setup_exchange()

    def setup_logging(self):
        log_file = os.path.join(DATA_DIR, "trade_log.txt")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_exchange(self):
        api_key = os.environ.get('KRAKEN_API_KEY')
        secret_key = os.environ.get('KRAKEN_SECRET_KEY')
        if not api_key or not secret_key:
            self.logger.warning("API Keys não encontradas — modo paper trading.")
        self.exchange = ccxt.kraken({
            'apiKey': api_key,
            'secret': secret_key,
            'enableRateLimit': True,
        })

    # ── State persistence ─────────────────────────────────────────────────────

    def save_state(self):
        state = {
            "simulated_fiat": self.simulated_fiat,
            "in_position": self.in_position,
            "entry_price": self.entry_price,
            "position_amount": self.position_amount,
            "candles_in_position": self.candles_in_position,
            "candles_since_loss": self.candles_since_loss,
            "week_trades": self.week_trades,
            "weekly_report_sent_date": self.weekly_report_sent_date,
        }
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            self.logger.error(f"Erro ao salvar estado: {e}")

    def load_state(self):
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            self.simulated_fiat = state.get("simulated_fiat", self.initial_fiat)
            self.in_position = state.get("in_position", False)
            self.entry_price = state.get("entry_price", 0.0)
            self.position_amount = state.get("position_amount", 0.0)
            self.candles_in_position = state.get("candles_in_position", 0)
            self.candles_since_loss = state.get("candles_since_loss", 999)
            self.week_trades = state.get("week_trades", [])
            self.weekly_report_sent_date = state.get("weekly_report_sent_date", "")
            self.logger.info(
                f"✅ Estado carregado: saldo={self.simulated_fiat:.2f} | em_posição={self.in_position}"
            )
        except Exception as e:
            self.logger.error(f"Erro ao carregar estado: {e}")

    # ── Market data ───────────────────────────────────────────────────────────

    def fetch_market_data(self):
        bars = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=300)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.ta.ema(length=9, append=True)
        df.ta.ema(length=21, append=True)
        df.ta.ema(length=200, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.supertrend(length=self.st_length, multiplier=self.st_multiplier, append=True)
        df.ta.bbands(length=20, std=2.0, append=True)
        df['vol_ma20'] = df['volume'].rolling(20).mean()
        if self._st_dir_col is None:
            dir_cols = [c for c in df.columns if c.startswith('SUPERTd')]
            val_cols = [c for c in df.columns if c.startswith('SUPERT_')]
            if dir_cols:
                self._st_dir_col = dir_cols[0]
            if val_cols:
                self._st_val_col = val_cols[0]
        return df

    def _get_st_dir(self, row):
        if self._st_dir_col and self._st_dir_col in row.index:
            return float(row[self._st_dir_col])
        return 0

    # ── Signals ───────────────────────────────────────────────────────────────

    def should_buy(self, df):
        last = df.iloc[-2]
        prev = df.iloc[-3]
        price  = float(last['close'])
        ema200 = float(last['EMA_200'])
        rsi    = float(last['RSI_14'])
        macd_h = float(last['MACDh_12_26_9'])
        volume = float(last['volume'])
        vol_ma = float(last['vol_ma20'])
        ema9   = float(last['EMA_9'])
        ema21  = float(last['EMA_21'])
        st_dir      = self._get_st_dir(last)
        prev_st_dir = self._get_st_dir(prev)
        prev_macd_h = float(prev['MACDh_12_26_9'])

        if price <= ema200:
            return False, ""
        if not (self.rsi_min_buy <= rsi < self.rsi_max_buy):
            return False, ""
        if vol_ma > 0 and volume < self.volume_factor * vol_ma:
            return False, ""
        if self.candles_since_loss < self.loss_cooldown_candles:
            return False, ""

        if prev_st_dir == -1 and st_dir == 1:
            return True, f"Supertrend flip BULL | RSI:{rsi:.0f} | Vol:{volume/vol_ma:.1f}x"
        if st_dir == 1 and ema9 > ema21 and macd_h > 0 and macd_h > prev_macd_h:
            return True, f"ST+EMA+MACD alinhados | RSI:{rsi:.0f} | Vol:{volume/vol_ma:.1f}x"
        return False, ""

    def should_sell(self, df):
        current_price = float(df.iloc[-1]['close'])
        last = df.iloc[-2]
        prev = df.iloc[-3]
        rsi        = float(last['RSI_14'])
        macd_h     = float(last['MACDh_12_26_9'])
        prev_mh    = float(prev['MACDh_12_26_9'])
        st_dir     = self._get_st_dir(last)
        prev_st_dir = self._get_st_dir(prev)
        lucro_pct  = (current_price - self.entry_price) / self.entry_price

        if lucro_pct >= self.take_profit_pct:
            return True, f"Take Profit +{self.take_profit_pct*100:.1f}% ({lucro_pct*100:.2f}%)"
        if lucro_pct <= -self.hard_stop_pct:
            return True, f"Hard Stop -{self.hard_stop_pct*100:.1f}% ({lucro_pct*100:.2f}%)"
        if self.candles_in_position >= self.max_hold_candles:
            return True, f"Max hold {self.max_hold_candles} candles ({lucro_pct*100:.2f}%)"
        if prev_st_dir == 1 and st_dir == -1:
            return True, f"Supertrend flip BEAR ({lucro_pct*100:.2f}%)"
        if rsi > 72 and lucro_pct > 0:
            return True, f"RSI sobrecomprado {rsi:.0f} ({lucro_pct*100:.2f}%)"
        if prev_mh > 0 and macd_h <= 0 and lucro_pct >= 0.005:
            return True, f"MACD momentum esgotado ({lucro_pct*100:.2f}%)"
        return False, ""

    # ── Trade execution ───────────────────────────────────────────────────────

    def execute_trade(self, action, current_price, reason=""):
        try:
            fee_rate = 0.0026
            now = datetime.datetime.utcnow().isoformat()

            if action == 'BUY':
                invest_amount = self.simulated_fiat * self.trade_allocation
                if self.simulated_fiat < 10.0:
                    self.logger.warning("⛔ Saldo insuficiente.")
                    return
                if invest_amount < 10.0:
                    invest_amount = 10.0
                amount_to_buy = (invest_amount * (1 - fee_rate)) / current_price
                self.simulated_fiat -= invest_amount
                self.in_position = True
                self.entry_price = current_price
                self.position_amount = amount_to_buy
                tp = current_price * (1 + self.take_profit_pct)
                sl = current_price * (1 - self.hard_stop_pct)
                self.logger.info(f"🟢 COMPRA: {reason}")
                self.logger.info(
                    f"   -> {amount_to_buy:.6f} BTC @ {current_price:.2f} {self.quote_currency}"
                    f" | Valor: ${invest_amount:.2f} | TP: {tp:.2f} | SL: {sl:.2f}"
                    f" | Saldo: {self.simulated_fiat:.2f}"
                )
                self.week_trades.append({
                    "timestamp": now, "action": "BUY",
                    "price": current_price, "reason": reason,
                    "invest_cad": round(invest_amount, 2),
                    "balance_after": round(self.simulated_fiat, 2),
                })

            elif action == 'SELL':
                gross = self.position_amount * current_price
                net = gross * (1 - fee_rate)
                lucro_trade = net - (self.position_amount * self.entry_price)
                self.simulated_fiat += net
                total = self.simulated_fiat - self.initial_fiat
                total_pct = (total / self.initial_fiat) * 100
                self.logger.info(f"🔴 VENDA: {reason}")
                self.logger.info(
                    f"   -> Entrada: {self.entry_price:.2f} | Saída: {current_price:.2f}"
                    f" | Trade: {lucro_trade:+.2f} {self.quote_currency}"
                )
                self.logger.info(
                    f"💰 SALDO: {self.simulated_fiat:.2f} {self.quote_currency}"
                    f" | Acumulado: {total:+.2f} ({total_pct:+.2f}%)"
                )
                if lucro_trade < 0:
                    self.candles_since_loss = 0
                self.week_trades.append({
                    "timestamp": now, "action": "SELL",
                    "entry_price": self.entry_price,
                    "exit_price": current_price,
                    "reason": reason,
                    "pnl_cad": round(lucro_trade, 4),
                    "pnl_pct": round((current_price - self.entry_price) / self.entry_price * 100, 3),
                    "balance_after": round(self.simulated_fiat, 2),
                })
                self.in_position = False
                self.entry_price = 0.0
                self.position_amount = 0.0
                self.candles_in_position = 0

            self.save_state()

        except Exception as e:
            self.logger.error(f"Erro ao simular {action}: {e}")

    # ── Weekly report ─────────────────────────────────────────────────────────

    def generate_weekly_summary(self):
        now = datetime.datetime.utcnow()
        week_start = (now - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
        week_end = now.strftime("%Y-%m-%d")

        sells = [t for t in self.week_trades if t["action"] == "SELL"]
        wins   = [t for t in sells if t["pnl_cad"] > 0]
        losses = [t for t in sells if t["pnl_cad"] <= 0]
        n = len(sells)
        total_pnl   = sum(t["pnl_cad"] for t in sells)
        win_rate    = (len(wins) / n * 100) if n > 0 else 0
        best        = max((t["pnl_cad"] for t in sells), default=0)
        worst       = min((t["pnl_cad"] for t in sells), default=0)
        avg         = total_pnl / n if n > 0 else 0
        cumulative_pct = ((self.simulated_fiat - self.initial_fiat) / self.initial_fiat) * 100

        return {
            "week_start": week_start,
            "week_end": week_end,
            "symbol": self.symbol,
            "algo_version": "v5.0",
            "params": {
                "timeframe": self.timeframe,
                "st_length": self.st_length,
                "st_multiplier": self.st_multiplier,
                "take_profit_pct": self.take_profit_pct * 100,
                "hard_stop_pct": self.hard_stop_pct * 100,
                "rsi_range": [self.rsi_min_buy, self.rsi_max_buy],
                "volume_factor": self.volume_factor,
                "max_hold_candles": self.max_hold_candles,
            },
            "initial_balance_cad": round(self.initial_fiat, 2),
            "final_balance_cad": round(self.simulated_fiat, 2),
            "cumulative_pnl_cad": round(self.simulated_fiat - self.initial_fiat, 4),
            "cumulative_pnl_pct": round(cumulative_pct, 3),
            "weekly_pnl_cad": round(total_pnl, 4),
            "num_completed_trades": n,
            "win_trades": len(wins),
            "loss_trades": len(losses),
            "win_rate_pct": round(win_rate, 2),
            "best_trade_cad": round(best, 4),
            "worst_trade_cad": round(worst, 4),
            "avg_trade_cad": round(avg, 4),
            "in_open_position": self.in_position,
            "trades": self.week_trades,
        }

    def upload_to_gist(self, summary):
        token = os.environ.get('GITHUB_PAT')
        if not token:
            self.logger.warning("GITHUB_PAT não configurado — Gist não atualizado")
            return
        headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json',
        }
        description = 'day-trade-bot-weekly-summary'
        content = json.dumps(summary, indent=2, ensure_ascii=False)
        gist_id = None
        try:
            r = _requests.get(
                'https://api.github.com/gists',
                headers=headers, params={'per_page': 30}, timeout=15
            )
            if r.ok:
                for g in r.json():
                    if g.get('description') == description:
                        gist_id = g['id']
                        break
        except Exception as e:
            self.logger.error(f"Erro listando Gists: {e}")
        try:
            payload = {'files': {'summary.json': {'content': content}}}
            if gist_id:
                r = _requests.patch(
                    f'https://api.github.com/gists/{gist_id}',
                    json=payload, headers=headers, timeout=15
                )
            else:
                r = _requests.post(
                    'https://api.github.com/gists',
                    json={**payload, 'description': description, 'public': True},
                    headers=headers, timeout=15
                )
            if r.ok:
                self.logger.info(f"📊 Gist: {r.json().get('html_url', '')}")
            else:
                self.logger.error(f"Erro Gist: {r.status_code} {r.text[:200]}")
        except Exception as e:
            self.logger.error(f"Erro upload Gist: {e}")

    def send_email_report(self, summary):
        smtp_user = os.environ.get('GMAIL_USER')
        smtp_pass = os.environ.get('GMAIL_APP_PASSWORD')
        recipient = os.environ.get('REPORT_EMAIL', smtp_user)
        if not smtp_user or not smtp_pass:
            self.logger.warning("Gmail não configurado — email não enviado")
            return

        pnl = summary['weekly_pnl_cad']
        base = summary['final_balance_cad'] - pnl
        pnl_week_pct = (pnl / base * 100) if base else 0
        emoji  = "📈" if pnl >= 0 else "📉"
        status = "LUCRO" if pnl >= 0 else "PREJUÍZO"
        subject = (
            f"{emoji} Trade Bot {summary['week_end']} | "
            f"{status}: {pnl:+.2f} CAD ({pnl_week_pct:+.2f}%)"
        )

        trades_rows = ""
        for t in summary['trades']:
            if t['action'] == 'SELL':
                color = '#2e7d32' if t['pnl_cad'] > 0 else '#c62828'
                trades_rows += (
                    f"<tr>"
                    f"<td style='padding:8px'>{t['timestamp'][:16].replace('T',' ')}</td>"
                    f"<td style='padding:8px;text-align:right'>{t['entry_price']:,.2f}</td>"
                    f"<td style='padding:8px;text-align:right'>{t['exit_price']:,.2f}</td>"
                    f"<td style='padding:8px;text-align:right;color:{color};font-weight:bold'>{t['pnl_cad']:+.2f}</td>"
                    f"<td style='padding:8px;text-align:right;color:{color}'>{t['pnl_pct']:+.2f}%</td>"
                    f"<td style='padding:8px;font-size:12px'>{t['reason'][:45]}</td>"
                    f"</tr>"
                )
        if not trades_rows:
            trades_rows = (
                "<tr><td colspan='6' style='padding:16px;text-align:center;color:#999'>"
                "Nenhum trade fechado esta semana</td></tr>"
            )

        wr_color = '#2e7d32' if summary['win_rate_pct'] >= 50 else '#c62828'
        cum_color = '#2e7d32' if summary['cumulative_pnl_cad'] >= 0 else '#c62828'

        html = f"""<!DOCTYPE html>
<html><body style="font-family:Arial,sans-serif;background:#f0f0f0;margin:0;padding:20px">
<div style="max-width:680px;margin:auto">
  <div style="background:#0d1b2a;color:white;padding:24px;border-radius:12px 12px 0 0;text-align:center">
    <h2 style="margin:0 0 4px">🤖 Day Trade Bot — Relatório Semanal</h2>
    <p style="margin:0;color:#8899aa;font-size:14px">{summary['symbol']} | {summary['algo_version']} | {summary['week_start']} → {summary['week_end']}</p>
  </div>
  <div style="background:white;padding:20px;display:flex;gap:12px">
    <div style="flex:1;text-align:center;padding:16px;border-radius:8px;background:#f8f8f8;border-top:4px solid {'#2e7d32' if pnl>=0 else '#c62828'}">
      <div style="font-size:26px;font-weight:bold;color:{'#2e7d32' if pnl>=0 else '#c62828'}">{pnl:+.2f} CAD</div>
      <div style="color:#666;font-size:13px">P&amp;L Semanal ({pnl_week_pct:+.2f}%)</div>
    </div>
    <div style="flex:1;text-align:center;padding:16px;border-radius:8px;background:#f8f8f8;border-top:4px solid #1565c0">
      <div style="font-size:26px;font-weight:bold">{summary['final_balance_cad']:.2f} CAD</div>
      <div style="color:#666;font-size:13px">Saldo Atual</div>
    </div>
    <div style="flex:1;text-align:center;padding:16px;border-radius:8px;background:#f8f8f8;border-top:4px solid {wr_color}">
      <div style="font-size:26px;font-weight:bold;color:{wr_color}">{summary['win_rate_pct']:.0f}%</div>
      <div style="color:#666;font-size:13px">Win Rate ({summary['win_trades']}W / {summary['loss_trades']}L)</div>
    </div>
  </div>
  <div style="background:white;padding:0 20px 4px">
    <div style="display:flex;gap:12px;padding-bottom:16px">
      <div style="flex:1;text-align:center;padding:12px;border-radius:8px;background:#f8f8f8">
        <div style="font-size:18px;font-weight:bold">{summary['num_completed_trades']}</div>
        <div style="color:#666;font-size:12px">Trades Fechados</div>
      </div>
      <div style="flex:1;text-align:center;padding:12px;border-radius:8px;background:#f8f8f8">
        <div style="font-size:18px;font-weight:bold;color:#2e7d32">{summary['best_trade_cad']:+.2f}</div>
        <div style="color:#666;font-size:12px">Melhor Trade</div>
      </div>
      <div style="flex:1;text-align:center;padding:12px;border-radius:8px;background:#f8f8f8">
        <div style="font-size:18px;font-weight:bold;color:#c62828">{summary['worst_trade_cad']:+.2f}</div>
        <div style="color:#666;font-size:12px">Pior Trade</div>
      </div>
      <div style="flex:1;text-align:center;padding:12px;border-radius:8px;background:#f8f8f8">
        <div style="font-size:18px;font-weight:bold;color:{cum_color}">{summary['cumulative_pnl_pct']:+.2f}%</div>
        <div style="color:#666;font-size:12px">Acumulado Total</div>
      </div>
    </div>
  </div>
  <div style="background:white;padding:0 20px 20px">
    <h3 style="margin:0 0 12px;color:#333">Trades desta Semana</h3>
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <thead>
        <tr style="background:#0d1b2a;color:white">
          <th style="padding:10px;text-align:left">Data/Hora</th>
          <th style="padding:10px;text-align:right">Entrada</th>
          <th style="padding:10px;text-align:right">Saída</th>
          <th style="padding:10px;text-align:right">P&amp;L (CAD)</th>
          <th style="padding:10px;text-align:right">%</th>
          <th style="padding:10px;text-align:left">Motivo</th>
        </tr>
      </thead>
      <tbody>{trades_rows}</tbody>
    </table>
  </div>
  <div style="background:white;padding:16px 20px 20px;border-top:1px solid #eee">
    <h3 style="margin:0 0 8px;color:#333;font-size:14px">Parâmetros Ativos</h3>
    <p style="margin:0;color:#666;font-size:12px;line-height:1.6">
      Supertrend({summary['params']['st_length']}, {summary['params']['st_multiplier']}) &nbsp;|&nbsp;
      Timeframe: {summary['params']['timeframe']} &nbsp;|&nbsp;
      TP: +{summary['params']['take_profit_pct']:.1f}% &nbsp;|&nbsp;
      SL: -{summary['params']['hard_stop_pct']:.1f}% &nbsp;|&nbsp;
      RSI: {summary['params']['rsi_range'][0]}-{summary['params']['rsi_range'][1]} &nbsp;|&nbsp;
      Volume: {summary['params']['volume_factor']}x &nbsp;|&nbsp;
      Max Hold: {summary['params']['max_hold_candles']} candles
    </p>
  </div>
  <div style="background:#0d1b2a;color:#556677;padding:12px 20px;border-radius:0 0 12px 12px;font-size:11px;text-align:center">
    Paper trading — sem dinheiro real &nbsp;|&nbsp; Gerado automaticamente pelo bot
  </div>
</div>
</body></html>"""

        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = smtp_user
        msg['To'] = recipient
        msg.attach(MIMEText(html, 'html'))
        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=30) as server:
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_user, recipient, msg.as_string())
            self.logger.info(f"📧 Relatório enviado para {recipient}")
        except Exception as e:
            self.logger.error(f"Erro ao enviar email: {e}")

    def check_weekly_report(self):
        now = datetime.datetime.utcnow()
        if now.weekday() != 0:
            return
        today = now.strftime("%Y-%m-%d")
        if self.weekly_report_sent_date == today:
            return
        # 14:00–14:59 UTC = ~10am ET
        if now.hour != 14:
            return
        self.logger.info("📊 Gerando relatório semanal...")
        summary = self.generate_weekly_summary()
        self.upload_to_gist(summary)
        self.send_email_report(summary)
        self.weekly_report_sent_date = today
        self.week_trades = []
        self.save_state()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        self.logger.info("=" * 70)
        self.logger.info(
            f"🤖 Bot v5.0 (Supertrend+EMA200+RSI+Volume)"
            f" | {self.symbol} | TF: {self.timeframe}"
        )
        self.logger.info(
            f"💵 Banca: {self.simulated_fiat:.2f} {self.quote_currency}"
            f" | Alocação: {self.trade_allocation*100:.0f}%"
            f" | TP: +{self.take_profit_pct*100:.1f}%"
            f" | Hard SL: -{self.hard_stop_pct*100:.1f}%"
        )
        self.logger.info(
            f"🔬 Supertrend({self.st_length},{self.st_multiplier})"
            f" | Volume: {self.volume_factor}x | RSI: {self.rsi_min_buy}-{self.rsi_max_buy}"
            f" | Max hold: {self.max_hold_candles} candles ({self.max_hold_candles*4}h)"
        )
        self.logger.info("=" * 70)

        last_candle_time = None

        while True:
            try:
                self.check_weekly_report()

                df = self.fetch_market_data()
                current_price = float(df.iloc[-1]['close'])
                last = df.iloc[-2]

                ema9   = float(last['EMA_9'])
                ema21  = float(last['EMA_21'])
                ema200 = float(last['EMA_200'])
                rsi    = float(last['RSI_14'])
                macd_h = float(last['MACDh_12_26_9'])
                st_dir = self._get_st_dir(last)

                required = [ema200, rsi, macd_h]
                if any(pd.isna(v) for v in required) or self._st_dir_col is None:
                    self.logger.info("⏳ Aguardando dados suficientes...")
                    time.sleep(60)
                    continue

                candle_time = df.iloc[-2]['timestamp']
                if last_candle_time != candle_time:
                    last_candle_time = candle_time
                    if self.in_position:
                        self.candles_in_position += 1
                    if self.candles_since_loss < 999:
                        self.candles_since_loss += 1

                st_str = "🟩BULL" if st_dir == 1 else "🟥BEAR"
                if self.in_position:
                    lucro = (current_price - self.entry_price) / self.entry_price * 100
                    pos_str = (
                        f"EM POSIÇÃO entrada:{self.entry_price:.2f}"
                        f" | {lucro:+.2f}% | {self.candles_in_position}/{self.max_hold_candles}c"
                    )
                else:
                    cd = self.loss_cooldown_candles - self.candles_since_loss
                    cd_str = f" | cooldown:{cd}c" if cd > 0 else ""
                    pos_str = f"SEM POSIÇÃO{cd_str}"

                self.logger.info(
                    f"Preço:{current_price:.2f} | ST:{st_str}"
                    f" | EMA9:{ema9:.0f} EMA21:{ema21:.0f} EMA200:{ema200:.0f}"
                    f" | RSI:{rsi:.1f} | MACD_H:{macd_h:.0f}"
                    f" | {pos_str}"
                )

                if self.in_position:
                    vende, motivo = self.should_sell(df)
                    if vende:
                        self.execute_trade('SELL', current_price, motivo)
                else:
                    compra, motivo = self.should_buy(df)
                    if compra:
                        self.execute_trade('BUY', current_price, motivo)

            except ccxt.NetworkError as e:
                self.logger.error(f"Erro de Rede: {e}")
            except ccxt.ExchangeError as e:
                self.logger.error(f"Erro na Exchange: {e}")
            except RequestException as e:
                self.logger.error(f"Erro HTTP: {e}")
            except Exception as e:
                self.logger.error(f"Erro inesperado: {e}", exc_info=True)

            time.sleep(60)


if __name__ == "__main__":
    bot = KrakenTradingBot(symbol='BTC/CAD', timeframe='4h')
    bot.run()
