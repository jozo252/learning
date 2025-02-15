import yfinance as yf
import pandas as pd
import numpy as np
import time
from scrape import *

def calculate_wacc(equity, debt, equity_cost, debt_cost, tax_rate):
    """Calculate Weighted Average Cost of Capital (WACC)."""
    V = equity + debt
    return (equity / V) * equity_cost + (debt / V) * debt_cost * (1 - tax_rate)

def calculate_terminal_value(last_cash_flow, growth_rate, discount_rate):
    """Calculate terminal value using the perpetuity growth model."""
    return last_cash_flow * (1 + growth_rate) / (discount_rate - growth_rate)

def discount_cash_flows(cash_flows, discount_rate):
    """Discount projected cash flows to present value."""
    return sum([cf / (1 + discount_rate)**i for i, cf in enumerate(cash_flows, start=1)])

def safe_get(data, key, default=np.nan):
    """Safely get values from a dictionary, returning a default value if missing."""
    return data.get(key, default)

def fetch_financial_data(ticker):
    """Fetch necessary financial data for DCF calculation."""
    stock = yf.Ticker(ticker)

    # Fetch balance sheet and cash flow statement
    balance_sheet = stock.balance_sheet
    cash_flow_statement = stock.cashflow

    # Get total liabilities
    possible_liability_keys = [
        "Total Liabilities Net Minority Interest",
        "Total Non Current Liabilities Net Minority Interest",
        "Current Liabilities"
    ]
    debt = next((balance_sheet.loc[key].iloc[0] for key in possible_liability_keys if key in balance_sheet.index), 0)

    # Get operating cash flow
    possible_ocf_keys = [
        "Operating Cash Flow",
        "Total Cash From Operating Activities Continuing Operations"
    ]
    operating_cash_flow = next((cash_flow_statement.loc[key].iloc[0] for key in possible_ocf_keys if key in cash_flow_statement.index), 0)

    # Get capital expenditures (CapEx)
    possible_capex_keys = [
        "Capital Expenditures",
        "Capital Expenditures Fixed Assets",
        "Capital Spending"
    ]
    capex = next((abs(cash_flow_statement.loc[key].iloc[0]) for key in possible_capex_keys if key in cash_flow_statement.index), 0)

    # Compute free cash flow
    free_cash_flow = operating_cash_flow - capex

    return {
        "debt": debt,
        "free_cash_flow": free_cash_flow
    }

def dcf_valuation_chatgpt(ticker, years=5, growth_rate=0.05, terminal_growth=0.02, discount_rate=0.08):
    """Perform Discounted Cash Flow (DCF) valuation."""
    stock = yf.Ticker(ticker)

    # Fetch financial data
    data = fetch_financial_data(ticker)
    debt = data["debt"]
    recent_fcf = data["free_cash_flow"]

    # Project Free Cash Flows
    projected_fcfs = [recent_fcf * (1 + growth_rate)**i for i in range(1, years + 1)]

    # Calculate Terminal Value
    terminal_value = calculate_terminal_value(projected_fcfs[-1], terminal_growth, discount_rate)

    # Discount Cash Flows
    discounted_fcfs = discount_cash_flows(projected_fcfs, discount_rate)
    discounted_terminal_value = terminal_value / (1 + discount_rate)**years

    # Calculate Enterprise Value
    enterprise_value = discounted_fcfs + discounted_terminal_value

    # Get market data
   
    shares_outstanding = safe_get(stock.info, "sharesOutstanding", np.nan)

    # Calculate intrinsic stock price
    equity_value = enterprise_value - debt
    intrinsic_value_per_share = equity_value / shares_outstanding

    return intrinsic_value_per_share