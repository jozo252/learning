import requests
from bs4 import BeautifulSoup
import json
import yfinance as yf
import pandas as pd
import math
import numpy as np
from decimal import Decimal
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
import os 
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import LSTM
from openai import OpenAI
from io import StringIO
import pandas_datareader.data as pdr

openai_key='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'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
def operating_cash_flow(ticker):
    ticker_obj = yf.Ticker(ticker)
    cf = ticker_obj.cashflow  
    if "Operating Cash Flow" in cf.index:
        OCF = cf.loc["Operating Cash Flow"]
        OCF=OCF.iloc[0]
        return OCF
    else: 
        print(f'{ticker} OCF error')

def free_cash_flow(ticker):
    ticker_obj = yf.Ticker(ticker)
    cf = ticker_obj.cashflow  
    if "Free Cash Flow" in cf.index:
        OCF = cf.loc["Free Cash Flow"]
        data = []
        for i in range(len(OCF)):
            data.append(OCF.iloc[i])
        return data
    else:
        print(f'{ticker} FCF error')

def shares_num(ticker):
    ticker_obj = yf.Ticker(ticker)
    cf = ticker_obj.info  
    try:
        shares_outstanding=cf.get('sharesOutstanding')
        return shares_outstanding
    except:
        print('error in shares_num')

def WACC(ticker):    #cf,balance_sht,financials
    try:
        ticker_obj = yf.Ticker(ticker)
        cf = ticker_obj.info
        balance_sht=ticker_obj.balance_sheet
        financials=ticker_obj.financials
        market_cap=cf.get('marketCap')                  #E - market_cap
        debt=debt_func(ticker)                              #D - debt
        beta=cf.get('beta',1.0)                         #beta
        Rd=cost_of_debt(ticker)         #Rd - cost of debt
        t=tax_rate(ticker)                          #t = tax rate
        risk_free_rate=0.04
        market_risk_premium=0.05
        Re=risk_free_rate + beta * market_risk_premium  #Re - cost of equity
        wacc= ((market_cap/(market_cap+debt))*Re)+(debt/(market_cap+debt))*Rd*(1-t)
        #print(f"WACC = {wacc * 100:.2f}%")              #WACC!!
        return wacc
    except:
        print('error in WACC')
    



def cost_of_debt(ticker):
        ticker_obj = yf.Ticker(ticker)
        financials = ticker_obj.financials
        balance_sht=ticker_obj.balance_sheet
        try:
            interest_expense=financials.loc['Interest Expense']
            results_interest_expense=[]
            for i in range(len(interest_expense)):
                results_interest_expense.append(interest_expense.iloc[i])
            cleaned_list = [x for x in results_interest_expense if not math.isnan(x)]
            result_cleaned_list=cleaned_list[0]             #interest expense
            if "Long Term Debt" in balance_sht.index:
                long_term_debt = balance_sht.loc["Long Term Debt"].iloc[0]
            else:
                long_term_debt = 0

            if "Short Long Term Debt" in balance_sht.index:
                short_debt = balance_sht.loc["Short Long Term Debt"].iloc[0]
            else:
                short_debt = 0
            book_debt = long_term_debt+short_debt           #book debt
            if book_debt !=0 and not math.isnan(result_cleaned_list):
                result_cost_of_debt = abs(result_cleaned_list)/book_debt
                return result_cost_of_debt
            else:
                return 0.02
        except:
            print('error in cost of debt')
def tax_rate(ticker):
    ticker_obj = yf.Ticker(ticker)
    financials = ticker_obj.financials
    try:
        if 'Tax Rate For Calcs' in financials.index:
            result_tax_rate=financials.loc['Tax Rate For Calcs'].iloc[0]
            return result_tax_rate
        else:
            if 'Tax Provision' in financials.index and 'Pretax Income' in financials.index:
                tax_provision = financials.loc['Tax Provision'].iloc[0]
                pretax_income = financials.loc['Pretax Income'].iloc[0]
                if pretax_income != 0:
                    t_effective = abs(tax_provision)/ abs(pretax_income)
                    return t_effective
                else:
                    return 0.21
            else:
                return 0.21
    except:
        print('error in tax rate')

def debt_func(ticker):
    ticker_obj = yf.Ticker(ticker)
    inf = ticker_obj.info
    try:
        total_debt=inf.get('totalDebt')
        return total_debt
        
    except:
        print('error in debt func')



def ebit(ticker):
    ticker_obj = yf.Ticker(ticker)
    financials=ticker_obj.financials
    try:
        ebit=financials.loc['EBIT']
        return ebit.iloc[0]
        
    except:
        print('error in ebit')

def daa(ticker):
    ticker_obj = yf.Ticker(ticker)
    cf=ticker_obj.cash_flow
    daa=cf.loc['Depreciation And Amortization']
    try:
        return daa.iloc[0]
    except:
        print('error in daa')
        
def capital_expediture(ticker):
    ticker_obj = yf.Ticker(ticker)
    cf=ticker_obj.cash_flow
    ce=cf.loc['Capital Expenditure']
    try:
        return ce.iloc[0]
    except:
        print('error in Capital Expenditure')

def change_wc(ticker):
    ticker_obj = yf.Ticker(ticker)
    cf=ticker_obj.cash_flow
    wc=cf.loc['Change In Working Capital']
    try:
        return wc.iloc[0]
    except:
        print('error in Change In Working Capital')



def finan(ticker):
    ticker_obj = yf.Ticker(ticker)
    financials=ticker_obj.financials
    return financials

def cashFlow(ticker):
    ticker_obj = yf.Ticker(ticker)
    cf=ticker_obj.cash_flow
    return cf

def inf(ticker):
    ticker_obj = yf.Ticker(ticker)
    inf=ticker_obj.info
    return inf

def fcff(ticker):
    
    try:
        ebit_num=ebit(ticker)
        tax=tax_rate(ticker)
        daA=daa(ticker)
        capitalExpediture=capital_expediture(ticker)
        wc=change_wc(ticker)
        nopat=ebit_num*(1-tax)
        fcff_value = nopat + daA -capitalExpediture-wc
        return fcff_value
    except:
        print('error in fcff')

def fcff_forecast(ticker,years,growth):                      #lightweight forecast
    fcff_val=fcff(ticker)
    try:
        values=[]
        for i in range(years):
            fcff_forcast=fcff_val*(1+growth)
            fcff_val=fcff_forcast
            values.append(fcff_val)
        return values
    except:
        print('error in fcff forecast')

def cash_equivalent(ticker):                    #currency is fucked!!!!!!!!!!!!!!!! maybe
    ticker_obj = yf.Ticker(ticker)
    cf=ticker_obj.cash_flow
    try:
        cashAndEquivalent=cf.loc['End Cash Position']
        return cashAndEquivalent.iloc[0]
    except:
        print('error in cash equivalent')

def net_debt(ticker):
    ticker_obj = yf.Ticker(ticker)
    cf=ticker_obj.cash_flow
    inf = ticker_obj.info
    try:
        total_debt=debt_func(ticker)
        ce=cash_equivalent(ticker)
        netDebt=total_debt-ce
        return netDebt
    except:
        print('error in net debt')

def dcf_valuation(ticker, years, growth, terminal_growth):
    try:
        ticker_obj = yf.Ticker(ticker)
        inf = ticker_obj.info

        # Forecast Free Cash Flow to Firm (FCFF)
        fcff_forecasts = fcff_forecast(ticker, years, growth)
        wacc = WACC(ticker)
        netDebt = net_debt(ticker)
        shares_outstanding = shares_num(ticker)

        # Ensure WACC > Terminal Growth
        if wacc <= terminal_growth:
            print(f"WACC: {wacc}, Terminal Growth: {terminal_growth}")
            raise ValueError("WACC must be greater than Terminal Growth Rate")

        # Discount FCFFs
        pv_fcffs = [fcff / ((1 + wacc) ** t) for t, fcff in enumerate(fcff_forecasts, start=1)]

        # Terminal Value Calculation
        last_fcff = fcff_forecasts[-1]
        fcff_n_plus_1 = last_fcff * (1 + terminal_growth)
        terminal_value = fcff_n_plus_1 / (wacc - terminal_growth)
        pv_terminal = terminal_value / ((1 + wacc) ** len(fcff_forecasts))

        # Enterprise & Equity Value
        enterprise_value = sum(pv_fcffs) + pv_terminal
        equity_value = enterprise_value - abs(netDebt)  # Ensures correct subtraction

        # Fair Value Per Share
        fair_value_per_share = equity_value / shares_outstanding

        # Fetch Current Price (Handles Missing Data)

        return fair_value_per_share

    except Exception as e:
        print(f"Error in DCF valuation: {e}")
        return None




def skuska(ticker):
    ticker_obj = yf.Ticker(ticker)
    cf=ticker_obj.cash_flow
    inf=ticker_obj.info
    financials=ticker_obj.financials
    balance_sht=ticker_obj.balance_sheet
    price=inf['currentPrice']
    return price


def pe_forward_price(ticker):
    ticker_obj=yf.Ticker(ticker)
    inf=ticker_obj.info
    try:
        forward_pe=inf.get('forwardPE')
        forward_eps=inf.get('forwardEps')
        fair_price_forward  = forward_eps  * forward_pe
        return fair_price_forward
    except:
        print('ERROR in forward pe')

def pe_trailing_price(ticker):                                      #problem with missing pe
    ticker_obj=yf.Ticker(ticker)
    inf=ticker_obj.info
    try:
        trailing_pe=inf.get('trailingPE')
        trailing_eps=inf.get('trailingEps')
        fair_price_trailing = trailing_eps * trailing_pe
        
        return fair_price_trailing
    except:
        print('ERROR in trailing pe')

def pe_trailing(ticker):                                      #problem with missing pe
    ticker_obj=yf.Ticker(ticker)
    inf=ticker_obj.info
    try:
        trailing_pe=inf.get('trailingPE')
        if trailing_pe is None:
            forwardPE=inf.get('forwardPE')
            return forwardPE
        else:
            return trailing_pe
        
    except:
        print('ERROR in trailing pe')

def info(ticker):
    ticker_obj = yf.Ticker(ticker)
    inf=ticker_obj.info
    trailingPe=pe_trailing_price(ticker)
    forwardPe=pe_forward_price(ticker)
    dcfPrice=dcf_valuation(ticker,0.05)
    price=inf['currentPrice']
    return trailingPe,forwardPe,dcfPrice,price

def current_price(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        inf=ticker_obj.info
        price=inf['currentPrice']
        return price
    except:
        print('error in current_price')


def recomendation(ticker):                                      #buy or sell ratings
    try:
        ticker_obj = yf.Ticker(ticker)
        recommendations_df = ticker_obj.recommendations  
        return recommendations_df.iloc[0]  
    except:
        print('error in recomendation')          
def revenue(ticker):
    try:
        ticker_obj=yf.Ticker(ticker)
        income_stmt_df = ticker_obj.income_stmt
        total_revenue=income_stmt_df.loc['Total Revenue']
        total_revenue=total_revenue.dropna()
        return total_revenue
    except:
        print('error in revenue')
def earnings(ticker):
    try:
        ticker_obj=yf.Ticker(ticker)
        income_stmt_df = ticker_obj.income_stmt
        net_income=income_stmt_df.loc['Net Income']
        net_income=net_income.dropna()
        return net_income
    except:
        print('error in earnings')


def cash_flow_graph(ticker):
        try:
            data = free_cash_flow(ticker)  
            current_year = datetime.now().year
            
            years = []
            for i in range(len(data)):
                years.append(current_year - len(data) + i)  

            plt.style.use('bmh')
            plt.figure(figsize=(15, 6))

            x = years[::-1]
            y = data
            plt.title(f"{ticker} Cash Flow")
            plt.xlabel('Year', fontsize=18)
            plt.ylabel('USD', fontsize=16)

            plt.scatter(x, y)
            plt.plot(x, y)
            plt.ticklabel_format(style='plain', axis='y', useOffset=False)
            plt.gca().invert_xaxis()
            output_dir = r"C:\Users\pc\Documents\cestakodovanim\learning\myenv\graphs_folder"
            update_date=str(datetime.now().year)+str(datetime.now().month)
            file_path = os.path.join(output_dir, f"{ticker}_{update_date}_cashflow.png")
            plt.savefig(file_path)
        except:
            print('error in cash flow graph')
    

def revenueVSearnings(ticker):
        try:
            total_revenue=revenue(ticker)
            net_income=earnings(ticker)
            current_year=datetime.now().year
            years=[]
            for i in range(len(total_revenue)):                             #nan is problem
                years.append(str(current_year-(i+1)))
            
            df_plot = pd.DataFrame({"Revenue": total_revenue, "Net Income": net_income}).T
            ax=df_plot.T.plot(kind='bar', figsize=(15,6))
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xticklabels(years, rotation=0, ha="right")
            plt.title(f"{ticker} Revenue vs. Net Income")
            plt.xlabel("Fiscal Period")
            plt.ylabel("USD")
            plt.ticklabel_format(style='plain', axis='y', useOffset=False)
            output_dir = r"C:\Users\pc\Documents\cestakodovanim\learning\myenv\graphs_folder"
            update_date=str(datetime.now().year)+str(datetime.now().month)
            file_path = os.path.join(output_dir, f"{ticker}_{update_date}_revenue.png")
            plt.savefig(file_path)
        except:
            print('error in revenue vs earnings')

def margin_graph(ticker):
        try:
            
            total_revenue=revenue(ticker)
            net_income=earnings(ticker) 
            current_year = datetime.now().year
            data=[]
            for i in range(len(total_revenue)):
                margin=(net_income.iloc[i]/total_revenue.iloc[i])*100
                data.append(margin)
            years = []
            for i in range(len(data)):
                years.append(current_year - len(data) + i)  

            plt.style.use('bmh')
            plt.figure(figsize=(15, 6))

            x = years
            y = data
            plt.title(f"{ticker} Margin")
            plt.xlabel('Year', fontsize=18)
            plt.ylabel('Margin in %', fontsize=16)
            plt.scatter(x, y)
            plt.plot(x, y)
            plt.ticklabel_format(style='plain', axis='y', useOffset=False)
            plt.gca().invert_xaxis()
            output_dir = r"C:\Users\pc\Documents\cestakodovanim\learning\myenv\graphs_folder"
            update_date=str(datetime.now().year)+str(datetime.now().month)
            file_path = os.path.join(output_dir, f"{ticker}_{update_date}_margin.png")
            plt.savefig(file_path)
        except:
            print('error in margin graph')


def debt_graph(ticker):
        try:
            ticker_obj=yf.Ticker(ticker)
            balance_sht=ticker_obj.balance_sheet
            data=balance_sht.loc['Total Debt']
            

            plt.style.use('bmh')
            plt.figure(figsize=(15, 6))

            x = data.index
            y = data
            plt.title(f"{ticker} Debt level")
            plt.xlabel('Year', fontsize=18)
            plt.ylabel('USD', fontsize=16)

            plt.scatter(x, y)
            plt.plot(x, y)
            plt.ticklabel_format(style='plain', axis='y', useOffset=False)
            plt.gca().invert_xaxis()
            output_dir = r"C:\Users\pc\Documents\cestakodovanim\learning\myenv\graphs_folder"
            update_date=str(datetime.now().year)+str(datetime.now().month)
            file_path = os.path.join(output_dir, f"{ticker}_{update_date}_debt.png")
            plt.savefig(file_path)
        except:
            print('error in debt graph')

    
def short_info(ticker):
    ticker_obj=yf.Ticker(ticker)
    inf=ticker_obj.info
    return inf.get('longBusinessSummary')

def divident_yield_func(ticker):
    try:
        ticker_obj=yf.Ticker(ticker)
        inf=ticker_obj.info
        dividend_yield=inf.get('dividendYield',0)
        return dividend_yield
    except:
        print('error in divident_yield')


def dividend_graph(ticker):
    try:
        ticker_obj=yf.Ticker(ticker)
        data=ticker_obj.dividends
        

        plt.style.use('bmh')
        plt.figure(figsize=(15, 6))

        x = data.index
        y = data
        plt.title(f"{ticker} Dividend history")
        plt.xlabel('Year', fontsize=18)
        plt.ylabel('USD', fontsize=16)

        plt.scatter(x, y)
        plt.plot(x, y)
        plt.ticklabel_format(style='plain', axis='y', useOffset=False)
        plt.gca().invert_xaxis()
        output_dir = r"C:\Users\pc\Documents\cestakodovanim\learning\myenv\graphs_folder"
        update_date=str(datetime.now().year)+str(datetime.now().month)
        file_path = os.path.join(output_dir, f"{ticker}_{update_date}_dividend.png")
        plt.savefig(file_path)
    except:
        print('error in debt graph')



def price_graph(ticker):
    try:
        timeframes = {
        "1 Day": "1d",
        "1 Month": "1mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "5 Years": "5y",
        "Max": "max"
        }

        # Fetch stock data for different timeframes
        stock_data = {label: yf.Ticker(ticker).history(period=period) for label, period in timeframes.items()}

        # Create figure
        fig = go.Figure()

        # Add traces for each timeframe
        for label, data in stock_data.items():
            fig.add_trace(go.Scatter(
                x=data.index, y=data["Close"], mode="lines",
                name=label, visible=(label == "Max")  # Default view is 1 Year
            ))

        # Create dropdown menu
        dropdown_buttons = [
            dict(
                label=label,
                method="update",
                args=[{"visible": [lbl == label for lbl in stock_data.keys()]}]
            ) for label in stock_data.keys()
        ]

        # Update layout
        fig.update_layout(
            title=f"{ticker} Stock Price - Select Timeframe",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            updatemenus=[{"buttons": dropdown_buttons, "direction": "down", "showactive": True}]
        )

        return fig
    except:
        print('error in price graph')
def get_stock_news(ticker):
    try:
        """ Fetch latest stock news including images, titles, and summaries """
        news_api_key='f1792c1facb7448abf7db17ef630a05c'
        url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={news_api_key}"
        response = requests.get(url)

        if response.status_code != 200:
            return []  # If API call fails, return empty

        articles = response.json().get("articles", [])[:5]  # Get top 5 articles

        news_list = []
        for article in articles:
            title = article.get("title", "No Title Available")
            link = article.get("url", "#")  # If no link, use "#"
            summary = article.get("description") or "No Summary Available"  # Handle missing summary
            image_url = article.get("urlToImage") or "https://via.placeholder.com/400"  # Placeholder for missing image

            news_list.append({"title": title, "link": link, "summary": summary, "image": image_url})

        return news_list
    except:
        print('error in stock news')



def get_stock_data(ticker):
    try:
        """ Fetch historical stock data using yfinance """
        stock = yf.Ticker(ticker)
        return stock.history(period="5y")  # Fetch 5 years of data
    except: 
        print('error in get stock data')


def train_lstm_model(data):
    try:
            
        """ Train an LSTM model on stock data """
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

        # Prepare training data
        x_train, y_train = [], []
        for i in range(60, len(scaled_data) - 1):
            x_train.append(scaled_data[i - 60:i, 0])
            y_train.append(scaled_data[i + 1, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Build LSTM Model
        model = Sequential([
        Input(shape=(x_train.shape[1], 1)),  # Explicitly define the input shape
        LSTM(50, return_sequences=True),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])

        # Compile and Train
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1)

        return model, scaler
    except:
        print('error in train lstm model')


def predict_future_prices(model, scaler, data, days=7):
    try:
        """ Predict stock prices for the next N days """
        future_prices = []
        last_60_days = data["Close"].values[-60:]

        for _ in range(days):
            scaled_data = scaler.transform(last_60_days.reshape(-1, 1))
            x_test = np.array([scaled_data])
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

            predicted_price = model.predict(x_test)
            predicted_price = scaler.inverse_transform(predicted_price)

            future_prices.append(predicted_price[0][0])
            last_60_days = np.append(last_60_days[1:], predicted_price[0][0])  # Append new prediction

        return future_prices
    except:
        print('error in predict future orices')

def generate_openai_analysis(ticker):
    try:
        prompt = f"""
        Provide a concise, professional stock analysis for {ticker}
        
        
        Include insights on the company's performance, growth potential, and any risks based on the financial metrics provided. 
        End with an investment sentiment (bullish, bearish, or neutral).

        Note: This analysis is AI-generated and not financial advice.
        """

        client = OpenAI(api_key=openai_key)  # You can omit api_key if set as an environment variable

    # Create Chat Completion
        chat_completion = client.chat.completions.create(
            model="gpt-4o",  # Using the latest GPT-4o model
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        )

    # Display the response
        return chat_completion.choices[0].message.content
    except: print('error in openai analysis')

def data_sort(series_data):
    try:
        pd.set_option('display.float_format', '{:.0f}'.format)
        df = series_data.reset_index()
        df.columns = ['Date', 'Value']
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        return df
    except:
        print('error in data sort')


def alles_gut(ticker):
    print(f'free cash flow {free_cash_flow(ticker)}')
    print(f'fcff value {fcff(ticker)}')
    print(f'fcff forecast {fcff_forecast(ticker,5,0.08)}')
    print(f'wacc {WACC(ticker)}')
    print(f'shares  {shares_num(ticker)}')
    print(f'net debt {net_debt(ticker)}')
    print(f'ebit {ebit(ticker)}')
    print(f'tax rate {tax_rate(ticker)}')
    print(f'daa {daa(ticker)}')
    print(f'capital expediture {capital_expediture(ticker)}')
    print(f'change wc {change_wc(ticker)}')
    print(f'net income {earnings(ticker)}')
    print(f'cost of debt {cost_of_debt(ticker)}')
    print(f'cash equivalent  {cash_equivalent(ticker)}')
    print(f'total debt  {debt_func(ticker)}')
    print(f'cost of debt  {cost_of_debt(ticker)}')
    print(f'net debt  {net_debt(ticker)}')
    print(f'dcf_valuation  {dcf_valuation(ticker,years=5,growth=0.03,terminal_growth=0.03)}')


#                                                                       ranking!!!!!!



stock = yf.Ticker('aapl')
daco = stock.info
financials = stock.financials
prev_eps = daco.get("trailingEps", np.nan)

#prev_revenue = financials.loc['Free Cash Flow'].iloc[1]
inficko=daco.get("totalRevenue")
#print(prev_eps)



