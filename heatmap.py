import streamlit as st
import akshare as ak
import efinance as ef
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import os
import requests
import json
import re
import traceback
from datetime import datetime, timedelta
from io import StringIO
# 缓存数据获取函数（减少重复请求）
@st.cache_data(ttl=3600)
def get_board_data():
    try:
        # 获取行业板块名称列表
        board_df = ak.stock_board_industry_name_em()
        
        # 检查返回的数据是否为空
        if board_df is None or board_df.empty:
            st.error("无法获取行业板块列表，请检查网络连接")
            return pd.DataFrame()
            
        # 创建目录以保存缓存数据
        os.makedirs("data_cache", exist_ok=True)
        
        # 确保我们有"板块名称"列，如果没有，尝试重命名
        if "板块名称" not in board_df.columns:
            # 查找可能的列名
            for col in board_df.columns:
                if "名称" in col or "板块" in col or "行业" in col:
                    board_df = board_df.rename(columns={col: "板块名称"})
                    break
            
            # 如果仍然没有找到，使用第一列
            if "板块名称" not in board_df.columns and not board_df.empty:
                first_col = board_df.columns[0]
                board_df = board_df.rename(columns={first_col: "板块名称"})
                st.warning(f"未找到板块名称列，使用'{first_col}'列作为板块名称")

        data_list = []
        success_count = 0
        error_count = 0
        total_boards = len(board_df)
        
        for index, row in board_df.iterrows():
            try:
                # 获取每个板块的历史数据
                df = ak.stock_board_industry_hist_em(
                    symbol = row["板块名称"],
                    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d"),
                    end_date = datetime.now().strftime("%Y%m%d"),
                    adjust = ""
                )
                
                # 确保获取到数据
                if df is not None and not df.empty:
                    latest = df.iloc[-1].to_dict()
                    latest["板块名称"] = row["板块名称"]
                    data_list.append(latest)
                    success_count += 1
                else:
                    error_count += 1
                    
                # 每处理10个板块后，短暂休息以防止API限制
                if (index + 1) % 10 == 0:
                    time.sleep(0.5)
                    
            except Exception as e:
                error_count += 1
                # 只打印前几个错误，避免日志过多
                if error_count <= 3:
                    st.warning(f"获取'{row['板块名称']}'数据失败: {e}")
                continue
                
        # 显示处理结果摘要
        if success_count > 0:
            st.info(f"成功获取 {success_count}/{total_boards} 个板块的数据")
            
            # 将结果转为DataFrame并保存缓存
            result_df = pd.DataFrame(data_list)
            result_df.to_csv("data_cache/board_data.csv", index=False)
            return result_df
        else:
            st.error("所有板块数据获取失败")
            # 尝试加载缓存数据
            try:
                if os.path.exists("data_cache/board_data.csv"):
                    st.info("使用缓存的板块数据")
                    return pd.read_csv("data_cache/board_data.csv")
            except:
                pass
            return pd.DataFrame()
                
    except Exception as e:
        st.error(f"获取板块数据失败: {e}")
        # 尝试加载缓存数据
        try:
            if os.path.exists("data_cache/board_data.csv"):
                st.info("使用缓存的板块数据")
                return pd.read_csv("data_cache/board_data.csv")
        except Exception as cache_e:
            st.warning(f"读取缓存板块数据失败: {cache_e}")
        
        return pd.DataFrame()

# 获取A股股票列表
@st.cache_data(ttl=86400)  # 缓存24小时
def get_stock_list():
    """
    获取A股股票列表
    """
    try:
        # 使用akshare获取股票列表
        stock_list = ak.stock_info_a_code_name()
        
        # 确保列名正确并包含'代码'和'名称'
        if '代码' not in stock_list.columns or '名称' not in stock_list.columns:
            if '证券代码' in stock_list.columns and '证券简称' in stock_list.columns:
                stock_list = stock_list.rename(columns={'证券代码': '代码', '证券简称': '名称'})
            elif 'code' in stock_list.columns and 'name' in stock_list.columns:
                stock_list = stock_list.rename(columns={'code': '代码', 'name': '名称'})
            elif len(stock_list.columns) >= 2:
                # 假设第一列是代码，第二列是名称
                stock_list = stock_list.rename(columns={stock_list.columns[0]: '代码', stock_list.columns[1]: '名称'})
        
        # 移除可能的备份文件生成代码
        # stock_list.to_csv('stock_list_backup.csv', index=False)
        
        return stock_list
    except Exception as e:
        st.warning(f"通过akshare获取股票列表失败: {e}")
    
    # 尝试备用方法1：使用另一个akshare接口
    try:
        # 使用另一个akshare接口
        stock_list = ak.stock_zh_a_spot_em()
        
        # 检查并重命名列
        if '代码' not in stock_list.columns or '名称' not in stock_list.columns:
            if '代码' in stock_list.columns and '名称' in stock_list.columns:
                pass  # 已经有正确的列名
            elif '序号' in stock_list.columns and '代码' in stock_list.columns and '名称' in stock_list.columns:
                stock_list = stock_list[['代码', '名称']]
            elif '代码' in stock_list.columns and '股票简称' in stock_list.columns:
                stock_list = stock_list.rename(columns={'股票简称': '名称'})
            elif '证券代码' in stock_list.columns and '证券简称' in stock_list.columns:
                stock_list = stock_list.rename(columns={'证券代码': '代码', '证券简称': '名称'})
            else:
                # 尝试使用最可能的列
                code_candidates = [col for col in stock_list.columns if '代码' in col or 'code' in col.lower()]
                name_candidates = [col for col in stock_list.columns if '名称' in col or '简称' in col or 'name' in col.lower()]
                
                if code_candidates and name_candidates:
                    stock_list = stock_list[[code_candidates[0], name_candidates[0]]]
                    stock_list = stock_list.rename(columns={code_candidates[0]: '代码', name_candidates[0]: '名称'})
        
        # 移除可能的备份文件生成代码
        # stock_list.to_csv('stock_list_backup.csv', index=False)
        
        return stock_list
    except Exception as e:
        st.warning(f"通过备用方法获取股票列表失败: {e}")
    
    # 尝试备用方法2：从本地文件加载（如果之前成功获取过）
    # try:
    #     stock_list = pd.read_csv('stock_list_backup.csv')
    #     if not stock_list.empty:
    #         return stock_list
    # except Exception as e:
    #     st.warning(f"从本地文件加载股票列表失败: {e}")
    
    # 所有方法都失败，返回一个默认股票列表（包含一些常见大盘股）
    st.warning("无法获取完整的股票列表，使用有限的默认列表")
    default_stocks = [
        {"代码": "000001", "名称": "平安银行"},
        {"代码": "000002", "名称": "万科A"},
        {"代码": "000063", "名称": "中兴通讯"},
        {"代码": "000333", "名称": "美的集团"},
        {"代码": "000651", "名称": "格力电器"},
        {"代码": "000858", "名称": "五粮液"},
        {"代码": "002415", "名称": "海康威视"},
        {"代码": "600036", "名称": "招商银行"},
        {"代码": "600276", "名称": "恒瑞医药"},
        {"代码": "600519", "名称": "贵州茅台"},
        {"代码": "600887", "名称": "伊利股份"},
        {"代码": "601318", "名称": "中国平安"},
        {"代码": "601398", "名称": "工商银行"},
        {"代码": "601857", "名称": "中国石油"},
        {"代码": "603288", "名称": "海天味业"}
    ]
    
    return pd.DataFrame(default_stocks)

# 使用efinance获取股票数据
def get_stock_data_from_efinance(stock_code, start_date, end_date):
    """
    从efinance获取股票K线数据
    """
    # 确保股票代码格式正确（去除可能的后缀）
    stock_code = stock_code.strip().upper().replace('.SH', '').replace('.SZ', '').replace('.BJ', '')
    
    # 缓存文件路径
    os.makedirs("data_cache", exist_ok=True)
    cache_file = f"data_cache/{stock_code}_{start_date}_{end_date}.csv"
    
    # 尝试从缓存加载
    if os.path.exists(cache_file):
        cache_time = os.path.getmtime(cache_file)
        # 如果缓存是今天的数据，直接使用缓存
        if datetime.fromtimestamp(cache_time).date() == datetime.now().date():
            try:
                df = pd.read_csv(cache_file)
                if not df.empty:
                    return df
            except Exception as e:
                st.warning(f"读取缓存数据失败: {e}")
    
    try:
        # 使用efinance获取数据
        df = ef.stock.get_quote_history(stock_code, beg=start_date, end=end_date)
        
        if not df.empty:
            # 重命名列以匹配我们的格式
            df = df.rename(columns={
                '日期': '日期',
                '开盘': '开盘',
                '收盘': '收盘',
                '最高': '最高',
                '最低': '最低',
                '成交量': '成交量',
                '成交额': '成交额',
                '振幅': '振幅',
                '涨跌幅': '涨跌幅',
                '涨跌额': '涨跌额',
                '换手率': '换手率'
            })
            
            # 确保数据类型正确
            numeric_cols = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '涨跌幅', '换手率']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 将涨跌幅转换为小数形式
            if '涨跌幅' in df.columns:
                df['涨跌幅'] = df['涨跌幅'] / 100
            
            # 明确过滤掉非交易日数据
            # 非交易日通常成交量为0或开盘价等于收盘价等于最高价等于最低价
            if '成交量' in df.columns:
                df = df[df['成交量'] > 0]
            
            # 过滤掉明显的无效数据点
            if all(col in df.columns for col in ['开盘', '收盘', '最高', '最低']):
                # 去除开盘、收盘、最高、最低价全部相等且成交量接近于0的记录
                df = df[~((df['开盘'] == df['收盘']) & 
                         (df['收盘'] == df['最高']) & 
                         (df['最高'] == df['最低']) & 
                         (df['成交量'] < 100))]
            
            # 缓存结果
            df.to_csv(cache_file, index=False)
            return df
    except Exception as e:
        st.error(f"efinance获取数据失败: {e}")
        st.error(traceback.format_exc())  # 打印详细错误信息
    
    return pd.DataFrame()

# 从efinance获取基本面数据
def get_fundamental_from_efinance(stock_code):
    """
    从efinance获取股票基本面数据
    """
    # 添加日志选项
    DEBUG_MODE = False  # 设置为False以隐藏详细日志
    
    # 日志函数，只在调试模式下显示
    def debug_log(message, level="info"):
        if DEBUG_MODE:
            if level == "info":
                st.info(message)
            elif level == "warning":
                st.warning(message)
            elif level == "error":
                st.error(message)
            elif level == "success":
                st.success(message)
    
    # 确保股票代码格式正确
    stock_code = stock_code.strip().upper().replace('.SH', '').replace('.SZ', '').replace('.BJ', '')
    
    # 缓存文件路径
    os.makedirs("data_cache", exist_ok=True)
    cache_file = f"data_cache/{stock_code}_fundamental.csv"
    
    # 检查缓存是否存在且是今天的数据
    if os.path.exists(cache_file):
        cache_time = os.path.getmtime(cache_file)
        if datetime.fromtimestamp(cache_time).date() == datetime.now().date():
            try:
                df = pd.read_csv(cache_file)
                if not df.empty:
                    debug_log(f"使用缓存的基本面数据: {stock_code}")
                    return df
            except Exception as e:
                debug_log(f"读取基本面缓存数据失败: {e}", "warning")
    
    try:
        debug_log(f"正在获取 {stock_code} 的基本面数据(efinance)...")
        
        # 尝试获取股票基本信息
        try:
            stock_info = ef.stock.get_base_info(stock_code)
            debug_log(f"成功获取股票基本信息: {type(stock_info)}")
            if stock_info:
                debug_log(f"基本信息包含字段: {stock_info.keys() if isinstance(stock_info, dict) else '非字典格式'}")
        except Exception as e:
            debug_log(f"获取股票基本信息失败: {e}", "warning")
            stock_info = {}
        
        # 尝试获取财务数据
        financial_data = pd.DataFrame()
        history_financial = pd.DataFrame()
        
        try:
            # 获取资产负债表
            balance_sheet = ef.stock.get_balance_sheet(stock_code)
            if not balance_sheet.empty:
                debug_log(f"成功获取资产负债表，包含列: {balance_sheet.columns.tolist()}")
        except Exception as e:
            debug_log(f"获取资产负债表失败: {e}", "warning")
            balance_sheet = pd.DataFrame()
            
        try:
            # 获取利润表
            income_statement = ef.stock.get_income(stock_code)
            if not income_statement.empty:
                debug_log(f"成功获取利润表，包含列: {income_statement.columns.tolist()}")
        except Exception as e:
            debug_log(f"获取利润表失败: {e}", "warning")
            income_statement = pd.DataFrame()
            
        try:
            # 获取现金流量表
            cash_flow = ef.stock.get_cash_flow(stock_code)
            if not cash_flow.empty:
                debug_log(f"成功获取现金流量表，包含列: {cash_flow.columns.tolist()}")
        except Exception as e:
            debug_log(f"获取现金流量表失败: {e}", "warning")
            cash_flow = pd.DataFrame()
        
        # 提取需要的指标
        pe, pb, roe, revenue_growth, profit_growth = None, None, None, None, None
        
        # 从基本信息提取市盈率和市净率
        if isinstance(stock_info, dict):
            # 尝试获取市盈率(TTM)
            pe_keys = ['市盈率(TTM)', '市盈率', 'PE', 'PE(TTM)']
            for key in pe_keys:
                if key in stock_info and stock_info[key] is not None:
                    pe = stock_info[key]
                    debug_log(f"从基本信息中提取到市盈率: {pe} (键: {key})")
                    break
            
            # 尝试获取市净率
            pb_keys = ['市净率', 'PB']
            for key in pb_keys:
                if key in stock_info and stock_info[key] is not None:
                    pb = stock_info[key]
                    debug_log(f"从基本信息中提取到市净率: {pb} (键: {key})")
                    break
            
            # 转换为数值
            if pe and isinstance(pe, str):
                try:
                    pe = float(pe.replace('%', ''))
                except:
                    pe = None
            
            if pb and isinstance(pb, str):
                try:
                    pb = float(pb.replace('%', ''))
                except:
                    pb = None
        
        # 计算ROE (如果能从利润表和资产负债表中提取数据)
        if not income_statement.empty and not balance_sheet.empty:
            try:
                # 获取最新的净利润数据
                if '净利润' in income_statement.columns and not income_statement['净利润'].empty:
                    latest_net_profit = income_statement['净利润'].iloc[0]
                    debug_log(f"最新净利润: {latest_net_profit}")
                    
                    # 获取最新的所有者权益数据
                    equity_cols = [col for col in balance_sheet.columns if '所有者权益' in col or '股东权益' in col]
                    if equity_cols and not balance_sheet[equity_cols[0]].empty:
                        latest_equity = balance_sheet[equity_cols[0]].iloc[0]
                        debug_log(f"最新所有者权益: {latest_equity}")
                        
                        # 计算ROE
                        if isinstance(latest_net_profit, str):
                            latest_net_profit = float(latest_net_profit.replace(',', ''))
                        if isinstance(latest_equity, str):
                            latest_equity = float(latest_equity.replace(',', ''))
                            
                        if latest_equity > 0:
                            roe = (latest_net_profit / latest_equity) * 100
                            debug_log(f"计算得到的ROE: {roe:.2f}%")
            except Exception as e:
                debug_log(f"计算ROE失败: {e}", "warning")
        
        # 计算收入增长率和利润增长率 (如果能从利润表中提取足够的历史数据)
        if not income_statement.empty and len(income_statement) >= 2:
            try:
                # 计算营收增长率
                if '营业收入' in income_statement.columns:
                    current_revenue = income_statement['营业收入'].iloc[0]
                    prev_revenue = income_statement['营业收入'].iloc[1]
                    
                    if isinstance(current_revenue, str):
                        current_revenue = float(current_revenue.replace(',', ''))
                    if isinstance(prev_revenue, str):
                        prev_revenue = float(prev_revenue.replace(',', ''))
                    
                    if prev_revenue > 0:
                        revenue_growth = ((current_revenue / prev_revenue) - 1) * 100
                        debug_log(f"计算得到的营收增长率: {revenue_growth:.2f}%")
                
                # 计算净利润增长率
                if '净利润' in income_statement.columns:
                    current_profit = income_statement['净利润'].iloc[0]
                    prev_profit = income_statement['净利润'].iloc[1]
                    
                    if isinstance(current_profit, str):
                        current_profit = float(current_profit.replace(',', ''))
                    if isinstance(prev_profit, str):
                        prev_profit = float(prev_profit.replace(',', ''))
                    
                    if prev_profit > 0:
                        profit_growth = ((current_profit / prev_profit) - 1) * 100
                        debug_log(f"计算得到的净利润增长率: {profit_growth:.2f}%")
            except Exception as e:
                debug_log(f"计算增长率失败: {e}", "warning")
        
        # 尝试从行情数据获取市盈率和市净率（如果之前没有获取到）
        if pe is None or pb is None:
            try:
                # 尝试获取行情数据
                quote_df = ef.stock.get_quote_snapshot(stock_code)
                debug_log(f"行情快照数据类型: {type(quote_df)}")
                
                if isinstance(quote_df, dict):
                    # 如果返回的是字典，尝试直接提取PE和PB
                    for key in quote_df.keys():
                        if pe is None and ('市盈率' in key or 'PE' in key.upper()):
                            pe = quote_df[key]
                            debug_log(f"从行情快照中提取到市盈率: {pe} (键: {key})")
                        
                        if pb is None and ('市净率' in key or 'PB' in key.upper()):
                            pb = quote_df[key]
                            debug_log(f"从行情快照中提取到市净率: {pb} (键: {key})")
                
                elif isinstance(quote_df, pd.DataFrame) and not quote_df.empty:
                    # 如果是DataFrame，寻找相关列
                    for col in quote_df.columns:
                        if pe is None and ('市盈率' in col or 'PE' in col.upper()):
                            pe = quote_df[col].iloc[0]
                            debug_log(f"从行情快照DataFrame中提取到市盈率: {pe} (列: {col})")
                            
                        if pb is None and ('市净率' in col or 'PB' in col.upper()):
                            pb = quote_df[col].iloc[0]
                            debug_log(f"从行情快照DataFrame中提取到市净率: {pb} (列: {col})")
            except Exception as e:
                debug_log(f"获取行情快照失败: {e}", "warning")
        
        # 使用默认值填充缺失指标
        if pe is None or not isinstance(pe, (int, float)) or pd.isna(pe):
            debug_log("使用默认PE值20.0", "warning")
            pe = 20.0
        if pb is None or not isinstance(pb, (int, float)) or pd.isna(pb): 
            debug_log("使用默认PB值2.0", "warning")
            pb = 2.0
        if roe is None or not isinstance(roe, (int, float)) or pd.isna(roe): 
            debug_log("使用默认ROE值10.0", "warning")
            roe = 10.0
        if revenue_growth is None or not isinstance(revenue_growth, (int, float)) or pd.isna(revenue_growth): 
            debug_log("使用默认营收增长率5.0%", "warning")
            revenue_growth = 5.0
        if profit_growth is None or not isinstance(profit_growth, (int, float)) or pd.isna(profit_growth): 
            debug_log("使用默认净利润增长率5.0%", "warning")
            profit_growth = 5.0
        
        # 创建结果DataFrame
        result = {
            "股票代码": stock_code,
            "市盈率(动态)": float(pe),
            "市净率": float(pb),
            "ROE": float(roe),
            "营收增长率(%)": float(revenue_growth),
            "净利润增长率(%)": float(profit_growth)
        }
        
        result_df = pd.DataFrame([result])
        result_df.to_csv(cache_file, index=False)
        
        debug_log(f"成功获取并保存 {stock_code} 的基本面数据", "success")
        return result_df
        
    except Exception as e:
        debug_log(f"efinance获取基本面数据失败: {e}", "error")
        if DEBUG_MODE:
            st.error(traceback.format_exc())  # 仅在调试模式下显示详细错误信息
    
    # 返回默认估计值
    debug_log(f"无法获取 {stock_code} 的基本面数据，使用默认值", "warning")
    return pd.DataFrame([{
        "股票代码": stock_code,
        "市盈率(动态)": 20.0,
        "市净率": 2.0,
        "ROE": 10.0,
        "营收增长率(%)": 5.0,
        "净利润增长率(%)": 5.0
    }])

# 更新获取个股K线数据函数，改用efinance接口
@st.cache_data(ttl=3600)
def get_stock_data(stock_code, start_date, end_date):
    """
    获取股票K线数据，优先使用efinance接口
    """
    # 使用spinner替代直接显示info消息
    with st.spinner(f"正在获取 {stock_code} 的行情数据..."):
        # 使用efinance接口获取数据
        df = get_stock_data_from_efinance(stock_code, start_date, end_date)
        
        if not df.empty:
            # 确保数据是按日期排序的
            if '日期' in df.columns:
                df = df.sort_values(by='日期')
            return df
        
        # 如果efinance接口失败，尝试新浪接口
        try:
            df = get_stock_data_from_sina(stock_code, start_date, end_date)
            
            if not df.empty:
                # 过滤非交易日（成交量为0的记录）
                if '成交量' in df.columns:
                    df = df[df['成交量'] > 0]
                
                # 确保数据是按日期排序的
                if '日期' in df.columns:
                    df = df.sort_values(by='日期')
                return df
        except Exception:
            pass
        
        # 如果新浪接口也失败，尝试akshare
        try:
            # 格式化股票代码
            formatted_code = format_stock_code(stock_code)
            
            # 尝试使用akshare获取数据
            df = ak.stock_zh_a_hist(symbol=formatted_code, start_date=start_date, end_date=end_date, adjust="qfq")
            
            if not df.empty:
                # 过滤非交易日（成交量为0的记录）
                if '成交量' in df.columns:
                    df = df[df['成交量'] > 0]
                
                # 确保数据是按日期排序的
                if '日期' in df.columns:
                    df = df.sort_values(by='日期')
                
                # 缓存数据
                cache_file = f"data_cache/{stock_code}_{start_date}_{end_date}.csv"
                df.to_csv(cache_file, index=False)
                return df
        except Exception:
            # 捕获但不显示错误信息
            pass
        
        # 所有方法都失败，返回空DataFrame
        return pd.DataFrame()

# 更新获取基本面数据函数，改用efinance接口
@st.cache_data(ttl=86400)
def get_stock_fundamental(stock_code):
    """
    获取股票基本面数据，优先使用efinance接口
    """
    # 使用可被隐藏的提示信息
    with st.spinner(f"正在获取 {stock_code} 的基本面数据..."):
        # 使用efinance接口获取数据
        df = get_fundamental_from_efinance(stock_code)
        
        if not df.empty and not all(pd.isna(df.iloc[0])):
            return df
        
        # 如果efinance接口失败，尝试新浪接口
        df = get_fundamental_from_sina(stock_code)
        
        if not df.empty and not all(pd.isna(df.iloc[0])):
            return df
        
        # 如果新浪接口失败，尝试原来的akshare方法
        try:
            # 格式化股票代码
            formatted_code = format_stock_code(stock_code)
            
            # 尝试使用akshare获取数据
            financial = ak.stock_financial_analysis_indicator(symbol=formatted_code)
            
            if financial is None or financial.empty:
                financial = ak.stock_financial_analysis_indicator(symbol=stock_code)
            
            if not financial.empty:
                latest_financial = financial.iloc[0]
                
                try:
                    stock_info = ak.stock_a_lg_indicator(symbol=formatted_code)
                    
                    if stock_info is None or stock_info.empty:
                        stock_info = ak.stock_a_lg_indicator(symbol=stock_code)
                except:
                    stock_info = pd.DataFrame()
                
                # 提取指标
                pe = None
                pb = None
                
                if not stock_info.empty:
                    if '市盈率(动态)' in stock_info.columns:
                        pe = stock_info['市盈率(动态)'].values[0]
                    elif '指标名称' in stock_info.columns:
                        pe_row = stock_info.loc[stock_info['指标名称'] == '市盈率(动态)']
                        if not pe_row.empty:
                            pe = pe_row['最新值'].values[0]
                    
                    if '市净率' in stock_info.columns:
                        pb = stock_info['市净率'].values[0]
                    elif '指标名称' in stock_info.columns:
                        pb_row = stock_info.loc[stock_info['指标名称'] == '市净率']
                        if not pb_row.empty:
                            pb = pb_row['最新值'].values[0]
                
                # 合并数据
                result = {
                    "股票代码": stock_code,
                    "市盈率(动态)": pe,
                    "市净率": pb,
                    "ROE": latest_financial.get('净资产收益率加权(%)'),
                    "营收增长率(%)": latest_financial.get('营业收入同比增长率(%)'),
                    "净利润增长率(%)": latest_financial.get('净利润同比增长率(%)')
                }
                
                result_df = pd.DataFrame([result])
                
                # 缓存数据
                cache_file = f"data_cache/{stock_code}_fundamental.csv"
                result_df.to_csv(cache_file, index=False)
                
                return result_df
        except Exception as e:
            # 捕获但不显示错误
            pass
    
    # 所有方法都失败时返回默认值
    return pd.DataFrame([{
        "股票代码": stock_code,
        "市盈率(动态)": 20,  # 默认估计值
        "市净率": 2,  # 默认估计值
        "ROE": 10,  # 默认估计值
        "营收增长率(%)": 5,  # 默认估计值
        "净利润增长率(%)": 5  # 默认估计值
    }])

# 格式化股票代码以匹配akshare接口要求
def format_stock_code(code):
    # 移除可能的前缀和多余的空格
    code = code.strip()
    
    # 检查是否已经有交易所前缀
    if code.startswith(('sh', 'sz', 'bj')):
        return code
        
    # 根据股票代码规则添加交易所前缀
    if code.startswith('6'):
        # 上海证券交易所
        return f"{code}.SH"
    elif code.startswith(('0', '3')):
        # 深圳证券交易所
        return f"{code}.SZ"
    elif code.startswith(('4', '8')):
        # 北京证券交易所
        return f"{code}.BJ"
    else:
        # 默认返回原代码
        return code

# 计算技术指标
def calculate_indicators(df):
    if df.empty:
        return df
    
    # 确保必要的列存在
    required_cols = ['收盘', '开盘', '最高', '最低', '成交量']
    if not all(col in df.columns for col in required_cols):
        return df
    
    # 计算MA
    df['MA5'] = df['收盘'].rolling(5).mean()
    df['MA10'] = df['收盘'].rolling(10).mean()
    df['MA20'] = df['收盘'].rolling(20).mean()
    
    # 计算MACD
    df['EMA12'] = df['收盘'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['收盘'].ewm(span=26, adjust=False).mean()
    df['DIF'] = df['EMA12'] - df['EMA26']
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = 2 * (df['DIF'] - df['DEA'])
    
    # 计算KDJ
    low_9 = df['最低'].rolling(window=9).min()
    high_9 = df['最高'].rolling(window=9).max()
    df['RSV'] = (df['收盘'] - low_9) / (high_9 - low_9) * 100
    df['K'] = df['RSV'].ewm(alpha=1/3, adjust=False).mean()
    df['D'] = df['K'].ewm(alpha=1/3, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    # 计算RSI
    delta = df['收盘'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 计算威廉指标WR (21日)，并取绝对值
    period = 21
    highest_high = df['最高'].rolling(window=period).max()
    lowest_low = df['最低'].rolling(window=period).min()
    # 计算原始WR值，然后取绝对值
    df['WR21'] = abs(-100 * ((highest_high - df['收盘']) / (highest_high - lowest_low)))
    
    return df

# 数据处理函数
def process_data(df):
    numeric_cols = ['开盘','收盘','最高','最低','成交量','成交额','振幅','涨跌幅','换手率']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    df['量价强度'] = df['涨跌幅'] * df['换手率']
    df['成交额（亿）'] = df['成交额'] / 1e8
    df['成交量（万手）'] = df['成交量'] / 10000
    df['涨跌幅'] = df['涨跌幅'] * 100 # 确保为百分比值
    # 新增四舍五入处理（保留两位小数）
    round_cols = ['涨跌幅', '换手率', '量价强度', '成交额（亿）', '成交量（万手）']
    df[round_cols] = df[round_cols].round(0)
    df['涨跌幅'] = df['涨跌幅'] / 100  # 确保为百分比值
    return df.dropna(subset=['涨跌幅'])

# 修改选股函数，使用efinance获取基本面数据
def filter_stocks(pe_min, pe_max, pb_min, pb_max, roe_min, growth_min, max_stocks=200):
    st.info("正在筛选股票，请稍候...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 获取股票列表
    stock_list = get_stock_list()
    
    if stock_list.empty:
        st.error("无法获取股票列表，选股功能无法继续")
        return pd.DataFrame()
    
    # 限制处理的股票数量以提高性能
    if len(stock_list) > max_stocks:
        st.warning(f"为提高性能，将只筛选前 {max_stocks} 只股票")
        stock_list = stock_list.head(max_stocks)
    
    filtered_stocks = []
    total_stocks = len(stock_list)
    processed_count = 0
    error_count = 0
    
    # 尝试从缓存文件中获取已处理的结果
    cache_dir = "data_cache/fundamentals"
    os.makedirs(cache_dir, exist_ok=True)
    
    # 创建一个包含所有筛选条件的标识符
    filter_id = f"pe_{pe_min}_{pe_max}_pb_{pb_min}_{pb_max}_roe_{roe_min}_growth_{growth_min}"
    cache_file = f"{cache_dir}/filter_{filter_id}.csv"
    
    # 检查是否有最近的缓存结果(1小时内)
    if os.path.exists(cache_file):
        cache_time = os.path.getmtime(cache_file)
        if (time.time() - cache_time) < 3600:  # 1小时内的缓存
            try:
                cache_df = pd.read_csv(cache_file)
                st.success("使用缓存的筛选结果")
                return cache_df
            except:
                pass
    
    for i, (_, row) in enumerate(stock_list.iterrows()):
        # 更新进度条和状态
        progress = min(i / total_stocks, 1.0)
        progress_bar.progress(progress)
        processed_count += 1
        
        if i % 5 == 0:  # 每处理5只股票更新一次状态
            status_text.text(f"已处理: {processed_count}/{total_stocks} | 符合条件: {len(filtered_stocks)} | 错误: {error_count}")
        
        try:
            stock_code = row['代码']
            
            # 检查是否有个股缓存数据
            fund_cache_file = f"data_cache/{stock_code}_fundamental.csv"
            if os.path.exists(fund_cache_file):
                try:
                    fund_data = pd.read_csv(fund_cache_file)
                except:
                    # 获取基本面数据 - 优先使用efinance
                    fund_data = get_fundamental_from_efinance(stock_code)
            else:
                # 获取基本面数据 - 优先使用efinance
                fund_data = get_fundamental_from_efinance(stock_code)
                
                # 如果efinance接口失败，尝试其他方法
                if fund_data.empty or all(pd.isna(fund_data.iloc[0])):
                    fund_data = get_stock_fundamental(stock_code)
            
            if fund_data.empty:
                continue
                
            # 应用筛选条件，增加健壮性检查
            valid_data = True
            required_fields = ['市盈率(动态)', '市净率', 'ROE', '营收增长率(%)', '净利润增长率(%)']
            
            # 检查是否所有必需字段都存在且不是NaN
            for field in required_fields:
                if field not in fund_data.columns or pd.isna(fund_data.iloc[0][field]):
                    valid_data = False
                    break
            
            if valid_data:
                pe = float(fund_data.iloc[0]['市盈率(动态)'])
                pb = float(fund_data.iloc[0]['市净率'])
                roe = float(fund_data.iloc[0]['ROE'])
                growth = float(fund_data.iloc[0]['营收增长率(%)'])
                
                # 排除负值和异常值
                if pe < 0 or pb < 0 or pe > 2000 or pb > 100:
                    continue
                
                if (pe_min <= pe <= pe_max or pe_min == pe_max == 0) and \
                   (pb_min <= pb <= pb_max or pb_min == pb_max == 0) and \
                   roe >= roe_min and growth >= growth_min:
                    filtered_stocks.append({
                        "代码": stock_code,
                        "名称": row['名称'],
                        "市盈率": pe,
                        "市净率": pb,
                        "ROE(%)": roe,
                        "营收增长率(%)": growth,
                        "净利润增长率(%)": fund_data.iloc[0]['净利润增长率(%)']
                    })
            
            # 为避免频繁请求导致API限制，添加短暂延迟
            if i % 10 == 0 and i > 0:
                time.sleep(0.5)
                
        except Exception as e:
            error_count += 1
            # 如果错误太多，提前终止
            if error_count > 20:
                st.warning(f"遇到过多错误，提前终止筛选: {e}")
                break
            continue
    
    progress_bar.progress(1.0)
    status_text.text(f"筛选完成，共找到 {len(filtered_stocks)} 只符合条件的股票")
    
    if not filtered_stocks:
        st.warning("没有找到符合条件的股票，请尝试放宽筛选条件")
        return pd.DataFrame()
    
    # 保存结果到缓存
    result_df = pd.DataFrame(filtered_stocks)
    result_df.to_csv(cache_file, index=False)
    
    return result_df

# 辅助函数，获取月份标签
def get_monthly_ticks(df):
    """
    从日期数据中提取每个月的第一个交易日作为刻度标签
    """
    if '月份' not in df.columns:
        df = df.copy()
    
    df['月份'] = pd.to_datetime(df['日期']).dt.strftime('%Y-%m')
    monthly_ticks = []
    monthly_labels = []
    prev_month = None
    
    for i, row in df.iterrows():
        current_month = row['月份']
        if current_month != prev_month:
            monthly_ticks.append(row['日期'])
            monthly_labels.append(current_month)
            prev_month = current_month
    
    return monthly_ticks, monthly_labels

# 绘制K线图
def plot_candlestick(df):
    if df.empty:
        return go.Figure()
    
    # 创建蜡烛图
    fig = go.Figure()
    
    # 添加K线图
    fig.add_trace(go.Candlestick(
        x=df['日期'],
        open=df['开盘'],
        high=df['最高'],
        low=df['最低'],
        close=df['收盘'],
        name='K线'
    ))
    
    # 添加MA线
    if 'MA5' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['日期'],
            y=df['MA5'],
            mode='lines',
            name='MA5',
            line=dict(color='blue', width=1)
        ))
    
    if 'MA10' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['日期'],
            y=df['MA10'],
            mode='lines',
            name='MA10',
            line=dict(color='orange', width=1)
        ))
    
    if 'MA20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['日期'],
            y=df['MA20'],
            mode='lines',
            name='MA20',
            line=dict(color='purple', width=1)
        ))
    
    # 获取月份标签
    monthly_ticks, monthly_labels = get_monthly_ticks(df)
    
    # 设置图表布局
    fig.update_layout(
        title='K线图表',
        xaxis_title='日期',
        yaxis_title='价格',
        xaxis_rangeslider_visible=False,
        height=500,
        # 配置x轴只显示每个月的第一个交易日
        xaxis=dict(
            type='category',  # 使用category类型而不是date类型
            categoryorder='array',
            categoryarray=df['日期'].tolist(),  # 确保只显示有数据的日期
            tickmode='array',
            tickvals=monthly_ticks,  # 只在月份变化时显示刻度
            ticktext=monthly_labels,  # 使用月份作为刻度标签
        ),
        # 添加网格线以提高可读性
        xaxis_gridcolor='lightgrey',
        yaxis_gridcolor='lightgrey',
        plot_bgcolor='white'
    )
    
    # 设置连线仅在有数据点处显示，不跨越非交易日
    for trace in fig.data:
        if trace.type == 'scatter':  # 确保只修改折线图，不影响K线图
            trace.connectgaps = False
    
    return fig

# 绘制成交量图
def plot_volume(df):
    if df.empty:
        return go.Figure()
    
    colors = ['red' if row['收盘'] - row['开盘'] >= 0 else 'green' for _, row in df.iterrows()]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['日期'],
        y=df['成交量'],
        marker_color=colors,
        name='成交量'
    ))
    
    # 获取月份标签
    monthly_ticks, monthly_labels = get_monthly_ticks(df)
    
    fig.update_layout(
        title='成交量',
        xaxis_title='日期',
        yaxis_title='成交量',
        height=200,
        # 配置x轴只显示每个月的第一个交易日
        xaxis=dict(
            type='category',  # 使用category类型而不是date类型
            categoryorder='array',
            categoryarray=df['日期'].tolist(),  # 确保只显示有数据的日期
            tickmode='array',
            tickvals=monthly_ticks,  # 只在月份变化时显示刻度
            ticktext=monthly_labels,  # 使用月份作为刻度标签
        ),
        # 添加网格线以提高可读性
        xaxis_gridcolor='lightgrey',
        yaxis_gridcolor='lightgrey',
        plot_bgcolor='white'
    )
    
    return fig

# 主程序
def main():
    st.set_page_config(
        page_title="股票分析平台",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 创建选项卡
    tab1, tab2, tab3 = st.tabs(["板块热力图", "个股分析", "选股工具"])
    
    # 热力图选项卡
    with tab1:
        st.title("📈 实时板块资金流向热力图")
        st.markdown("""
        **数据说明：**
        - 颜色映射：绿色表示下跌，红色表示上涨
        - 数据更新：{}
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M")))

        # 侧边栏控件
        col1, col2 = st.columns(2)
        with col1:
            color_metric = st.selectbox(
                "颜色指标",
                options=['涨跌幅','换手率','量价强度'],
                index=0,
                key="heatmap_color"
            )
            size_metric = st.selectbox(
                "板块大小指标",
                options=['成交额（亿）','成交量（万手）','换手率'],
                index=0,
                key="heatmap_size"
            )
        with col2:
            date_range = st.slider(
                "回溯天数",
                min_value=1,
                max_value=30,
                value=7,
                key="heatmap_days"
            )
            color_scale = st.selectbox(
                "配色方案",
                options=['RdYlGn_r','BrBG_r','PiYG_r','RdBu_r'], # 全部使用反转色阶
                index=0,
                key="heatmap_color_scale"
            )

        # 数据加载
        with st.spinner('正在获取最新行情数据...'):
            raw_df = get_board_data()
            processed_df = process_data(raw_df)

        # 数据过滤
        filtered_df = processed_df[
            processed_df['日期'] >= (datetime.now() - timedelta(days=date_range)).strftime("%Y-%m-%d")
            ]

        # 创建可视化
        fig = px.treemap(
            filtered_df,
            path=['板块名称'],
            values=size_metric,
            color=color_metric,
            color_continuous_scale=color_scale,
            range_color=[filtered_df[color_metric].min(), filtered_df[color_metric].max()],
            hover_data={
                '涨跌幅':':.2f%',
                '换手率':':.2f%',
                '成交额（亿）':':.2f',
                '量价强度':':.2f'
            },
            height=600
        )

        # 样式调整
        fig.update_layout(
            margin=dict(t=0, l=0, r=0, b=0),
            coloraxis_colorbar=dict(
                title=color_metric + (" (%)"if color_metric =="涨跌幅"else""),
                tickformat=".1f"if color_metric =="涨跌幅"else".1f",
                thickness=15
            )
        )

        fig.update_traces(
            texttemplate='%{label} %{customdata[0]:.2f} % ',
            hovertemplate = ('<b>%{label}</b>'
                f'{color_metric}: %{{color:.2f}}{"%" if color_metric == "涨跌幅" else ""}'
                '换手率: %{customdata[1]:.2f}%'
                '成交额: %{customdata[2]:.2f}亿'
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # 数据表格
        with st.expander("查看原始数据"):
            st.dataframe(
                filtered_df.sort_values(by='涨跌幅', ascending=False),
                column_config={
                    "日期":"日期",
                    "板块名称": st.column_config.TextColumn(width="large"),
                    "涨跌幅": st.column_config.NumberColumn(format="▁%.2f%%",help="颜色映射："),
                    "换手率": st.column_config.NumberColumn(format="%.2f%%"),
                    "成交额（亿）": st.column_config.NumberColumn(format="%.1f 亿")
                },
                height=300,
                hide_index=True
            )

    # 个股分析选项卡
    with tab2:
        st.title("🔍 个股详细分析")
        
        # 获取股票列表用于搜索
        with st.spinner("正在加载股票列表..."):
            stock_list = get_stock_list()
        
        if stock_list.empty:
            st.error("无法获取股票列表，请检查网络连接或刷新页面重试")
            st.stop()
        
        # 再次确认列名
        if '代码' not in stock_list.columns or '名称' not in stock_list.columns:
            st.error("股票列表数据格式错误，缺少'代码'或'名称'列")
            # 尝试打印出实际的列名以便调试
            st.write("实际列名:", stock_list.columns.tolist())
            
            # 紧急措施：使用前两列并重命名
            if len(stock_list.columns) >= 2:
                cols = stock_list.columns.tolist()
                st.warning(f"尝试将 '{cols[0]}' 列作为'代码'，'{cols[1]}' 列作为'名称'")
                stock_list = stock_list.rename(columns={
                    cols[0]: '代码',
                    cols[1]: '名称'
                })
            else:
                st.stop()
        
        # 创建股票代码和名称的映射字典
        stock_dict = dict(zip(stock_list['名称'], stock_list['代码']))
        
        # 股票选择组件
        col1, col2 = st.columns([3, 1])
        with col1:
            # 添加手动输入股票代码的选项
            input_option = st.radio(
                "选择输入方式",
                options=["从列表选择", "手动输入代码"],
                index=0,
                horizontal=True,
                key="input_method"
            )
            
            if input_option == "从列表选择":
                # 从列表选择股票
                if len(stock_dict) > 0:
                    selected_stock = st.selectbox(
                        "选择股票",
                        options=list(stock_dict.keys()),
                        key="stock_selector"
                    )
                    if selected_stock:
                        stock_code = stock_dict[selected_stock]
                else:
                    st.error("股票列表为空，请使用手动输入方式")
                    selected_stock = None
                    stock_code = None
            else:
                # 手动输入股票代码
                stock_code = st.text_input(
                    "输入股票代码（如：000001）",
                    key="manual_stock_code"
                )
                selected_stock = None
                # 验证输入的股票代码
                if stock_code:
                    # 尝试在列表中找到对应名称
                    matching_stocks = stock_list[stock_list['代码'] == stock_code]
                    if not matching_stocks.empty:
                        selected_stock = matching_stocks.iloc[0]['名称']
                    else:
                        selected_stock = "未知股票"
        
        with col2:
            days = st.number_input(
                "数据周期(天)",
                min_value=5,
                max_value=365,
                value=365,
                key="stock_days"
            )
        
        # 处理股票数据
        if (selected_stock and input_option == "从列表选择") or (stock_code and input_option == "手动输入代码"):
            # 确保stock_code是有效的
            if input_option == "从列表选择":
                stock_code = stock_dict[selected_stock]
            
            # 显示股票名称和代码
            st.subheader(f"{selected_stock or '股票'} ({stock_code})")
            
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
            end_date = datetime.now().strftime("%Y%m%d")
            
            # 获取并处理数据
            with st.spinner(f"正在获取 {selected_stock or stock_code} 数据..."):
                try:
                    stock_data = get_stock_data(stock_code, start_date, end_date)
                    if not stock_data.empty:
                        # 计算技术指标
                        stock_data = calculate_indicators(stock_data)
                except Exception as e:
                    st.error(f"获取数据时出错: {e}")
                    stock_data = pd.DataFrame()
            
            if not stock_data.empty:
                # 显示交易日期范围信息
                if len(stock_data) > 0:
                    earliest_date = stock_data['日期'].min()
                    latest_date = stock_data['日期'].max()
                    trading_days = len(stock_data)
                    st.info(f"数据范围: {earliest_date} 至 {latest_date}，共 {trading_days} 个交易日")
                
                # 绘制K线图
                with st.container():
                    st.subheader("K线图")
                    candlestick_fig = plot_candlestick(stock_data)
                    st.plotly_chart(candlestick_fig, use_container_width=True)
                
                # 绘制成交量图
                with st.container():
                    st.subheader("成交量")
                    volume_fig = plot_volume(stock_data)
                    st.plotly_chart(volume_fig, use_container_width=True)
                
                # 绘制技术指标
                with st.container():
                    st.subheader("技术指标")
                    
                    # 切换成2行2列布局，每行两个指标
                    indicator_row1 = st.columns(2)
                    indicator_row2 = st.columns(2)
                    
                    # 获取月份标签
                    monthly_ticks, monthly_labels = get_monthly_ticks(stock_data)
                    
                    # 第一行第一列：MACD
                    with indicator_row1[0]:
                        st.write("MACD")
                        macd_fig = go.Figure()
                        macd_fig.add_trace(go.Scatter(x=stock_data['日期'], y=stock_data['DIF'], mode='lines', name='DIF', connectgaps=False))
                        macd_fig.add_trace(go.Scatter(x=stock_data['日期'], y=stock_data['DEA'], mode='lines', name='DEA', connectgaps=False))
                        macd_fig.add_trace(go.Bar(x=stock_data['日期'], y=stock_data['MACD'], name='MACD柱'))
                        macd_fig.update_layout(
                            height=300, 
                            margin=dict(l=10, r=10, t=30, b=10),
                            xaxis=dict(
                                type='category',
                                categoryorder='array',
                                categoryarray=stock_data['日期'].tolist(),
                                tickmode='array',
                                tickvals=monthly_ticks,  # 使用月份标签
                                ticktext=monthly_labels,
                            ),
                            plot_bgcolor='white',
                            xaxis_gridcolor='lightgrey',
                            yaxis_gridcolor='lightgrey'
                        )
                        st.plotly_chart(macd_fig, use_container_width=True)
                    
                    # 第一行第二列：KDJ
                    with indicator_row1[1]:
                        st.write("KDJ")
                        kdj_fig = go.Figure()
                        kdj_fig.add_trace(go.Scatter(x=stock_data['日期'], y=stock_data['K'], mode='lines', name='K', connectgaps=False))
                        kdj_fig.add_trace(go.Scatter(x=stock_data['日期'], y=stock_data['D'], mode='lines', name='D', connectgaps=False))
                        kdj_fig.add_trace(go.Scatter(x=stock_data['日期'], y=stock_data['J'], mode='lines', name='J', connectgaps=False))
                        kdj_fig.update_layout(
                            height=300, 
                            margin=dict(l=10, r=10, t=30, b=10),
                            xaxis=dict(
                                type='category',
                                categoryorder='array',
                                categoryarray=stock_data['日期'].tolist(),
                                tickmode='array',
                                tickvals=monthly_ticks,  # 使用月份标签
                                ticktext=monthly_labels,
                            ),
                            plot_bgcolor='white',
                            xaxis_gridcolor='lightgrey',
                            yaxis_gridcolor='lightgrey'
                        )
                        st.plotly_chart(kdj_fig, use_container_width=True)
                    
                    # 第二行第一列：RSI
                    with indicator_row2[0]:
                        st.write("RSI")
                        rsi_fig = go.Figure()
                        rsi_fig.add_trace(go.Scatter(x=stock_data['日期'], y=stock_data['RSI'], mode='lines', name='RSI', connectgaps=False))
                        # 添加超买超卖参考线
                        rsi_fig.add_shape(type="line", x0=stock_data['日期'].iloc[0], y0=70, x1=stock_data['日期'].iloc[-1], y1=70,
                                        line=dict(color="red", width=1, dash="dash"))
                        rsi_fig.add_shape(type="line", x0=stock_data['日期'].iloc[0], y0=30, x1=stock_data['日期'].iloc[-1], y1=30,
                                        line=dict(color="green", width=1, dash="dash"))
                        rsi_fig.update_layout(
                            height=300, 
                            margin=dict(l=10, r=10, t=30, b=10),
                            xaxis=dict(
                                type='category',
                                categoryorder='array',
                                categoryarray=stock_data['日期'].tolist(),
                                tickmode='array',
                                tickvals=monthly_ticks,  # 使用月份标签
                                ticktext=monthly_labels,
                            ),
                            plot_bgcolor='white',
                            xaxis_gridcolor='lightgrey',
                            yaxis_gridcolor='lightgrey'
                        )
                        st.plotly_chart(rsi_fig, use_container_width=True)
                        
                    # 第二行第二列：威廉WR指标
                    with indicator_row2[1]:
                        st.write("威廉WR指标")
                        wr_fig = go.Figure()
                        wr_fig.add_trace(go.Scatter(x=stock_data['日期'], y=stock_data['WR21'], mode='lines', name='WR(21)', connectgaps=False))
                        
                        # 添加超买超卖参考线（取绝对值后，原来的-20变成20，-80变成80）
                        wr_fig.add_shape(type="line", x0=stock_data['日期'].iloc[0], y0=20, x1=stock_data['日期'].iloc[-1], y1=20,
                                        line=dict(color="red", width=1, dash="dash"))
                        wr_fig.add_shape(type="line", x0=stock_data['日期'].iloc[0], y0=80, x1=stock_data['日期'].iloc[-1], y1=80,
                                        line=dict(color="green", width=1, dash="dash"))
                        
                        wr_fig.update_layout(
                            height=300, 
                            margin=dict(l=10, r=10, t=30, b=10),
                            xaxis=dict(
                                type='category',
                                categoryorder='array',
                                categoryarray=stock_data['日期'].tolist(),
                                tickmode='array',
                                tickvals=monthly_ticks,  # 使用月份标签
                                ticktext=monthly_labels,
                            ),
                            yaxis=dict(
                                range=[0, 100],  # 取绝对值后，范围变为0到100
                                autorange=False
                            ),
                            plot_bgcolor='white',
                            xaxis_gridcolor='lightgrey',
                            yaxis_gridcolor='lightgrey'
                        )
                        st.plotly_chart(wr_fig, use_container_width=True)
                
                # 显示近期数据
                with st.expander("查看历史交易数据（仅交易日）"):
                    st.dataframe(
                        stock_data.sort_values('日期', ascending=False),
                        height=300
                    )
            else:
                st.error(f"未能获取到 {selected_stock} 的数据，请尝试其他股票。")
    
    # 选股工具选项卡
    with tab3:
        st.title("🔎 多维度选股工具")
        
        st.markdown("""
        ### 使用说明
        - 设置下面的筛选条件，系统将为您从A股市场筛选符合条件的股票
        - 留空或设置为0表示不限制该条件
        - 为提高性能，系统将只处理部分股票
        - 筛选可能需要一些时间，请耐心等待
        """)
        
        # 筛选条件输入
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("市盈率(PE)")
            pe_min = st.number_input("最小PE", min_value=0.0, max_value=1000.0, value=0.0, step=1.0)
            pe_max = st.number_input("最大PE", min_value=0.0, max_value=1000.0, value=50.0, step=1.0)
        
        with col2:
            st.subheader("市净率(PB)")
            pb_min = st.number_input("最小PB", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
            pb_max = st.number_input("最大PB", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
        
        with col3:
            st.subheader("其他指标")
            roe_min = st.number_input("最小ROE(%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
            growth_min = st.number_input("最小营收增长率(%)", min_value=-100.0, max_value=1000.0, value=5.0, step=1.0)
            
        # 添加高级选项
        with st.expander("高级选项"):
            max_stocks = st.slider("最大处理股票数量", min_value=50, max_value=500, value=200, step=50,
                                help="增加此值会提高筛选结果的全面性，但会降低性能")
        
        # 开始筛选按钮
        if st.button("开始筛选", key="start_filter"):
            # 执行筛选
            result_df = filter_stocks(pe_min, pe_max, pb_min, pb_max, roe_min, growth_min, max_stocks)
            
            # 显示结果
            if not result_df.empty:
                st.success(f"共找到 {len(result_df)} 只符合条件的股票")
                
                # 添加排序选项
                sort_column = st.selectbox(
                    "排序依据",
                    options=["ROE(%)", "市盈率", "市净率", "营收增长率(%)", "净利润增长率(%)"],
                    index=0
                )
                
                sort_order = st.radio(
                    "排序方式",
                    options=["降序", "升序"],
                    index=0,
                    horizontal=True
                )
                
                # 显示筛选结果
                st.dataframe(
                    result_df.sort_values(by=sort_column, ascending=(sort_order=="升序")),
                    column_config={
                        "代码": st.column_config.TextColumn(width="small"),
                        "名称": st.column_config.TextColumn(width="medium"),
                        "市盈率": st.column_config.NumberColumn(format="%.2f"),
                        "市净率": st.column_config.NumberColumn(format="%.2f"),
                        "ROE(%)": st.column_config.NumberColumn(format="%.2f%%"),
                        "营收增长率(%)": st.column_config.NumberColumn(format="%.2f%%"),
                        "净利润增长率(%)": st.column_config.NumberColumn(format="%.2f%%")
                    },
                    height=500,
                    hide_index=True
                )
                
                # 提供导出功能
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="导出为CSV",
                    data=csv,
                    file_name=f"选股结果_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime='text/csv',
                )
            else:
                st.warning("未找到符合条件的股票，请尝试放宽筛选条件")

if __name__ == "__main__":
    main()