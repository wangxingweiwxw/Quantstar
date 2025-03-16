import streamlit as st
import akshare as ak
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import time

# 获取A股股票列表
@st.cache_data(ttl=86400)  # 缓存24小时
def get_stock_list():
    """
    获取A股股票列表
    """
    try:
        # 主要方法：使用akshare获取
        stock_list = ak.stock_info_a_code_name()
        
        # 检查并标准化列名
        cols = stock_list.columns.tolist()
        column_mapping = {}
        
        # 查找可能的"代码"列
        for col in cols:
            if '代码' in col or 'code' in col.lower() or 'symbol' in col.lower():
                column_mapping[col] = '代码'
                break
        
        # 查找可能的"名称"列
        for col in cols:
            if '名称' in col or 'name' in col.lower() or '简称' in col:
                column_mapping[col] = '名称'
                break
        
        # 重命名列
        if column_mapping:
            stock_list = stock_list.rename(columns=column_mapping)
        
        # 验证列是否存在，若不存在则尝试其他方法
        if '代码' not in stock_list.columns or '名称' not in stock_list.columns:
            raise KeyError("未找到'代码'或'名称'列")
            
        return stock_list
    except Exception as e:
        st.warning(f"通过akshare获取股票列表失败: {e}")
    
    # 尝试备用方法：使用不同的akshare接口
    try:
        # 使用另一个akshare接口
        stock_list = ak.stock_zh_a_spot_em()
        if not stock_list.empty:
            # 检查并调整列名
            if '代码' in stock_list.columns and '名称' in stock_list.columns:
                # 如果已经有正确的列名，只保留这两列
                stock_list = stock_list[['代码', '名称']]
            else:
                # 尝试查找并重命名
                cols = stock_list.columns.tolist()
                # 查找代码列
                code_col = None
                for col in cols:
                    if '代码' in col or 'code' in col.lower() or 'symbol' in col.lower():
                        code_col = col
                        break
                
                # 查找名称列
                name_col = None
                for col in cols:
                    if '名称' in col or 'name' in col.lower() or '简称' in col:
                        name_col = col
                        break
                
                if code_col and name_col:
                    # 重命名并只保留这两列
                    stock_list = stock_list.rename(columns={code_col: '代码', name_col: '名称'})
                    stock_list = stock_list[['代码', '名称']]
                else:
                    # 如果找不到合适的列，使用前两列并重命名
                    first_two_cols = stock_list.columns[:2].tolist()
                    stock_list = stock_list.rename(columns={
                        first_two_cols[0]: '代码',
                        first_two_cols[1]: '名称'
                    })
                    stock_list = stock_list[['代码', '名称']]
            
            return stock_list
    except Exception as e:
        st.warning(f"通过备用akshare接口获取股票列表失败: {e}")
    
    # 最后的备选方案：使用硬编码的常用股票列表
    st.error("无法获取完整股票列表，将使用有限的股票列表")
    # 提供一些常用股票作为最低限度的备选
    default_stocks = [
        {"代码": "000001", "名称": "平安银行"},
        {"代码": "000002", "名称": "万科A"},
        {"代码": "000063", "名称": "中兴通讯"},
        {"代码": "000333", "名称": "美的集团"},
        {"代码": "000651", "名称": "格力电器"},
        {"代码": "000858", "名称": "五粮液"},
        {"代码": "002415", "名称": "海康威视"},
        {"代码": "600000", "名称": "浦发银行"},
        {"代码": "600036", "名称": "招商银行"},
        {"代码": "600276", "名称": "恒瑞医药"},
        {"代码": "600519", "名称": "贵州茅台"},
        {"代码": "601318", "名称": "中国平安"},
        {"代码": "601857", "名称": "中国石油"},
        {"代码": "601398", "名称": "工商银行"},
        {"代码": "601988", "名称": "中国银行"},
        {"代码": "603288", "名称": "海天味业"},
        {"代码": "601888", "名称": "中国中免"},
        {"代码": "600050", "名称": "中国联通"},
        {"代码": "600009", "名称": "上海机场"},
        {"代码": "688981", "名称": "中芯国际"}
    ]
    return pd.DataFrame(default_stocks)

# 获取个股K线数据
@st.cache_data(ttl=3600)
def get_stock_data(stock_code, start_date, end_date):
    try:
        df = ak.stock_zh_a_hist(symbol=stock_code, start_date=start_date, end_date=end_date, adjust="qfq")
        return df
    except Exception as e:
        st.error(f"获取股票数据失败: {e}")
        return pd.DataFrame()

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
    
    # 计算威廉WR指标(窗口期21，使用滚动均值)
    highest_high_21_mean = df['最高'].rolling(window=21).mean()
    lowest_low_21_mean = df['最低'].rolling(window=21).mean()
    df['WR21'] = -100 * (highest_high_21_mean - df['收盘']) / (highest_high_21_mean - lowest_low_21_mean)
    
    return df

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
    
    # 设置图表布局
    fig.update_layout(
        title='K线图表',
        xaxis_title='日期',
        yaxis_title='价格',
        xaxis_rangeslider_visible=False,
        height=500
    )
    
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
    
    fig.update_layout(
        title='成交量',
        xaxis_title='日期',
        yaxis_title='成交量',
        height=200
    )
    
    return fig

# 主程序
def main():
    st.set_page_config(
        page_title="个股分析平台",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🔍 个股详细分析")
    
    # 获取股票列表用于搜索
    with st.spinner("正在加载股票列表..."):
        stock_list = get_stock_list()
    
    if stock_list.empty:
        st.error("无法获取股票列表，请检查网络连接或刷新页面重试")
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
            selected_stock = st.selectbox(
                "选择股票",
                options=list(stock_dict.keys()),
                key="stock_selector"
            )
            if selected_stock:
                stock_code = stock_dict[selected_stock]
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
            value=60,
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
            # 创建指标选择行
            indicator_col1, indicator_col2 = st.columns(2)
            with indicator_col1:
                selected_indicators = st.multiselect(
                    "选择技术指标",
                    options=["MACD", "KDJ", "RSI", "WR21"],
                    default=["MACD"],
                    key="tech_indicators"
                )
            
            # 创建K线图
            k_fig = plot_candlestick(stock_data)
            st.plotly_chart(k_fig, use_container_width=True)
            
            # 创建成交量图
            vol_fig = plot_volume(stock_data)
            st.plotly_chart(vol_fig, use_container_width=True)
            
            # 创建技术指标图
            if "MACD" in selected_indicators:
                macd_fig = go.Figure()
                macd_fig.add_trace(go.Scatter(x=stock_data['日期'], y=stock_data['DIF'], mode='lines', name='DIF'))
                macd_fig.add_trace(go.Scatter(x=stock_data['日期'], y=stock_data['DEA'], mode='lines', name='DEA'))
                
                # 添加MACD柱状图
                colors = ['red' if val >= 0 else 'green' for val in stock_data['MACD']]
                macd_fig.add_trace(go.Bar(x=stock_data['日期'], y=stock_data['MACD'], name='MACD', marker_color=colors))
                
                macd_fig.update_layout(title='MACD指标', height=200)
                st.plotly_chart(macd_fig, use_container_width=True)
                
            if "KDJ" in selected_indicators:
                kdj_fig = go.Figure()
                kdj_fig.add_trace(go.Scatter(x=stock_data['日期'], y=stock_data['K'], mode='lines', name='K'))
                kdj_fig.add_trace(go.Scatter(x=stock_data['日期'], y=stock_data['D'], mode='lines', name='D'))
                kdj_fig.add_trace(go.Scatter(x=stock_data['日期'], y=stock_data['J'], mode='lines', name='J'))
                
                kdj_fig.update_layout(title='KDJ指标', height=200)
                st.plotly_chart(kdj_fig, use_container_width=True)
                
            if "RSI" in selected_indicators:
                rsi_fig = go.Figure()
                rsi_fig.add_trace(go.Scatter(x=stock_data['日期'], y=stock_data['RSI'], mode='lines', name='RSI'))
                
                # 添加参考线
                rsi_fig.add_shape(type="line", x0=stock_data['日期'].iloc[0], y0=80, x1=stock_data['日期'].iloc[-1], y1=80,
                                  line=dict(color="red", width=1, dash="dash"))
                rsi_fig.add_shape(type="line", x0=stock_data['日期'].iloc[0], y0=20, x1=stock_data['日期'].iloc[-1], y1=20,
                                  line=dict(color="green", width=1, dash="dash"))
                
                rsi_fig.update_layout(title='RSI指标', height=200)
                st.plotly_chart(rsi_fig, use_container_width=True)
            
            if "WR21" in selected_indicators:
                wr_fig = go.Figure()
                wr_fig.add_trace(go.Scatter(x=stock_data['日期'], y=stock_data['WR21'], mode='lines', name='WR(21日均值)'))
                
                # 添加超买超卖参考线
                wr_fig.add_shape(type="line", x0=stock_data['日期'].iloc[0], y0=-20, x1=stock_data['日期'].iloc[-1], y1=-20,
                                line=dict(color="red", width=1, dash="dash"))
                wr_fig.add_shape(type="line", x0=stock_data['日期'].iloc[0], y0=-80, x1=stock_data['日期'].iloc[-1], y1=-80,
                                line=dict(color="green", width=1, dash="dash"))
                
                wr_fig.update_layout(title='威廉WR指标(21日均值)', height=200)
                st.plotly_chart(wr_fig, use_container_width=True)
            
            # 显示近期数据
            with st.expander("查看历史数据"):
                st.dataframe(
                    stock_data.sort_values('日期', ascending=False),
                    height=300
                )
        else:
            st.error(f"未能获取到 {selected_stock} 的数据，请尝试其他股票。")

if __name__ == "__main__":
    main()