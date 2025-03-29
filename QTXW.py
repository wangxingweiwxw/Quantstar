import streamlit as st
import akshare as ak
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

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
    
    # 计算威廉指标(WR)
    # 计算21日最高价和最低价
    high_21 = df['最高'].rolling(window=21).max()
    low_21 = df['最低'].rolling(window=21).min()
    # 计算WR
    df['WR21'] = (high_21 - df['收盘']) / (high_21 - low_21) * 100
    
    # 计算成交量比率
    df['MA5_Volume'] = df['成交量'].rolling(window=5).mean()
    df['MA10_Volume'] = df['成交量'].rolling(window=10).mean()
    df['Volume_Ratio'] = df['成交量'] / df['MA10_Volume']
    
    return df

# 市场趋势分析
def market_analysis(df):
    if df.empty:
        return {}
    
    latest_data = df.iloc[-1]

    # 判断当前趋势
    if latest_data['DIF'] > latest_data['DEA']:
        trend = "上升"
    elif latest_data['DIF'] < latest_data['DEA']:
        trend = "下降"
    else:
        trend = "盘整"

    # 判断市场强度
    if latest_data['RSI'] > 60:
        market_strength = "强"
    elif latest_data['RSI'] < 30:
        market_strength = "弱"
    else:
        market_strength = "中"

    # 判断爆发力
    if 'Volume_Ratio' in latest_data and not pd.isna(latest_data['Volume_Ratio']):
        if latest_data['Volume_Ratio'] > 1.5:
            explosive_power = "强"
        elif latest_data['Volume_Ratio'] > 1.0:
            explosive_power = "中"
        else:
            explosive_power = "弱"
    else:
        explosive_power = "弱"

    # 判断交易方向
    if trend == "上升" and market_strength != "弱":
        direction = "做多"
    elif trend == "下降" and market_strength != "强":
        direction = "做空"
    else:
        direction = "观望"

    return {
        "交易方向": direction,
        "当前趋势": trend,
        "市场强度": market_strength,
        "爆发力": explosive_power
    }

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
        name='K线',
        increasing_line_color='red',  # 上涨为红色
        decreasing_line_color='green',  # 下跌为绿色
        increasing_fillcolor='red',  # 上涨填充为红色
        decreasing_fillcolor='green'  # 下跌填充为绿色
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
        height=500,
        # 设置x轴，排除非交易日（周末和节假日）
        xaxis=dict(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # 排除周末
                dict(values=["2022-01-01", "2022-01-31", "2022-02-01", "2022-02-02", "2022-02-03", 
                            "2022-04-05", "2022-05-01", "2022-05-02", "2022-05-03", "2022-05-04", 
                            "2022-06-03", "2022-09-10", "2022-09-11", "2022-09-12", "2022-10-01", 
                            "2022-10-02", "2022-10-03", "2022-10-04", "2022-10-05", "2022-10-06", "2022-10-07",
                            "2023-01-01", "2023-01-02", "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24", "2023-01-25", "2023-01-26", "2023-01-27",
                            "2023-04-05", "2023-04-29", "2023-04-30", "2023-05-01", "2023-05-02", "2023-05-03",
                            "2023-06-22", "2023-06-23", "2023-09-29", "2023-09-30", "2023-10-01", "2023-10-02", "2023-10-03", "2023-10-04", "2023-10-05", "2023-10-06",
                            "2024-01-01", "2024-02-10", "2024-02-11", "2024-02-12", "2024-02-13", "2024-02-14", "2024-02-15", "2024-02-16", "2024-02-17",
                            "2024-04-04", "2024-04-05", "2024-04-06", "2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-05",
                            "2024-06-10", "2024-09-15", "2024-09-16", "2024-09-17", "2024-10-01", "2024-10-02", "2024-10-03", "2024-10-04", "2024-10-05", "2024-10-06", "2024-10-07"])  # 排除假日
            ]
        )
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
        height=200,
        # 设置x轴，排除非交易日（周末和节假日）
        xaxis=dict(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # 排除周末
                dict(values=["2022-01-01", "2022-01-31", "2022-02-01", "2022-02-02", "2022-02-03", 
                            "2022-04-05", "2022-05-01", "2022-05-02", "2022-05-03", "2022-05-04", 
                            "2022-06-03", "2022-09-10", "2022-09-11", "2022-09-12", "2022-10-01", 
                            "2022-10-02", "2022-10-03", "2022-10-04", "2022-10-05", "2022-10-06", "2022-10-07",
                            "2023-01-01", "2023-01-02", "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24", "2023-01-25", "2023-01-26", "2023-01-27",
                            "2023-04-05", "2023-04-29", "2023-04-30", "2023-05-01", "2023-05-02", "2023-05-03",
                            "2023-06-22", "2023-06-23", "2023-09-29", "2023-09-30", "2023-10-01", "2023-10-02", "2023-10-03", "2023-10-04", "2023-10-05", "2023-10-06",
                            "2024-01-01", "2024-02-10", "2024-02-11", "2024-02-12", "2024-02-13", "2024-02-14", "2024-02-15", "2024-02-16", "2024-02-17",
                            "2024-04-04", "2024-04-05", "2024-04-06", "2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-05",
                            "2024-06-10", "2024-09-15", "2024-09-16", "2024-09-17", "2024-10-01", "2024-10-02", "2024-10-03", "2024-10-04", "2024-10-05", "2024-10-06", "2024-10-07"])  # 排除假日
            ]
        )
    )
    
    return fig

# 准备LSTM模型的训练数据
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# 预测股票价格函数
@st.cache_resource
def predict_stock_price(stock_data, predict_days=30):
    """
    使用LSTM模型预测股票价格
    
    参数:
        stock_data: 包含历史股价的DataFrame
        predict_days: 预测未来的天数，默认30天
        
    返回:
        预测结果和相关数据的字典
    """
    try:
        # 使用收盘价进行训练和预测
        closing_prices = stock_data['收盘'].values.reshape(-1, 1)
        
        # 数据归一化
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(closing_prices)
        
        # 准备训练数据
        time_step = 60  # 使用过去60天的数据预测下一天
        X, y = create_dataset(scaled_data, time_step)
        
        # 重塑输入为LSTM需要的格式 [samples, time steps, features]
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # 训练测试集分割 (80% 训练, 20% 测试)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # 创建并编译LSTM模型
        with st.spinner('正在训练LSTM模型...'):
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # 训练模型
            model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                    epochs=20, batch_size=32, verbose=0)
        
        # 准备预测数据（使用最近的time_step天数据）
        inputs = scaled_data[-time_step:]
        inputs = inputs.reshape(1, time_step, 1)
        
        # 预测未来predict_days天
        future_predictions = []
        current_batch = inputs
        
        for _ in range(predict_days):
            # 预测下一天
            current_pred = model.predict(current_batch, verbose=0)[0]
            future_predictions.append(current_pred[0])
            
            # 更新当前批次（移除最早的值，添加新预测的值）
            current_batch = np.append(current_batch[:, 1:, :], 
                                     [[current_pred]], 
                                     axis=1)
        
        # 将预测结果转换回原始比例
        future_predictions = np.array(future_predictions).reshape(-1, 1)
        predicted_prices = scaler.inverse_transform(future_predictions)
        
        # 生成未来日期（确保不重复）
        last_date = pd.to_datetime(stock_data['日期'].iloc[-1])
        future_dates = []
        current_date = last_date
        
        while len(future_dates) < predict_days:
            current_date += pd.Timedelta(days=1)
            if current_date.weekday() < 5:  # 只添加工作日（周一到周五）
                future_dates.append(current_date)
        
        # 确保日期和预测价格一一对应
        future_dates = future_dates[:len(predicted_prices)]
        predicted_prices = predicted_prices[:len(future_dates)]
        
        # 返回结果
        return {
            'dates': future_dates,
            'predicted_prices': predicted_prices.flatten(),
            'model': model,
            'last_price': closing_prices[-1][0]
        }
    except Exception as e:
        st.error(f"预测过程中出错: {e}")
        return None

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
            # 添加市场趋势分析
            market_status = market_analysis(stock_data)
            
            # 创建市场趋势显示行
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                direction_color = {
                    "做多": "#2E8B57",  # dark green
                    "做空": "#B22222",  # dark red
                    "观望": "#696969"   # dark grey
                }.get(market_status.get("交易方向", "观望"), "#696969")
                
                st.markdown(f"""
                <div style="background-color: {direction_color}; padding: 8px; border-radius: 5px; text-align: center; height: 80px; display: flex; flex-direction: column; justify-content: center; box-shadow: 0 1px 3px rgba(0,0,0,0.3);">
                    <p style="margin: 0; font-size: 0.9rem; font-weight: bold; color: white;">交易方向</p>
                    <p style="margin: 0; font-size: 1.2rem; color: white;">{market_status.get("交易方向", "观望")}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                trend_color = {
                    "上升": "#2E8B57",  # dark green
                    "下降": "#B22222",  # dark red
                    "盘整": "#DAA520"   # dark yellow
                }.get(market_status.get("当前趋势", "盘整"), "#DAA520")
                
                st.markdown(f"""
                <div style="background-color: {trend_color}; padding: 8px; border-radius: 5px; text-align: center; height: 80px; display: flex; flex-direction: column; justify-content: center; box-shadow: 0 1px 3px rgba(0,0,0,0.3);">
                    <p style="margin: 0; font-size: 0.9rem; font-weight: bold; color: white;">当前趋势</p>
                    <p style="margin: 0; font-size: 1.2rem; color: white;">{market_status.get("当前趋势", "盘整")}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                strength_color = {
                    "强": "#2E8B57",     # dark green
                    "中": "#DAA520",     # dark yellow
                    "弱": "#B22222"      # dark red
                }.get(market_status.get("市场强度", "中"), "#DAA520")
                
                st.markdown(f"""
                <div style="background-color: {strength_color}; padding: 8px; border-radius: 5px; text-align: center; height: 80px; display: flex; flex-direction: column; justify-content: center; box-shadow: 0 1px 3px rgba(0,0,0,0.3);">
                    <p style="margin: 0; font-size: 0.9rem; font-weight: bold; color: white;">市场强度</p>
                    <p style="margin: 0; font-size: 1.2rem; color: white;">{market_status.get("市场强度", "中")}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                power_color = {
                    "强": "#2E8B57",    # dark green
                    "中": "#DAA520",    # dark yellow
                    "弱": "#B22222"     # dark red
                }.get(market_status.get("爆发力", "弱"), "#B22222")
                
                st.markdown(f"""
                <div style="background-color: {power_color}; padding: 8px; border-radius: 5px; text-align: center; height: 80px; display: flex; flex-direction: column; justify-content: center; box-shadow: 0 1px 3px rgba(0,0,0,0.3);">
                    <p style="margin: 0; font-size: 0.9rem; font-weight: bold; color: white;">爆发力</p>
                    <p style="margin: 0; font-size: 1.2rem; color: white;">{market_status.get("爆发力", "弱")}</p>
                </div>
                """, unsafe_allow_html=True)
            
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
                
                macd_fig.update_layout(
                    title='MACD指标', 
                    height=500,
                    xaxis=dict(
                        rangebreaks=[
                            dict(bounds=["sat", "mon"]),  # 排除周末
                            dict(values=["2022-01-01", "2022-01-31", "2022-02-01", "2022-02-02", "2022-02-03", 
                                        "2022-04-05", "2022-05-01", "2022-05-02", "2022-05-03", "2022-05-04", 
                                        "2022-06-03", "2022-09-10", "2022-09-11", "2022-09-12", "2022-10-01", 
                                        "2022-10-02", "2022-10-03", "2022-10-04", "2022-10-05", "2022-10-06", "2022-10-07",
                                        "2023-01-01", "2023-01-02", "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24", "2023-01-25", "2023-01-26", "2023-01-27",
                                        "2023-04-05", "2023-04-29", "2023-04-30", "2023-05-01", "2023-05-02", "2023-05-03",
                                        "2023-06-22", "2023-06-23", "2023-09-29", "2023-09-30", "2023-10-01", "2023-10-02", "2023-10-03", "2023-10-04", "2023-10-05", "2023-10-06",
                                        "2024-01-01", "2024-02-10", "2024-02-11", "2024-02-12", "2024-02-13", "2024-02-14", "2024-02-15", "2024-02-16", "2024-02-17",
                                        "2024-04-04", "2024-04-05", "2024-04-06", "2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-05",
                                        "2024-06-10", "2024-09-15", "2024-09-16", "2024-09-17", "2024-10-01", "2024-10-02", "2024-10-03", "2024-10-04", "2024-10-05", "2024-10-06", "2024-10-07"])  # 排除假日
                        ]
                    )
                )
                st.plotly_chart(macd_fig, use_container_width=True)
                
            if "KDJ" in selected_indicators:
                kdj_fig = go.Figure()
                kdj_fig.add_trace(go.Scatter(x=stock_data['日期'], y=stock_data['K'], mode='lines', name='K'))
                kdj_fig.add_trace(go.Scatter(x=stock_data['日期'], y=stock_data['D'], mode='lines', name='D'))
                kdj_fig.add_trace(go.Scatter(x=stock_data['日期'], y=stock_data['J'], mode='lines', name='J'))
                
                kdj_fig.update_layout(
                    title='KDJ指标', 
                    height=500,
                    xaxis=dict(
                        rangebreaks=[
                            dict(bounds=["sat", "mon"]),  # 排除周末
                            dict(values=["2022-01-01", "2022-01-31", "2022-02-01", "2022-02-02", "2022-02-03", 
                                        "2022-04-05", "2022-05-01", "2022-05-02", "2022-05-03", "2022-05-04", 
                                        "2022-06-03", "2022-09-10", "2022-09-11", "2022-09-12", "2022-10-01", 
                                        "2022-10-02", "2022-10-03", "2022-10-04", "2022-10-05", "2022-10-06", "2022-10-07",
                                        "2023-01-01", "2023-01-02", "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24", "2023-01-25", "2023-01-26", "2023-01-27",
                                        "2023-04-05", "2023-04-29", "2023-04-30", "2023-05-01", "2023-05-02", "2023-05-03",
                                        "2023-06-22", "2023-06-23", "2023-09-29", "2023-09-30", "2023-10-01", "2023-10-02", "2023-10-03", "2023-10-04", "2023-10-05", "2023-10-06",
                                        "2024-01-01", "2024-02-10", "2024-02-11", "2024-02-12", "2024-02-13", "2024-02-14", "2024-02-15", "2024-02-16", "2024-02-17",
                                        "2024-04-04", "2024-04-05", "2024-04-06", "2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-05",
                                        "2024-06-10", "2024-09-15", "2024-09-16", "2024-09-17", "2024-10-01", "2024-10-02", "2024-10-03", "2024-10-04", "2024-10-05", "2024-10-06", "2024-10-07"])  # 排除假日
                        ]
                    )
                )
                st.plotly_chart(kdj_fig, use_container_width=True)
                
            if "RSI" in selected_indicators:
                rsi_fig = go.Figure()
                rsi_fig.add_trace(go.Scatter(x=stock_data['日期'], y=stock_data['RSI'], mode='lines', name='RSI'))
                
                # 添加参考线
                rsi_fig.add_shape(type="line", x0=stock_data['日期'].iloc[0], y0=80, x1=stock_data['日期'].iloc[-1], y1=80,
                                  line=dict(color="red", width=1, dash="dash"))
                rsi_fig.add_shape(type="line", x0=stock_data['日期'].iloc[0], y0=20, x1=stock_data['日期'].iloc[-1], y1=20,
                                  line=dict(color="green", width=1, dash="dash"))
                
                rsi_fig.update_layout(
                    title='RSI指标', 
                    height=500,
                    xaxis=dict(
                        rangebreaks=[
                            dict(bounds=["sat", "mon"]),  # 排除周末
                            dict(values=["2022-01-01", "2022-01-31", "2022-02-01", "2022-02-02", "2022-02-03", 
                                        "2022-04-05", "2022-05-01", "2022-05-02", "2022-05-03", "2022-05-04", 
                                        "2022-06-03", "2022-09-10", "2022-09-11", "2022-09-12", "2022-10-01", 
                                        "2022-10-02", "2022-10-03", "2022-10-04", "2022-10-05", "2022-10-06", "2022-10-07",
                                        "2023-01-01", "2023-01-02", "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24", "2023-01-25", "2023-01-26", "2023-01-27",
                                        "2023-04-05", "2023-04-29", "2023-04-30", "2023-05-01", "2023-05-02", "2023-05-03",
                                        "2023-06-22", "2023-06-23", "2023-09-29", "2023-09-30", "2023-10-01", "2023-10-02", "2023-10-03", "2023-10-04", "2023-10-05", "2023-10-06",
                                        "2024-01-01", "2024-02-10", "2024-02-11", "2024-02-12", "2024-02-13", "2024-02-14", "2024-02-15", "2024-02-16", "2024-02-17",
                                        "2024-04-04", "2024-04-05", "2024-04-06", "2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-05",
                                        "2024-06-10", "2024-09-15", "2024-09-16", "2024-09-17", "2024-10-01", "2024-10-02", "2024-10-03", "2024-10-04", "2024-10-05", "2024-10-06", "2024-10-07"])  # 排除假日
                        ]
                    )
                )
                st.plotly_chart(rsi_fig, use_container_width=True)
            
            # 在技术指标图表部分添加WR21图表
            if "WR21" in selected_indicators:
                wr_fig = go.Figure()
                wr_fig.add_trace(go.Scatter(x=stock_data['日期'], y=stock_data['WR21'], mode='lines', name='WR21'))
                
                # 添加参考线
                wr_fig.add_shape(type="line", x0=stock_data['日期'].iloc[0], y0=20, x1=stock_data['日期'].iloc[-1], y1=20,
                                  line=dict(color="red", width=1, dash="dash"))
                wr_fig.add_shape(type="line", x0=stock_data['日期'].iloc[0], y0=80, x1=stock_data['日期'].iloc[-1], y1=80,
                                  line=dict(color="green", width=1, dash="dash"))
                
                wr_fig.update_layout(
                    title='威廉指标(WR21)', 
                    height=500,
                    xaxis=dict(
                        rangebreaks=[
                            dict(bounds=["sat", "mon"]),  # 排除周末
                            dict(values=["2022-01-01", "2022-01-31", "2022-02-01", "2022-02-02", "2022-02-03", 
                                        "2022-04-05", "2022-05-01", "2022-05-02", "2022-05-03", "2022-05-04", 
                                        "2022-06-03", "2022-09-10", "2022-09-11", "2022-09-12", "2022-10-01", 
                                        "2022-10-02", "2022-10-03", "2022-10-04", "2022-10-05", "2022-10-06", "2022-10-07",
                                        "2023-01-01", "2023-01-02", "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24", "2023-01-25", "2023-01-26", "2023-01-27",
                                        "2023-04-05", "2023-04-29", "2023-04-30", "2023-05-01", "2023-05-02", "2023-05-03",
                                        "2023-06-22", "2023-06-23", "2023-09-29", "2023-09-30", "2023-10-01", "2023-10-02", "2023-10-03", "2023-10-04", "2023-10-05", "2023-10-06",
                                        "2024-01-01", "2024-02-10", "2024-02-11", "2024-02-12", "2024-02-13", "2024-02-14", "2024-02-15", "2024-02-16", "2024-02-17",
                                        "2024-04-04", "2024-04-05", "2024-04-06", "2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-05",
                                        "2024-06-10", "2024-09-15", "2024-09-16", "2024-09-17", "2024-10-01", "2024-10-02", "2024-10-03", "2024-10-04", "2024-10-05", "2024-10-06", "2024-10-07"])  # 排除假日
                        ]
                    )
                )
                st.plotly_chart(wr_fig, use_container_width=True)
            
            # 添加股价预测
            st.subheader("📈 股价预测 (LSTM模型)")
            
            # 添加交易建议区域
            st.markdown("""
            <style>
            .recommendation-box {
                background-color: #3A3A3A;
                border-radius: 10px;
                padding: 15px;
                margin-top: 10px;
                margin-bottom: 20px;
                color: white;
            }
            .recommendation-box h4 {
                color: white !important;
                margin-top: 0;
                margin-bottom: 10px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="recommendation-box">
                <h4>交易建议</h4>
                <p>根据当前市场分析，建议采取<b>{market_status.get("交易方向", "观望")}</b>策略。
                市场处于<b>{market_status.get("当前趋势", "盘整")}</b>趋势，市场强度<b>{market_status.get("市场强度", "中")}</b>，
                成交量爆发力<b>{market_status.get("爆发力", "弱")}</b>。</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 预测天数选择
            predict_days = st.slider(
                "选择预测天数", 
                min_value=3, 
                max_value=14, 
                value=7, 
                step=1,
                help="选择3至14天的预测期限"
            )
            
            # 添加预测按钮
            if st.button("生成股价预测"):
                # 确保有足够的历史数据进行预测
                if len(stock_data) >= 60:  # 至少需要60天的数据
                    with st.spinner(f"正在预测未来{predict_days}天的股价...这可能需要一些时间..."):
                        prediction_results = predict_stock_price(stock_data, predict_days=predict_days)
                    
                    if prediction_results:
                        # 创建预测图表
                        fig = go.Figure()
                        
                        # 添加历史价格
                        fig.add_trace(go.Scatter(
                            x=stock_data['日期'][-30:],  # 显示最近30天历史数据
                            y=stock_data['收盘'][-30:],
                            mode='lines',
                            name='历史价格',
                            line=dict(color='blue')
                        ))
                        
                        # 添加预测价格
                        fig.add_trace(go.Scatter(
                            x=prediction_results['dates'],
                            y=prediction_results['predicted_prices'],
                            mode='lines',
                            name='预测价格',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        # 设置图表布局
                        fig.update_layout(
                            title='股价预测 (LSTM模型)',
                            xaxis_title='日期',
                            yaxis_title='价格',
                            height=400,
                            xaxis=dict(
                                rangebreaks=[
                                    dict(bounds=["sat", "mon"]),  # 排除周末
                                ]
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 显示预测指标
                        pred_df = pd.DataFrame({
                            '日期': prediction_results['dates'],
                            '预测价格': np.round(prediction_results['predicted_prices'], 2)
                        })
                        
                        # 计算预测期末的预期涨跌幅
                        last_price = stock_data['收盘'].iloc[-1]
                        last_pred_price = prediction_results['predicted_prices'][-1]
                        expected_change = (last_pred_price - last_price) / last_price * 100
                        
                        # 显示预期涨跌幅
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            # 根据涨跌幅确定背景颜色
                            price_color = "#2E8B57" if expected_change >= 0 else "#B22222"  # 涨-深绿色，跌-深红色
                            
                            st.markdown(f"""
                            <div style="background-color: {price_color}; padding: 8px; border-radius: 5px; text-align: center; height: 80px; display: flex; flex-direction: column; justify-content: center; box-shadow: 0 1px 3px rgba(0,0,0,0.3);">
                                <p style="margin: 0; font-size: 0.9rem; font-weight: bold; color: white;">{predict_days}天后预期价格</p>
                                <p style="margin: 0; font-size: 1.2rem; color: white;">¥{last_pred_price:.2f} ({expected_change:+.2f}%)</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.info("""
                            **模型说明**：预测基于LSTM深度学习模型，仅供参考。预测结果应结合上方的市场趋势分析，综合判断交易方向。
                            股市受多种因素影响，模型无法预测突发事件和政策变化。
                            """)
                        
                        # 显示预测数据表格
                        with st.expander("查看预测数据明细"):
                            st.dataframe(pred_df)
                else:
                    st.warning("历史数据不足，需要更多的交易日数据才能生成可靠预测")
            
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