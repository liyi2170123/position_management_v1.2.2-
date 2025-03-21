import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加时区处理函数
def ensure_naive_timestamp(ts):
    """确保时间戳没有时区信息"""
    if isinstance(ts, pd.Timestamp):
        if ts.tz is not None:
            return ts.tz_localize(None)
        return ts
    return ts

st.set_page_config(page_title="资金曲线分析工具", layout="wide")

st.title("资金曲线分析工具")

# 上传文件功能
uploaded_file = st.file_uploader("上传资金曲线CSV文件", type=["csv"])

if uploaded_file is not None:
    # 读取资金曲线数据
    try:
        equity_data = pd.read_csv(uploaded_file)
        
        # 记录数据信息
        logger.info(f"文件读取成功，原始数据形状: {equity_data.shape}")
        
        # 检查是否包含必要的列
        required_columns = ['candle_begin_time', '净值']
        if not all(col in equity_data.columns for col in required_columns):
            st.error("上传的CSV文件格式不正确，请确保包含 'candle_begin_time' 和 '净值' 列")
        else:
            # 转换时间列为datetime格式
            logger.info(f"转换前时间列样例: {equity_data['candle_begin_time'].iloc[0]}")
            equity_data['candle_begin_time'] = pd.to_datetime(equity_data['candle_begin_time'])
            logger.info(f"转换后时间列样例及时区: {equity_data['candle_begin_time'].iloc[0]}, 时区: {equity_data['candle_begin_time'].dt.tz}")
            
            if equity_data['candle_begin_time'].dt.tz is not None:
                equity_data['candle_begin_time'] = equity_data['candle_begin_time'].dt.tz_localize(None)
                logger.info(f"移除时区后时间列样例: {equity_data['candle_begin_time'].iloc[0]}, 时区: {equity_data['candle_begin_time'].dt.tz}")
            
            # 显示数据基本信息
            st.subheader("数据基本信息")
            st.write(f"数据时间范围: {equity_data['candle_begin_time'].min().strftime('%Y-%m-%d %H:%M:%S')} 至 {equity_data['candle_begin_time'].max().strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"数据点数量: {len(equity_data)}")
            
            # 时间范围选择
            st.subheader("时间范围选择")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "开始日期",
                    min(equity_data['candle_begin_time']).date(),
                    min_value=min(equity_data['candle_begin_time']).date(),
                    max_value=max(equity_data['candle_begin_time']).date()
                )
            with col2:
                end_date = st.date_input(
                    "结束日期",
                    max(equity_data['candle_begin_time']).date(),
                    min_value=min(equity_data['candle_begin_time']).date(),
                    max_value=max(equity_data['candle_begin_time']).date()
                )
            
            # 回撤范围选择
            st.subheader("回撤范围选择")
            col1, col2 = st.columns(2)
            with col1:
                min_drawdown = st.slider(
                    "最小回撤百分比 (%)",
                    min_value=0.0,
                    max_value=50.0,
                    value=5.0,
                    step=0.5
                )
            with col2:
                max_drawdown = st.slider(
                    "最大回撤百分比 (%)",
                    min_value=0.0,
                    max_value=50.0,
                    value=20.0,
                    step=0.5
                )
            
            # 回撤天数范围选择
            st.subheader("回撤持续天数范围选择")
            col1, col2 = st.columns(2)
            with col1:
                min_drawdown_days = st.slider(
                    "最小回撤持续天数",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=0.1
                )
            with col2:
                max_drawdown_days = st.slider(
                    "最大回撤持续天数",
                    min_value=0.0,
                    max_value=100.0,
                    value=30.0,
                    step=0.1
                )
            
            # 新增：新高间隔天数范围选择
            st.subheader("新高间隔天数范围选择")
            col1, col2 = st.columns(2)
            with col1:
                min_high_interval_days = st.slider(
                    "最小新高间隔天数",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=0.1
                )
            with col2:
                max_high_interval_days = st.slider(
                    "最大新高间隔天数",
                    min_value=0.0,
                    max_value=100.0,
                    value=50.0,
                    step=0.1
                )
            
            # 过滤数据
            start_datetime = pd.Timestamp(start_date)
            end_datetime = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            # 确保时区一致
            start_datetime = ensure_naive_timestamp(start_datetime)
            end_datetime = ensure_naive_timestamp(end_datetime)
            filtered_data = equity_data[(equity_data['candle_begin_time'] >= start_datetime) & 
                                       (equity_data['candle_begin_time'] <= end_datetime)].copy()
            
            if len(filtered_data) == 0:
                st.warning("所选时间范围内没有数据")
            else:
                # 重新计算净值，从1开始
                initial_value = filtered_data['净值'].iloc[0]
                filtered_data['重置净值'] = filtered_data['净值'] / initial_value
                
                # 计算关键指标
                initial_equity = 1.0  # 重置后的初始净值
                final_equity = filtered_data['重置净值'].iloc[-1]
                
                # 计算年化收益率
                days = (filtered_data['candle_begin_time'].iloc[-1] - filtered_data['candle_begin_time'].iloc[0]).days
                if days > 0:
                    annual_return = (final_equity / initial_equity) ** (365 / days) - 1
                else:
                    annual_return = 0
                
                # 计算最大回撤
                filtered_data['净值_cummax'] = filtered_data['重置净值'].cummax()
                filtered_data['drawdown'] = (filtered_data['重置净值'] - filtered_data['净值_cummax']) / filtered_data['净值_cummax']
                max_dd = filtered_data['drawdown'].min()
                
                # 找到最大回撤的开始和结束时间
                max_dd_end_idx = filtered_data['drawdown'].idxmin()
                max_dd_end_time = filtered_data.loc[max_dd_end_idx, 'candle_begin_time']
                
                # 找到最大回撤开始时间（净值达到高点的时间）
                temp_df = filtered_data.loc[:max_dd_end_idx]
                max_dd_start_idx = temp_df['重置净值'].idxmax()
                max_dd_start_time = filtered_data.loc[max_dd_start_idx, 'candle_begin_time']
                
                # 计算回撤持续天数
                max_dd_duration = (max_dd_end_time - max_dd_start_time).total_seconds() / (24 * 3600)
                
                # 计算最长不创新高时间
                filtered_data['is_new_high'] = filtered_data['重置净值'] >= filtered_data['重置净值'].cummax()
                filtered_data['high_group'] = (filtered_data['is_new_high'] != filtered_data['is_new_high'].shift()).cumsum()
                
                # 找出每个非新高组的开始和结束时间
                no_new_high_periods = []
                for group in filtered_data[~filtered_data['is_new_high']]['high_group'].unique():
                    group_data = filtered_data[filtered_data['high_group'] == group]
                    if not group_data['is_new_high'].any():  # 确保这是一个非新高组
                        start_time = group_data['candle_begin_time'].min()
                        end_time = group_data['candle_begin_time'].max()
                        duration = (end_time - start_time).total_seconds() / (24 * 3600)
                        no_new_high_periods.append((start_time, end_time, duration))
                
                # 找出最长的不创新高期间
                if no_new_high_periods:
                    longest_period = max(no_new_high_periods, key=lambda x: x[2])
                    longest_no_new_high_start = longest_period[0]
                    longest_no_new_high_end = longest_period[1]
                    longest_no_new_high_duration = longest_period[2]
                else:
                    longest_no_new_high_start = pd.NaT
                    longest_no_new_high_end = pd.NaT
                    longest_no_new_high_duration = 0
                
                # 找出最新一次新高时间
                if filtered_data['is_new_high'].any():
                    latest_high_idx = filtered_data[filtered_data['is_new_high']].index[-1]
                    latest_high_time = filtered_data.loc[latest_high_idx, 'candle_begin_time']
                else:
                    latest_high_time = pd.NaT
                
                # 计算距离前高天数
                if not pd.isna(latest_high_time):
                    days_since_high = (filtered_data['candle_begin_time'].iloc[-1] - latest_high_time).total_seconds() / (24 * 3600)
                else:
                    days_since_high = np.nan
                
                # 计算当前回撤百分比
                current_drawdown = filtered_data['drawdown'].iloc[-1] * 100
                
                # 计算收益率标准差（日收益率）
                filtered_data['daily_return'] = filtered_data['重置净值'].pct_change()
                std_dev = filtered_data['daily_return'].std() * (252 ** 0.5)  # 年化标准差
                
                # 计算指定回撤范围内的回撤次数
                # 初始化回撤事件列表
                drawdown_events = []
                
                # 找出所有新高点
                high_points = []
                for idx, row in filtered_data[filtered_data['is_new_high']].iterrows():
                    # 只记录每组新高的最后一个点（即该组的最高点）
                    group = row['high_group']
                    group_data = filtered_data[filtered_data['high_group'] == group]
                    if idx == group_data.index[-1]:  # 如果是该组的最后一个点
                        high_points.append({
                            'time': row['candle_begin_time'],
                            'value': row['重置净值'],
                            'index': idx
                        })
                
                # 检测回撤事件（从一个新高到下一个新高之间的最大回撤）
                if len(high_points) >= 2:
                    for i in range(len(high_points) - 1):
                        current_high = high_points[i]
                        next_high = high_points[i + 1]
                        
                        # 获取两个新高之间的数据
                        period_data = filtered_data.loc[current_high['index']:next_high['index']]
                        
                        if len(period_data) > 1:  # 确保有足够的数据点
                            # 找到期间最低点
                            min_idx = period_data['重置净值'].idxmin()
                            min_value = period_data.loc[min_idx, '重置净值']
                            min_time = period_data.loc[min_idx, 'candle_begin_time']
                            
                            # 计算最大回撤
                            max_drawdown_pct = (min_value - current_high['value']) / current_high['value'] * 100
                            
                            # 计算新高间隔天数
                            high_interval = (next_high['time'] - current_high['time']).total_seconds() / (24 * 3600)
                            
                            # 计算从回撤最低点到下一个新高的天数
                            recovery_duration = (next_high['time'] - min_time).total_seconds() / (24 * 3600)
                            
                            # 计算从回撤开始到最低点的天数（真正的回撤持续天数）
                            drawdown_duration = (min_time - current_high['time']).total_seconds() / (24 * 3600)
                            
                            # 检查是否满足筛选条件 - 修改为同时检查回撤幅度、回撤持续天数和新高间隔天数
                            if (abs(max_drawdown_pct) >= min_drawdown and
                                abs(max_drawdown_pct) <= max_drawdown and
                                drawdown_duration >= min_drawdown_days and  # 使用真正的回撤持续天数
                                drawdown_duration <= max_drawdown_days and
                                high_interval >= min_high_interval_days and  # 新增：检查新高间隔天数
                                high_interval <= max_high_interval_days):
                                
                                drawdown_events.append({
                                    'dd_start_time': current_high['time'],  # 回撤开始时间（当前新高）
                                    'dd_lowest_time': min_time,  # 回撤最低点时间
                                    'dd_end_time': next_high['time'],  # 回撤结束时间（下一个新高）
                                    'max_drawdown': max_drawdown_pct,  # 最大回撤百分比
                                    'drawdown_duration': drawdown_duration,  # 真正的回撤持续天数（从高点到最低点）
                                    'recovery_duration': recovery_duration,  # 从最低点到新高的恢复天数
                                    'high_interval': high_interval,  # 新高之间的间隔天数
                                    'is_last_point': False  # 标记是否为最后一个点
                                })
                
                # 添加最后一个高点到最新数据点的回撤（如果存在）
                if len(high_points) > 0:
                    last_high = high_points[-1]
                    # 获取最后一个高点之后的数据
                    last_period_data = filtered_data.loc[last_high['index']:]
                    
                    if len(last_period_data) > 1:  # 确保有足够的数据点
                        # 计算当前回撤
                        current_value = last_period_data['重置净值'].iloc[-1]
                        current_time = last_period_data['candle_begin_time'].iloc[-1]
                        
                        # 找到期间最低点
                        min_idx = last_period_data['重置净值'].idxmin()
                        min_value = last_period_data.loc[min_idx, '重置净值']
                        min_time = last_period_data.loc[min_idx, 'candle_begin_time']
                        
                        # 计算最大回撤
                        max_drawdown_pct = (min_value - last_high['value']) / last_high['value'] * 100
                        
                        # 计算回撤持续天数（从最后高点到最低点）
                        drawdown_duration = (min_time - last_high['time']).total_seconds() / (24 * 3600)
                        
                        # 计算从最低点到当前的天数
                        recovery_duration = (current_time - min_time).total_seconds() / (24 * 3600)
                        
                        # 计算从最后高点到当前的天数
                        high_interval = (current_time - last_high['time']).total_seconds() / (24 * 3600)
                        
                        # 如果当前不是新高且回撤满足条件
                        if (current_value < last_high['value'] and 
                            abs(max_drawdown_pct) >= min_drawdown and
                            abs(max_drawdown_pct) <= max_drawdown and
                            drawdown_duration >= min_drawdown_days and
                            drawdown_duration <= max_drawdown_days and
                            high_interval >= min_high_interval_days and
                            high_interval <= max_high_interval_days):
                            
                            drawdown_events.append({
                                'dd_start_time': last_high['time'],  # 回撤开始时间（最后高点）
                                'dd_lowest_time': min_time,  # 回撤最低点时间
                                'dd_end_time': current_time,  # 当前时间作为临时结束时间
                                'max_drawdown': max_drawdown_pct,  # 最大回撤百分比
                                'drawdown_duration': drawdown_duration,  # 回撤持续天数
                                'recovery_duration': recovery_duration,  # 从最低点到当前的天数
                                'high_interval': high_interval,  # 从最后高点到当前的天数
                                'is_last_point': True  # 标记为最后一个点
                            })
                
                # 计算符合条件的回撤次数
                drawdown_count = len(drawdown_events)
                
                # 按回撤天数分组统计
                if drawdown_events:
                    duration_bins = [0, 0.1, 0.5, 1, 3, 7, 14, 30, 60, 100, float('inf')]
                    duration_labels = ['<0.1天', '0.1-0.5天', '0.5-1天', '1-3天', '3-7天', '7-14天', '14-30天', '30-60天', '60-100天', '>100天']
                    
                    # 创建回撤持续时间的分布 - 使用drawdown_duration字段
                    durations = [event['drawdown_duration'] for event in drawdown_events]
                    duration_counts = pd.cut(durations, bins=duration_bins, labels=duration_labels).value_counts().sort_index()
                    
                    # 创建回撤幅度的分布
                    drawdown_bins = [-100, -20, -15, -10, -5, -3, -1, -0.5, 0]
                    drawdown_labels = ['>20%', '15-20%', '10-15%', '5-10%', '3-5%', '1-3%', '0.5-1%', '<0.5%']
                    drawdowns = [event['max_drawdown'] for event in drawdown_events]
                    drawdown_counts = pd.cut(drawdowns, bins=drawdown_bins, labels=drawdown_labels).value_counts().sort_index()
                
                # 显示资金曲线图
                st.subheader("资金曲线")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=filtered_data['candle_begin_time'],
                    y=filtered_data['重置净值'],
                    mode='lines',
                    name='净值'
                ))
                
                # 添加最大回撤区域
                fig.add_shape(
                    type="rect",
                    x0=max_dd_start_time,
                    y0=filtered_data.loc[max_dd_start_idx, '重置净值'] * 1.05,
                    x1=max_dd_end_time,
                    y1=filtered_data.loc[max_dd_end_idx, '重置净值'] * 0.95,
                    line=dict(color="Red", width=1, dash="dash"),
                    fillcolor="rgba(255, 0, 0, 0.1)",
                )
                
                # 标记所有符合条件的回撤事件
                for event in drawdown_events:
                    fig.add_shape(
                        type="rect",
                        x0=event['dd_start_time'],
                        x1=event['dd_end_time'],
                        y0=0.95,  # 相对位置
                        y1=1.05,
                        line=dict(color="Orange", width=1, dash="dot"),
                        fillcolor="rgba(255, 165, 0, 0.1)",
                        layer="below"
                    )
                
                fig.update_layout(
                    title="资金曲线图",
                    xaxis_title="日期",
                    yaxis_title="净值",
                    height=600,
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 显示回撤图
                st.subheader("回撤曲线")
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=filtered_data['candle_begin_time'],
                    y=filtered_data['drawdown'] * 100,
                    mode='lines',
                    name='回撤百分比',
                    line=dict(color='red')
                ))
                
                # 添加回撤范围区域
                fig_dd.add_shape(
                    type="rect",
                    x0=filtered_data['candle_begin_time'].min(),
                    x1=filtered_data['candle_begin_time'].max(),
                    y0=-max_drawdown,
                    y1=-min_drawdown,
                    line=dict(color="Blue", width=1, dash="dash"),
                    fillcolor="rgba(0, 0, 255, 0.1)",
                )
                
                fig_dd.update_layout(
                    title="回撤曲线图",
                    xaxis_title="日期",
                    yaxis_title="回撤百分比 (%)",
                    height=400,
                    hovermode="x unified"
                )
                st.plotly_chart(fig_dd, use_container_width=True)
                
                # 显示关键指标
                st.subheader("关键指标")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("累积净值", f"{final_equity:.4f}", f"{(final_equity/initial_equity - 1)*100:.2f}%")
                    st.metric("年化收益", f"{annual_return*100:.2f}%")
                    st.metric("最大回撤", f"{max_dd*100:.2f}%")
                    st.metric("最大回撤持续天数", f"{max_dd_duration:.2f}")
                    st.metric("当前回撤百分比", f"{current_drawdown:.2f}%")
                    st.metric(f"回撤范围 ({min_drawdown}% - {max_drawdown}%) 内的回撤次数", f"{drawdown_count}")
                    st.metric(f"回撤天数范围 ({min_drawdown_days} - {max_drawdown_days}天) 内的回撤次数", f"{drawdown_count}")
                    st.metric(f"新高间隔天数范围 ({min_high_interval_days} - {max_high_interval_days}天) 内的回撤次数", f"{drawdown_count}")
                
                with col2:
                    # 将Timestamp转换为字符串
                    max_dd_start_time_str = max_dd_start_time.strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(max_dd_start_time) else "N/A"
                    max_dd_end_time_str = max_dd_end_time.strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(max_dd_end_time) else "N/A"
                    longest_no_new_high_start_str = longest_no_new_high_start.strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(longest_no_new_high_start) else "N/A"
                    longest_no_new_high_end_str = longest_no_new_high_end.strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(longest_no_new_high_end) else "N/A"
                    
                    st.metric("最大回撤开始时间", max_dd_start_time_str)
                    st.metric("最大回撤结束时间", max_dd_end_time_str)
                    st.metric("最长不创新高开始时间", longest_no_new_high_start_str)
                    st.metric("最长不创新高结束时间", longest_no_new_high_end_str)
                    st.metric("不创新高持续天数", f"{longest_no_new_high_duration:.2f}")
                
                with col3:
                    # 将Timestamp转换为字符串
                    latest_high_time_str = latest_high_time.strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(latest_high_time) else "N/A"
                    
                    st.metric("最新一次新高时间", latest_high_time_str)
                    st.metric("距离前高天数", f"{days_since_high:.2f}")
                    st.metric("收益率标准差", f"{std_dev:.4f}")
                    
                    # 如果数据中有这些列，则显示这些指标
                    if '胜率' in equity_data.columns:
                        win_rate = equity_data['胜率'].iloc[-1]
                        st.metric("胜率", f"{win_rate:.2f}%")
                    
                    if '盈亏收益比' in equity_data.columns:
                        profit_loss_ratio = equity_data['盈亏收益比'].iloc[-1]
                        st.metric("盈亏收益比", f"{profit_loss_ratio:.4f}")
                    
                    if '总手续费' in equity_data.columns:
                        total_fee = equity_data['总手续费'].iloc[-1]
                        st.metric("总手续费", f"{total_fee:.2f}")
                
                # 显示回撤统计分析
                if drawdown_events:
                    st.subheader("回撤统计分析")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("回撤持续时间分布")
                        fig_duration = go.Figure(data=[
                            go.Bar(x=duration_counts.index, y=duration_counts.values)
                        ])
                        fig_duration.update_layout(
                            xaxis_title="持续时间",
                            yaxis_title="次数",
                            height=300
                        )
                        st.plotly_chart(fig_duration, use_container_width=True)
                    
                    with col2:
                        st.write("回撤幅度分布")
                        fig_magnitude = go.Figure(data=[
                            go.Bar(x=drawdown_counts.index, y=drawdown_counts.values)
                        ])
                        fig_magnitude.update_layout(
                            xaxis_title="回撤幅度",
                            yaxis_title="次数",
                            height=300
                        )
                        st.plotly_chart(fig_magnitude, use_container_width=True)
                
                # 显示回撤事件表格
                if drawdown_events:
                    st.subheader(f"回撤范围 ({min_drawdown}% - {max_drawdown}%) 且回撤持续天数 ({min_drawdown_days} - {max_drawdown_days}天) 且新高间隔天数 ({min_high_interval_days} - {max_high_interval_days}天) 内的回撤事件")
                    
                    # 创建回撤事件的DataFrame
                    events_df = pd.DataFrame(drawdown_events)
                    
                    # 创建排序选项
                    sort_options = [
                        "默认排序", 
                        "按最大回撤幅度升序排序", 
                        "按最大回撤幅度降序排序", 
                        "按回撤持续天数升序排序", 
                        "按回撤持续天数降序排序", 
                        "按新高间隔天数升序排序", 
                        "按新高间隔天数降序排序", 
                        "按恢复天数升序排序", 
                        "按恢复天数降序排序"
                    ]
                    selected_sort = st.selectbox("选择排序方式", sort_options)
                    
                    # 根据选择进行排序
                    if selected_sort == "按最大回撤幅度升序排序":
                        events_df = events_df.sort_values(by='max_drawdown', ascending=True)
                    elif selected_sort == "按最大回撤幅度降序排序":
                        events_df = events_df.sort_values(by='max_drawdown', ascending=False)
                    elif selected_sort == "按回撤持续天数升序排序":
                        events_df = events_df.sort_values(by='drawdown_duration', ascending=True)
                    elif selected_sort == "按回撤持续天数降序排序":
                        events_df = events_df.sort_values(by='drawdown_duration', ascending=False)
                    elif selected_sort == "按新高间隔天数升序排序":
                        events_df = events_df.sort_values(by='high_interval', ascending=True)
                    elif selected_sort == "按新高间隔天数降序排序":
                        events_df = events_df.sort_values(by='high_interval', ascending=False)
                    elif selected_sort == "按恢复天数升序排序":
                        events_df = events_df.sort_values(by='recovery_duration', ascending=True)
                    elif selected_sort == "按恢复天数降序排序":
                        events_df = events_df.sort_values(by='recovery_duration', ascending=False)
                    
                    # 转换格式用于显示
                    for col in ['dd_start_time', 'dd_lowest_time', 'dd_end_time']:
                        events_df[col] = events_df[col].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(x) else "N/A")
                    
                    # 对于最后一个点的回撤，添加标记
                    if 'is_last_point' in events_df.columns:
                        events_df.loc[events_df['is_last_point'], 'dd_end_time'] = events_df.loc[events_df['is_last_point'], 'dd_end_time'] + " (最后一刻)"
                        events_df = events_df.drop('is_last_point', axis=1)
                    
                    events_df['max_drawdown'] = events_df['max_drawdown'].apply(lambda x: f"{x:.2f}%")
                    events_df['drawdown_duration'] = events_df['drawdown_duration'].apply(lambda x: f"{x:.2f}天")
                    events_df['recovery_duration'] = events_df['recovery_duration'].apply(lambda x: f"{x:.2f}天" if not pd.isna(x) else "N/A")
                    events_df['high_interval'] = events_df['high_interval'].apply(lambda x: f"{x:.2f}天" if not pd.isna(x) else "N/A")
                    
                    # 重命名列
                    events_df.columns = ['回撤开始时间', '回撤最低点时间', '下一个新高时间', '最大回撤', '回撤持续天数', '恢复天数', '新高间隔天数']
                    
                    # 添加索引列
                    events_df.insert(0, '序号', range(1, len(events_df) + 1))
                    
                    # 显示表格
                    st.dataframe(events_df)
                    
                    # 显示提示信息
                    st.info(f"说明：\n"
                           f"- 回撤开始时间：达到新高后开始回撤的时间点\n"
                           f"- 回撤最低点时间：回撤期间净值达到最低点的时间\n"
                           f"- 下一个新高时间：回撤后达到下一个新高的时间点\n"
                           f"- 回撤持续天数：从回撤开始到最低点的天数\n"
                           f"- 恢复天数：从回撤最低点到下一个新高的天数\n"
                           f"- 新高间隔天数：两个新高之间的间隔天数")
                
                # 显示关键指标说明
                with st.expander("关键指标说明"):
                    st.markdown("""
                    - **最大回撤持续天数**: 从回撤开始（净值达到高点）到回撤结束（净值达到最低点）的天数
                    - **回撤持续天数**: 在回撤事件表格中，指从回撤开始到最低点的天数
                    - **恢复天数**: 从回撤最低点到下一个新高的天数
                    - **新高间隔天数**: 两个新高之间的间隔天数
                    """)
                
                # 添加年度回撤表格
                if drawdown_events and len(drawdown_events) > 0:
                    st.subheader("年度回撤表格")
                    
                    # 获取数据中包含的所有年份
                    years = sorted(filtered_data['candle_begin_time'].dt.year.unique())
                    
                    # 对于每个年份创建一个tab
                    tabs = st.tabs([f"{year}年" for year in years])
                    
                    for i, year in enumerate(years):
                        with tabs[i]:
                            # 计算当年的开始和结束时间
                            if year == filtered_data['candle_begin_time'].dt.year.min():
                                year_start = filtered_data['candle_begin_time'].min()
                            else:
                                year_start = pd.Timestamp(f"{year}-01-01")
                                year_start = ensure_naive_timestamp(year_start)

                            if year == filtered_data['candle_begin_time'].dt.year.max():
                                year_end = filtered_data['candle_begin_time'].max()
                            else:
                                year_end = pd.Timestamp(f"{year}-12-31 23:59:59")
                                year_end = ensure_naive_timestamp(year_end)

                            # 确保时区一致性
                            year_data = filtered_data[(filtered_data['candle_begin_time'] >= year_start) & 
                                                      (filtered_data['candle_begin_time'] <= year_end)].copy()
                            
                            # 显示当年的时间范围
                            st.write(f"时间范围: {year_start.strftime('%Y-%m-%d %H:%M:%S')} 至 {year_end.strftime('%Y-%m-%d %H:%M:%S')}")
                            
                            if len(year_data) > 0:
                                # 重新计算净值，从1开始
                                year_initial_value = year_data['净值'].iloc[0]
                                year_data['年度重置净值'] = year_data['净值'] / year_initial_value
                                
                                # 计算年度最大回撤
                                year_data['年度净值_cummax'] = year_data['年度重置净值'].cummax()
                                year_data['年度drawdown'] = (year_data['年度重置净值'] - year_data['年度净值_cummax']) / year_data['年度净值_cummax']
                                year_max_dd = year_data['年度drawdown'].min() * 100
                                
                                # 找出最大回撤的开始和结束时间点
                                max_dd_end_idx = year_data['年度drawdown'].idxmin()
                                max_dd_end_time = year_data.loc[max_dd_end_idx, 'candle_begin_time']
                                
                                # 找到最大回撤开始时间（净值达到高点的时间）
                                temp_df = year_data.loc[:max_dd_end_idx]
                                max_dd_start_idx = temp_df['年度重置净值'].idxmax()
                                max_dd_start_time = year_data.loc[max_dd_start_idx, 'candle_begin_time']
                                
                                # 计算回撤持续天数
                                max_dd_duration = (max_dd_end_time - max_dd_start_time).total_seconds() / (24 * 3600)
                                
                                # 初始化年度回撤事件列表
                                year_drawdown_events = []
                                
                                # 识别当年的所有新高点和回撤事件
                                # 在年度数据上检测新高和回撤
                                year_data['is_new_high'] = year_data['年度重置净值'] >= year_data['年度重置净值'].cummax()
                                year_data['high_group'] = (year_data['is_new_high'] != year_data['is_new_high'].shift()).cumsum()
                                
                                # 找出所有新高点
                                high_points = []
                                for idx, row in year_data[year_data['is_new_high']].iterrows():
                                    # 只记录每组新高的最后一个点（即该组的最高点）
                                    group = row['high_group']
                                    group_data = year_data[year_data['high_group'] == group]
                                    if idx == group_data.index[-1]:  # 如果是该组的最后一个点
                                        high_points.append({
                                            'time': row['candle_begin_time'],
                                            'value': row['年度重置净值'],
                                            'index': idx
                                        })
                                
                                # 检测回撤事件（从一个新高到下一个新高之间的最大回撤）
                                enhanced_drawdown_events = []
                                
                                # 如果有至少两个新高点
                                if len(high_points) >= 2:
                                    for i in range(len(high_points) - 1):
                                        current_high = high_points[i]
                                        next_high = high_points[i + 1]
                                        
                                        # 获取两个新高之间的数据
                                        period_data = year_data.loc[current_high['index']:next_high['index']]
                                        
                                        if len(period_data) > 1:  # 确保有足够的数据点
                                            # 找到期间最低点
                                            min_idx = period_data['年度重置净值'].idxmin()
                                            min_value = period_data.loc[min_idx, '年度重置净值']
                                            min_time = period_data.loc[min_idx, 'candle_begin_time']
                                            
                                            # 计算最大回撤
                                            max_drawdown_pct = (min_value - current_high['value']) / current_high['value'] * 100
                                            
                                            # 计算新高间隔天数
                                            high_interval = (next_high['time'] - current_high['time']).total_seconds() / (24 * 3600)
                                            
                                            # 计算从回撤最低点到下一个新高的天数
                                            recovery_duration = (next_high['time'] - min_time).total_seconds() / (24 * 3600)
                                            
                                            # 计算从回撤开始到最低点的天数（真正的回撤持续天数）
                                            drawdown_duration = (min_time - current_high['time']).total_seconds() / (24 * 3600)
                                            
                                            # 检查是否满足筛选条件 - 修改为同时检查回撤幅度、回撤持续天数和新高间隔天数
                                            if (abs(max_drawdown_pct) >= min_drawdown and
                                                abs(max_drawdown_pct) <= max_drawdown and
                                                drawdown_duration >= min_drawdown_days and  # 使用真正的回撤持续天数
                                                drawdown_duration <= max_drawdown_days and
                                                high_interval >= min_high_interval_days and  # 新增：检查新高间隔天数
                                                high_interval <= max_high_interval_days):
                                                
                                                enhanced_drawdown_events.append({
                                                    'dd_start_time': current_high['time'],  # 回撤开始时间（当前新高）
                                                    'dd_lowest_time': min_time,  # 回撤最低点时间
                                                    'dd_end_time': next_high['time'],  # 回撤结束时间（下一个新高）
                                                    'max_drawdown': max_drawdown_pct,  # 最大回撤百分比
                                                    'drawdown_duration': drawdown_duration,  # 真正的回撤持续天数（从高点到最低点）
                                                    'recovery_duration': recovery_duration,  # 从最低点到新高的恢复天数
                                                    'high_interval': high_interval,  # 新高之间的间隔天数
                                                    'is_last_point': False  # 标记是否为最后一个点
                                                })
                                
                                # 添加最后一个高点到年度最新数据点的回撤（如果存在）
                                if len(high_points) > 0:
                                    last_high = high_points[-1]
                                    # 获取最后一个高点之后的数据
                                    last_period_data = year_data.loc[last_high['index']:]
                                    
                                    if len(last_period_data) > 1:  # 确保有足够的数据点
                                        # 计算当前回撤
                                        current_value = last_period_data['年度重置净值'].iloc[-1]
                                        current_time = last_period_data['candle_begin_time'].iloc[-1]
                                        
                                        # 找到期间最低点
                                        min_idx = last_period_data['年度重置净值'].idxmin()
                                        min_value = last_period_data.loc[min_idx, '年度重置净值']
                                        min_time = last_period_data.loc[min_idx, 'candle_begin_time']
                                        
                                        # 计算最大回撤
                                        max_drawdown_pct = (min_value - last_high['value']) / last_high['value'] * 100
                                        
                                        # 计算回撤持续天数（从最后高点到最低点）
                                        drawdown_duration = (min_time - last_high['time']).total_seconds() / (24 * 3600)
                                        
                                        # 计算从最低点到当前的天数
                                        recovery_duration = (current_time - min_time).total_seconds() / (24 * 3600)
                                        
                                        # 计算从最后高点到当前的天数
                                        high_interval = (current_time - last_high['time']).total_seconds() / (24 * 3600)
                                        
                                        # 如果当前不是新高且回撤满足条件
                                        if (current_value < last_high['value'] and 
                                            abs(max_drawdown_pct) >= min_drawdown and
                                            abs(max_drawdown_pct) <= max_drawdown and
                                            drawdown_duration >= min_drawdown_days and
                                            drawdown_duration <= max_drawdown_days and
                                            high_interval >= min_high_interval_days and
                                            high_interval <= max_high_interval_days):
                                            
                                            enhanced_drawdown_events.append({
                                                'dd_start_time': last_high['time'],  # 回撤开始时间（最后高点）
                                                'dd_lowest_time': min_time,  # 回撤最低点时间
                                                'dd_end_time': current_time,  # 当前时间作为临时结束时间
                                                'max_drawdown': max_drawdown_pct,  # 最大回撤百分比
                                                'drawdown_duration': drawdown_duration,  # 回撤持续天数
                                                'recovery_duration': recovery_duration,  # 从最低点到当前的天数
                                                'high_interval': high_interval,  # 从最后高点到当前的天数
                                                'is_last_point': True  # 标记为最后一个点
                                            })
                                
                                # 使用增强后的回撤事件列表
                                year_drawdown_events = enhanced_drawdown_events
                                
                                # 转换为DataFrame
                                year_events_df = pd.DataFrame(year_drawdown_events) if year_drawdown_events else pd.DataFrame()
                                
                                # 计算当年的回撤次数
                                st.write(f"回撤次数: {len(year_events_df)}")
                                
                                if len(year_events_df) > 0:
                                    # 创建排序选项
                                    sort_options_year = [
                                        "默认排序", 
                                        "按最大回撤幅度升序排序", 
                                        "按最大回撤幅度降序排序", 
                                        "按回撤持续天数升序排序", 
                                        "按回撤持续天数降序排序", 
                                        "按新高间隔天数升序排序", 
                                        "按新高间隔天数降序排序", 
                                        "按恢复天数升序排序", 
                                        "按恢复天数降序排序"
                                    ]
                                    selected_sort_year = st.selectbox(f"{year}年排序方式", sort_options_year, key=f"sort_{year}")
                                    
                                    # 根据选择进行排序
                                    if selected_sort_year == "按最大回撤幅度升序排序":
                                        year_events_df = year_events_df.sort_values(by='max_drawdown', ascending=True)
                                    elif selected_sort_year == "按最大回撤幅度降序排序":
                                        year_events_df = year_events_df.sort_values(by='max_drawdown', ascending=False)
                                    elif selected_sort_year == "按回撤持续天数升序排序":
                                        year_events_df = year_events_df.sort_values(by='drawdown_duration', ascending=True)
                                    elif selected_sort_year == "按回撤持续天数降序排序":
                                        year_events_df = year_events_df.sort_values(by='drawdown_duration', ascending=False)
                                    elif selected_sort_year == "按新高间隔天数升序排序":
                                        year_events_df = year_events_df.sort_values(by='high_interval', ascending=True)
                                    elif selected_sort_year == "按新高间隔天数降序排序":
                                        year_events_df = year_events_df.sort_values(by='high_interval', ascending=False)
                                    elif selected_sort_year == "按恢复天数升序排序":
                                        year_events_df = year_events_df.sort_values(by='recovery_duration', ascending=True)
                                    elif selected_sort_year == "按恢复天数降序排序":
                                        year_events_df = year_events_df.sort_values(by='recovery_duration', ascending=False)
                                    
                                    # 转换格式用于显示
                                    for col in ['dd_start_time', 'dd_lowest_time', 'dd_end_time']:
                                        year_events_df[col] = year_events_df[col].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(x) else "N/A")
                                    
                                    # 对于最后一个点的回撤，添加标记
                                    if 'is_last_point' in year_events_df.columns:
                                        year_events_df.loc[year_events_df['is_last_point'], 'dd_end_time'] = year_events_df.loc[year_events_df['is_last_point'], 'dd_end_time'] + " (最后一刻)"
                                        year_events_df = year_events_df.drop('is_last_point', axis=1)
                                    
                                    year_events_df['max_drawdown'] = year_events_df['max_drawdown'].apply(lambda x: f"{x:.2f}%")
                                    year_events_df['drawdown_duration'] = year_events_df['drawdown_duration'].apply(lambda x: f"{x:.2f}天")
                                    year_events_df['recovery_duration'] = year_events_df['recovery_duration'].apply(lambda x: f"{x:.2f}天" if not pd.isna(x) else "N/A")
                                    year_events_df['high_interval'] = year_events_df['high_interval'].apply(lambda x: f"{x:.2f}天" if not pd.isna(x) else "N/A")
                                    
                                    # 重命名列
                                    year_events_df.columns = ['回撤开始时间', '回撤最低点时间', '下一个新高时间', '最大回撤', '回撤持续天数', '恢复天数', '新高间隔天数']
                                    
                                    # 添加索引列
                                    year_events_df.insert(0, '序号', range(1, len(year_events_df) + 1))
                                    
                                    # 显示表格
                                    st.dataframe(year_events_df)
                                    
                                    # 显示提示信息
                                    st.info(f"说明：\n"
                                           f"- 回撤开始时间：达到新高后开始回撤的时间点\n"
                                           f"- 回撤最低点时间：回撤期间净值达到最低点的时间\n"
                                           f"- 下一个新高时间：回撤后达到下一个新高的时间点\n"
                                           f"- 回撤持续天数：从回撤开始到最低点的天数\n"
                                           f"- 恢复天数：从回撤最低点到下一个新高的天数\n"
                                           f"- 新高间隔天数：两个新高之间的间隔天数")
                                
                                # 计算当年收益率
                                year_return = (year_data['年度重置净值'].iloc[-1] / year_data['年度重置净值'].iloc[0] - 1) * 100
                                
                                # 显示当年关键指标
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric(f"{year}年收益率", f"{year_return:.2f}%")
                                with col2:
                                    st.metric(f"{year}年最大回撤", f"{year_max_dd:.2f}%")
                                    st.metric(f"{year}年最大回撤持续天数", f"{max_dd_duration:.2f}天")
                                with col3:
                                    st.metric(f"{year}年回撤次数", f"{len(year_events_df)}")
                                
                                # 添加解释说明
                                st.info(f"注意：这里显示的'{year}年最大回撤'是该年度内的实际最大回撤。'{year}年最大回撤持续天数'是从回撤开始（净值达到高点）到回撤结束（净值达到最低点）的天数。表格中显示的是符合筛选条件(回撤范围{min_drawdown}%-{max_drawdown}%，持续天数{min_drawdown_days}-{max_drawdown_days}天)的回撤事件。")
                                
                                # 显示当年资金曲线图
                                fig_year = go.Figure()
                                fig_year.add_trace(go.Scatter(
                                    x=year_data['candle_begin_time'],
                                    y=year_data['年度重置净值'],
                                    mode='lines',
                                    name='净值'
                                ))
                                
                                fig_year.update_layout(
                                    title=f"{year}年资金曲线图",
                                    xaxis_title="日期",
                                    yaxis_title="净值",
                                    height=400,
                                    hovermode="x unified"
                                )
                                st.plotly_chart(fig_year, use_container_width=True)
                                
                                # 显示当年回撤曲线
                                fig_year_dd = go.Figure()
                                fig_year_dd.add_trace(go.Scatter(
                                    x=year_data['candle_begin_time'],
                                    y=year_data['年度drawdown'] * 100,
                                    mode='lines',
                                    name='回撤百分比',
                                    line=dict(color='red')
                                ))
                                
                                fig_year_dd.update_layout(
                                    title=f"{year}年回撤曲线图",
                                    xaxis_title="日期",
                                    yaxis_title="回撤百分比 (%)",
                                    height=300,
                                    hovermode="x unified"
                                )
                                st.plotly_chart(fig_year_dd, use_container_width=True)
                            else:
                                st.write(f"{year}年没有数据")
                
                # 显示原始数据表格
                with st.expander("查看原始数据"):
                    st.dataframe(filtered_data)
    
    except Exception as e:
        st.error(f"处理文件时出错: {e}")
else:
    # 显示使用说明
    st.info("""
    ### 使用说明
    1. 点击"上传资金曲线CSV文件"按钮上传您的资金曲线数据
    2. 上传成功后，您可以选择时间范围进行分析
    3. 您还可以设置回撤范围和回撤持续天数范围，查看指定条件内的回撤次数和详情
    4. 系统将自动计算并显示关键指标和图表
    
    ### 文件格式要求
    上传的CSV文件必须包含以下列：
    - candle_begin_time: 时间戳列
    - 净值: 账户净值列
    
    其他可选列：
    - 胜率
    - 盈亏收益比
    - 总手续费
    """)

st.sidebar.title("关于")
st.sidebar.info("""
### 资金曲线分析工具
这是一个用于分析交易策略资金曲线的工具，可以帮助您评估策略的性能和风险。

### 功能
- 上传资金曲线CSV文件
- 选择时间范围进行分析
- 设置回撤范围和回撤持续天数，统计回撤事件
- 计算关键指标
- 可视化资金曲线和回撤
- 回撤统计分析
- 按年度查看回撤事件和表现

### 指标说明
- **累积净值**: 策略最终的净值
- **年化收益**: 年化收益率
- **最大回撤**: 最大回撤百分比
- **最大回撤持续天数**: 从回撤开始（净值达到高点）到回撤结束（净值达到最低点）的天数
- **当前回撤百分比**: 当前相对于历史最高点的回撤百分比
- **不创新高持续天数**: 最长的不创新高持续天数
- **距离前高天数**: 距离最近一次创新高的天数
- **收益率标准差**: 收益率的标准差，反映波动性
""")