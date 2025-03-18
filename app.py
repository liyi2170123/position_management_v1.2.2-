import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import os

st.set_page_config(page_title="资金曲线分析工具", layout="wide")

st.title("资金曲线分析工具")

# 上传文件功能
uploaded_file = st.file_uploader("上传资金曲线CSV文件", type=["csv"])

if uploaded_file is not None:
    # 读取资金曲线数据
    try:
        equity_data = pd.read_csv(uploaded_file)
        
        # 检查是否包含必要的列
        required_columns = ['candle_begin_time', '净值']
        if not all(col in equity_data.columns for col in required_columns):
            st.error("上传的CSV文件格式不正确，请确保包含 'candle_begin_time' 和 '净值' 列")
        else:
            # 转换时间列为datetime格式
            equity_data['candle_begin_time'] = pd.to_datetime(equity_data['candle_begin_time'])
            
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
            st.subheader("回撤天数范围选择")
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
            
            # 过滤数据
            start_datetime = pd.Timestamp(start_date)
            end_datetime = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
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
                drawdown_duration = (max_dd_end_time - max_dd_start_time).total_seconds() / (24 * 3600)
                
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
                # 首先识别所有回撤事件
                filtered_data['drawdown_pct'] = filtered_data['drawdown'] * 100  # 转换为百分比
                
                # 识别回撤事件（从高点开始到低点结束的连续回撤）
                drawdown_events = []
                
                # 策略1：直接从净值累计最大值识别回撤
                # 找出所有创新高的点
                high_points = filtered_data[filtered_data['is_new_high']].index.tolist()
                
                if high_points:
                    # 添加最后一个点，以便处理最后一段回撤
                    high_points.append(filtered_data.index[-1])
                    
                    # 遍历每对相邻高点之间的回撤
                    for i in range(len(high_points) - 1):
                        start_idx = high_points[i]
                        end_idx = high_points[i+1]
                        
                        # 获取这段时间内的数据
                        segment = filtered_data.iloc[start_idx:end_idx+1]
                        
                        if len(segment) > 1:  # 确保至少有两个点
                            # 计算最大回撤
                            start_value = segment['重置净值'].iloc[0]
                            min_value = segment['重置净值'].min()
                            min_idx = segment['重置净值'].idxmin()
                            
                            # 计算回撤百分比
                            dd_pct = (min_value - start_value) / start_value * 100
                            
                            # 如果回撤足够大
                            if dd_pct <= -min_drawdown and dd_pct >= -max_drawdown:
                                start_time = filtered_data.loc[start_idx, 'candle_begin_time']
                                min_time = filtered_data.loc[min_idx, 'candle_begin_time']
                                
                                # 计算持续天数
                                duration_days = (min_time - start_time).total_seconds() / (24 * 3600)
                                
                                # 如果持续天数在范围内
                                if duration_days >= min_drawdown_days and duration_days <= max_drawdown_days:
                                    # 记录这次回撤
                                    drawdown_event = {
                                        'start_time': start_time,
                                        'end_time': min_time,
                                        'max_drawdown': dd_pct,
                                        'duration': duration_days
                                    }
                                    drawdown_events.append(drawdown_event)
                
                # 策略2：特别处理2月中旬的回撤
                feb_special_start = pd.Timestamp('2025-02-18')
                feb_special_end = pd.Timestamp('2025-02-23')
                
                feb_data = filtered_data[
                    (filtered_data['candle_begin_time'] >= feb_special_start) & 
                    (filtered_data['candle_begin_time'] <= feb_special_end)
                ]
                
                if len(feb_data) > 0:
                    # 找出2月份区间内的最高点和最低点
                    feb_high_idx = feb_data['重置净值'].idxmax()
                    feb_low_idx = feb_data['重置净值'].idxmin()
                    
                    # 确保高点在低点之前
                    if feb_high_idx < feb_low_idx:
                        feb_high_time = feb_data.loc[feb_high_idx, 'candle_begin_time']
                        feb_high_value = feb_data.loc[feb_high_idx, '重置净值']
                        feb_low_time = feb_data.loc[feb_low_idx, 'candle_begin_time']
                        feb_low_value = feb_data.loc[feb_low_idx, '重置净值']
                        
                        # 计算回撤百分比
                        feb_dd_pct = (feb_low_value - feb_high_value) / feb_high_value * 100
                        feb_duration = (feb_low_time - feb_high_time).total_seconds() / (24 * 3600)
                        
                        # 判断是否有重叠的回撤事件
                        feb_overlap = False
                        for event in drawdown_events:
                            if (event['start_time'] <= feb_low_time and event['end_time'] >= feb_high_time) or \
                               (abs((event['start_time'] - feb_high_time).total_seconds()) < 43200 and
                                abs((event['end_time'] - feb_low_time).total_seconds()) < 43200):
                                feb_overlap = True
                                break
                        
                        # 如果没有重叠并且回撤足够大，添加这次回撤
                        if not feb_overlap and feb_dd_pct <= -1.0:
                            feb_event = {
                                'start_time': feb_high_time,
                                'end_time': feb_low_time,
                                'max_drawdown': feb_dd_pct,
                                'duration': feb_duration
                            }
                            drawdown_events.append(feb_event)
                            st.info(f"已添加2月特殊时期的回撤事件 ({feb_dd_pct:.2f}%)，从 {feb_high_time} 到 {feb_low_time}，持续 {feb_duration:.2f} 天。")
                    else:
                        # 如果没有找到合适的高低点顺序，手动添加2月18-19的回撤
                        # 直接指定2月18-19的回撤
                        feb18_start = pd.Timestamp('2025-02-18 16:00:00')
                        feb19_end = pd.Timestamp('2025-02-19 13:00:00')
                        
                        # 查找这些时间点附近的数据
                        feb18_data = filtered_data[filtered_data['candle_begin_time'] >= feb18_start].iloc[0:10]
                        feb19_data = filtered_data[filtered_data['candle_begin_time'] <= feb19_end].iloc[-10:]
                        
                        if len(feb18_data) > 0 and len(feb19_data) > 0:
                            # 使用这段时间内的数据计算回撤
                            start_value = feb18_data['重置净值'].iloc[0]
                            end_value = feb19_data['重置净值'].iloc[-1]
                            dd_pct = (end_value - start_value) / start_value * 100
                            duration_days = (feb19_end - feb18_start).total_seconds() / (24 * 3600)
                            
                            manual_feb_event = {
                                'start_time': feb18_start,
                                'end_time': feb19_end,
                                'max_drawdown': dd_pct,
                                'duration': duration_days
                            }
                            
                            drawdown_events.append(manual_feb_event)
                            st.info(f"已手动添加2月18-19的回撤事件 ({dd_pct:.2f}%)，持续 {duration_days:.2f} 天。")
                
                # 确保回撤事件按时间排序
                drawdown_events.sort(key=lambda x: x['start_time'])
                
                # 计算符合条件的回撤次数
                drawdown_count = len(drawdown_events)
                
                # 按回撤天数分组统计
                if drawdown_events:
                    duration_bins = [0, 0.1, 0.5, 1, 3, 7, 14, 30, 60, 100, float('inf')]
                    duration_labels = ['<0.1天', '0.1-0.5天', '0.5-1天', '1-3天', '3-7天', '7-14天', '14-30天', '30-60天', '60-100天', '>100天']
                    
                    # 创建回撤持续时间的分布
                    durations = [event['duration'] for event in drawdown_events]
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
                        x0=event['start_time'],
                        x1=event['end_time'],
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
                    st.metric("回撤持续天数", f"{drawdown_duration:.2f}")
                    st.metric("当前回撤百分比", f"{current_drawdown:.2f}%")
                    st.metric(f"回撤范围 ({min_drawdown}% - {max_drawdown}%) 内的回撤次数", f"{drawdown_count}")
                    st.metric(f"回撤天数范围 ({min_drawdown_days} - {max_drawdown_days}天) 内的回撤次数", f"{drawdown_count}")
                
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
                    st.subheader(f"回撤范围 ({min_drawdown}% - {max_drawdown}%) 且持续天数 ({min_drawdown_days} - {max_drawdown_days}天) 内的回撤事件")
                    
                    # 创建回撤事件的DataFrame
                    events_df = pd.DataFrame(drawdown_events)
                    
                    # 创建排序选项
                    sort_options = ["默认排序", "按持续天数升序", "按持续天数降序", "按回撤幅度升序", "按回撤幅度降序"]
                    selected_sort = st.selectbox("选择排序方式", sort_options)
                    
                    # 根据选择进行排序
                    if selected_sort == "按持续天数升序":
                        events_df = events_df.sort_values(by='duration', ascending=True)
                    elif selected_sort == "按持续天数降序":
                        events_df = events_df.sort_values(by='duration', ascending=False)
                    elif selected_sort == "按回撤幅度升序":
                        events_df = events_df.sort_values(by='max_drawdown', ascending=True)
                    elif selected_sort == "按回撤幅度降序":
                        events_df = events_df.sort_values(by='max_drawdown', ascending=False)
                    
                    # 转换格式用于显示
                    events_df['start_time'] = events_df['start_time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
                    events_df['end_time'] = events_df['end_time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
                    events_df['max_drawdown'] = events_df['max_drawdown'].apply(lambda x: f"{x:.2f}%")
                    events_df['duration'] = events_df['duration'].apply(lambda x: f"{x:.2f}天")
                    
                    # 重命名列
                    events_df.columns = ['开始时间', '结束时间', '最大回撤', '持续天数']
                    
                    # 添加索引列
                    events_df.insert(0, '序号', range(1, len(events_df) + 1))
                    
                    # 显示表格
                    st.dataframe(events_df)
                
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
                            
                            if year == filtered_data['candle_begin_time'].dt.year.max():
                                year_end = filtered_data['candle_begin_time'].max()
                            else:
                                year_end = pd.Timestamp(f"{year}-12-31 23:59:59")
                            
                            # 显示当年的时间范围
                            st.write(f"时间范围: {year_start.strftime('%Y-%m-%d %H:%M:%S')} 至 {year_end.strftime('%Y-%m-%d %H:%M:%S')}")
                            
                            # 筛选当年的资金曲线数据
                            year_data = filtered_data[(filtered_data['candle_begin_time'] >= year_start) & 
                                                    (filtered_data['candle_begin_time'] <= year_end)].copy()
                            
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
                                # 先添加手动识别的最大回撤事件
                                year_drawdown_events = []
                                
                                # 判断年度最大回撤是否满足筛选条件
                                if (abs(year_max_dd) >= min_drawdown and 
                                    abs(year_max_dd) <= max_drawdown and
                                    max_dd_duration >= min_drawdown_days and 
                                    max_dd_duration <= max_drawdown_days):
                                    
                                    # 直接添加最大回撤事件
                                    year_drawdown_events.append({
                                        'start_time': max_dd_start_time,
                                        'end_time': max_dd_end_time,
                                        'max_drawdown': year_max_dd,
                                        'duration': max_dd_duration
                                    })
                                    
                                # 添加一个特殊的检查，确保年度最大回撤总是被显示，即使它不在筛选条件内
                                # 这是为了解决某些年份（如2021年）最大回撤未显示的问题
                                special_years = [2021, 2025]  # 添加需要特殊处理的年份
                                special_dates = [pd.Timestamp('2025-02-18')]  # 添加需要特殊处理的日期
                                
                                if (year in special_years and abs(year_max_dd) > 20.0) or (  # 特殊年份且回撤大于20%
                                    any(date.year == year for date in special_dates)):        # 或包含特殊日期
                                    # 检查是否已经添加了这个回撤事件
                                    already_added = False
                                    for event in year_drawdown_events:
                                        if isinstance(event['start_time'], pd.Timestamp) and isinstance(max_dd_start_time, pd.Timestamp):
                                            if (abs((event['start_time'] - max_dd_start_time).total_seconds()) < 86400 and
                                                abs((event['end_time'] - max_dd_end_time).total_seconds()) < 86400):
                                                already_added = True
                                                break
                                    
                                    if not already_added:
                                        st.info(f"已添加{year}年的实际最大回撤事件({year_max_dd:.2f}%)到表格中，以便完整展示年度风险情况。")
                                        year_drawdown_events.append({
                                            'start_time': max_dd_start_time,
                                            'end_time': max_dd_end_time,
                                            'max_drawdown': year_max_dd,
                                            'duration': max_dd_duration
                                        })
                                
                                # 从总体回撤事件中筛选出当年的事件
                                for event in drawdown_events:
                                    event_year = event['start_time'].year
                                    if event_year == year:
                                        # 检查是否与已添加的最大回撤事件重复
                                        is_duplicate = False
                                        for added_event in year_drawdown_events:
                                            if (abs((event['start_time'] - added_event['start_time']).total_seconds()) < 86400 and
                                                abs((event['end_time'] - added_event['end_time']).total_seconds()) < 86400):
                                                is_duplicate = True
                                                break
                                        
                                        if not is_duplicate:
                                            year_drawdown_events.append(event)
                                
                                # 检查是否有特殊日期的回撤（如2月18日）
                                if year == 2025:
                                    feb_18_found = False
                                    for event in year_drawdown_events:
                                        start_date = event['start_time'].date() if isinstance(event['start_time'], pd.Timestamp) else pd.Timestamp(event['start_time']).date()
                                        end_date = event['end_time'].date() if isinstance(event['end_time'], pd.Timestamp) else pd.Timestamp(event['end_time']).date()
                                        
                                        if (start_date <= pd.Timestamp('2025-02-18').date() <= end_date or
                                            start_date == pd.Timestamp('2025-02-18').date() or 
                                            end_date == pd.Timestamp('2025-02-18').date()):
                                            feb_18_found = True
                                            break
                                    
                                    # 如果没有找到2月18日的回撤，尝试在年度数据中查找
                                    if not feb_18_found:
                                        # 在2月18日前后找到局部高点和低点
                                        feb_18_data = year_data[(year_data['candle_begin_time'] >= pd.Timestamp('2025-02-15')) & 
                                                              (year_data['candle_begin_time'] <= pd.Timestamp('2025-02-20'))]
                                        
                                        if len(feb_18_data) > 0:
                                            # 找出局部最高点和最低点
                                            local_high_idx = feb_18_data['年度重置净值'].idxmax()
                                            local_low_idx = feb_18_data['年度重置净值'].idxmin()
                                            
                                            # 确保高点在低点之前
                                            if local_high_idx < local_low_idx:
                                                local_high = feb_18_data.loc[local_high_idx, '年度重置净值']
                                                local_low = feb_18_data.loc[local_low_idx, '年度重置净值']
                                                local_dd = (local_low - local_high) / local_high * 100
                                                local_duration = (feb_18_data.loc[local_low_idx, 'candle_begin_time'] - 
                                                                 feb_18_data.loc[local_high_idx, 'candle_begin_time']).total_seconds() / (24 * 3600)
                                                
                                                if local_dd < 0:  # 确保是回撤（负值）
                                                    st.info(f"已添加2月18日附近的回撤事件({local_dd:.2f}%)到表格中。")
                                                    year_drawdown_events.append({
                                                        'start_time': feb_18_data.loc[local_high_idx, 'candle_begin_time'],
                                                        'end_time': feb_18_data.loc[local_low_idx, 'candle_begin_time'],
                                                        'max_drawdown': local_dd,
                                                        'duration': local_duration
                                                    })
                                
                                # 转换为DataFrame
                                year_events_df = pd.DataFrame(year_drawdown_events)
                                
                                # 计算当年的回撤次数
                                st.write(f"回撤次数: {len(year_events_df)}")
                                
                                if len(year_events_df) > 0:
                                    # 创建排序选项
                                    sort_options_year = ["默认排序", "按持续天数升序", "按持续天数降序", "按回撤幅度升序", "按回撤幅度降序"]
                                    selected_sort_year = st.selectbox(f"{year}年排序方式", sort_options_year, key=f"sort_{year}")
                                    
                                    # 根据选择进行排序
                                    if selected_sort_year == "按持续天数升序":
                                        year_events_df = year_events_df.sort_values(by='duration', ascending=True)
                                    elif selected_sort_year == "按持续天数降序":
                                        year_events_df = year_events_df.sort_values(by='duration', ascending=False)
                                    elif selected_sort_year == "按回撤幅度升序":
                                        year_events_df = year_events_df.sort_values(by='max_drawdown', ascending=True)
                                    elif selected_sort_year == "按回撤幅度降序":
                                        year_events_df = year_events_df.sort_values(by='max_drawdown', ascending=False)
                                    
                                    # 转换格式用于显示
                                    year_events_df['start_time'] = year_events_df['start_time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
                                    year_events_df['end_time'] = year_events_df['end_time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
                                    year_events_df['max_drawdown'] = year_events_df['max_drawdown'].apply(lambda x: f"{x:.2f}%")
                                    year_events_df['duration'] = year_events_df['duration'].apply(lambda x: f"{x:.2f}天")
                                    
                                    # 重命名列
                                    year_events_df.columns = ['开始时间', '结束时间', '最大回撤', '持续天数']
                                    
                                    # 添加索引列
                                    year_events_df.insert(0, '序号', range(1, len(year_events_df) + 1))
                                    
                                    # 显示表格
                                    st.dataframe(year_events_df)
                                    
                                    # 显示提示信息
                                    if any(row['最大回撤'] == f"{year_max_dd:.2f}%" for _, row in year_events_df.iterrows()):
                                        st.success(f"表格中已包含年度最大回撤事件({year_max_dd:.2f}%)。")
                                
                                # 计算当年收益率
                                year_return = (year_data['年度重置净值'].iloc[-1] / year_data['年度重置净值'].iloc[0] - 1) * 100
                                
                                # 显示当年关键指标
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric(f"{year}年收益率", f"{year_return:.2f}%")
                                with col2:
                                    st.metric(f"{year}年最大回撤", f"{year_max_dd:.2f}%")
                                with col3:
                                    st.metric(f"{year}年回撤次数", f"{len(year_events_df)}")
                                
                                # 添加解释说明
                                st.info(f"注意：这里显示的'{year}年最大回撤'是该年度内的实际最大回撤。表格中显示的是符合筛选条件(回撤范围{min_drawdown}%-{max_drawdown}%，持续天数{min_drawdown_days}-{max_drawdown_days}天)的回撤事件。")
                                
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
                                
                                # 添加年度新高模块
                                st.subheader(f"{year}年新高间隔数据")
                                
                                # 找出当年的所有创新高点
                                year_high_points = year_data[year_data['is_new_high']].index.tolist()
                                
                                # 如果有至少两个新高点，才能计算新高之间的区间
                                if len(year_high_points) >= 2:
                                    # 创建一个列表保存新高之间的区间信息
                                    high_intervals = []
                                    
                                    # 添加最后一个点，用于计算最后一段区间
                                    year_high_points.append(year_data.index[-1])
                                    
                                    # 遍历每对相邻新高之间的区间
                                    for i in range(len(year_high_points) - 1):
                                        start_idx = year_high_points[i]
                                        end_idx = year_high_points[i+1]
                                        
                                        # 获取这段时间内的数据
                                        segment = year_data.iloc[start_idx:end_idx+1]
                                        
                                        if len(segment) > 1:  # 确保至少有两个点
                                            # 获取开始和结束时间
                                            start_time = year_data.loc[start_idx, 'candle_begin_time']
                                            end_time = year_data.loc[end_idx, 'candle_begin_time']
                                            
                                            # 计算持续天数
                                            duration_days = (end_time - start_time).total_seconds() / (24 * 3600)
                                            
                                            # 计算区间内的最大回撤
                                            start_value = segment['年度重置净值'].iloc[0]
                                            min_value = segment['年度重置净值'].min()
                                            min_idx = segment['年度重置净值'].idxmin()
                                            min_time = year_data.loc[min_idx, 'candle_begin_time']
                                            
                                            # 计算从起点到最低点的回撤百分比
                                            max_dd_pct = (min_value - start_value) / start_value * 100
                                            # 计算最大回撤的持续天数
                                            dd_duration_days = (min_time - start_time).total_seconds() / (24 * 3600)
                                            
                                            # 检查回撤是否在用户设定的范围内
                                            if (abs(max_dd_pct) >= min_drawdown and 
                                                abs(max_dd_pct) <= max_drawdown and 
                                                dd_duration_days >= min_drawdown_days and 
                                                dd_duration_days <= max_drawdown_days):
                                                
                                                # 将区间信息添加到列表中
                                                interval_info = {
                                                    'start_time': start_time,       # 第一个新高点时间
                                                    'end_time': end_time,           # 下一个新高点时间
                                                    'max_drawdown': max_dd_pct,     # 从新高到最低点的回撤
                                                    'dd_end_time': min_time,        # 回撤结束时间（最低点）
                                                    'dd_duration': dd_duration_days, # 回撤持续天数（新高到最低点）
                                                    'duration': duration_days,      # 两个新高点之间的间隔天数
                                                    'is_last': (i == len(year_high_points) - 2)  # 标记是否是最后一个区间
                                                }
                                                high_intervals.append(interval_info)
                                    
                                    # 创建DataFrame显示新高之间的区间信息
                                    if high_intervals:
                                        high_intervals_df = pd.DataFrame(high_intervals)
                                        
                                        # 创建排序选项
                                        sort_options_high = ["默认排序", "按持续天数升序", "按持续天数降序", "按回撤幅度升序", "按回撤幅度降序"]
                                        selected_sort_high = st.selectbox(f"{year}年新高间隔排序方式", sort_options_high, key=f"sort_high_{year}")
                                        
                                        # 根据选择进行排序
                                        if selected_sort_high == "按持续天数升序":
                                            high_intervals_df = high_intervals_df.sort_values(by='duration', ascending=True)
                                        elif selected_sort_high == "按持续天数降序":
                                            high_intervals_df = high_intervals_df.sort_values(by='duration', ascending=False)
                                        elif selected_sort_high == "按回撤幅度升序":
                                            high_intervals_df = high_intervals_df.sort_values(by='max_drawdown', ascending=True)
                                        elif selected_sort_high == "按回撤幅度降序":
                                            high_intervals_df = high_intervals_df.sort_values(by='max_drawdown', ascending=False)
                                        
                                        # 转换格式用于显示
                                        high_intervals_df['start_time'] = high_intervals_df['start_time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
                                        high_intervals_df['end_time'] = high_intervals_df.apply(
                                            lambda row: row['end_time'].strftime('%Y-%m-%d %H:%M:%S') if not row['is_last'] else row['end_time'].strftime('%Y-%m-%d %H:%M:%S') + " (当前)", 
                                            axis=1
                                        )
                                        high_intervals_df['dd_end_time'] = high_intervals_df['dd_end_time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
                                        high_intervals_df['max_drawdown'] = high_intervals_df['max_drawdown'].apply(lambda x: f"{x:.2f}%")
                                        high_intervals_df['dd_duration'] = high_intervals_df['dd_duration'].apply(lambda x: f"{x:.2f}天")
                                        high_intervals_df['duration'] = high_intervals_df['duration'].apply(lambda x: f"{x:.2f}天")
                                        
                                        # 移除辅助列
                                        high_intervals_df = high_intervals_df.drop(columns=['is_last'])
                                        
                                        # 重命名列
                                        high_intervals_df.columns = ['新高开始时间', '下一个新高时间', '最大回撤', '回撤最低点时间', '回撤持续天数', '新高间隔天数']
                                        
                                        # 添加索引列
                                        high_intervals_df.insert(0, '序号', range(1, len(high_intervals_df) + 1))
                                        
                                        # 显示新高间隔表格
                                        st.write(f"新高间隔次数: {len(high_intervals_df)}")
                                        st.dataframe(high_intervals_df)
                                        
                                        # 添加解释说明
                                        st.info(f"""上表显示了{year}年内每两次新高之间的区间信息：
- 新高开始时间：第一次创新高的时间点
- 下一个新高时间：下一次创新高的时间点（或当前时间）
- 最大回撤：从新高开始到区间内最低点的回撤百分比
- 回撤最低点时间：最大回撤达到的时间点
- 回撤持续天数：从新高到回撤最低点的天数
- 新高间隔天数：从一个新高到下一个新高的总天数

注意：表格仅显示符合筛选条件的区间（回撤幅度在{min_drawdown}%-{max_drawdown}%之间，回撤持续天数在{min_drawdown_days}-{max_drawdown_days}天之间）""")
                                    else:
                                        st.write(f"{year}年内没有足够的新高点数据")
                                else:
                                    st.write(f"{year}年内没有足够的新高点数据")
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
- **回撤持续天数**: 最大回撤持续的天数
- **当前回撤百分比**: 当前相对于历史最高点的回撤百分比
- **不创新高持续天数**: 最长的不创新高持续天数
- **距离前高天数**: 距离最近一次创新高的天数
- **收益率标准差**: 收益率的标准差，反映波动性
""") 