"""
Dashboard Screen Implementations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Any


class DashboardScreens:
    """Implementation of all dashboard screens"""
    
    def __init__(self, dashboard):
        self.dashboard = dashboard
    
    def show_top_opportunities_screen(self):
        """Show top opportunities screen with full functionality"""
        st.markdown("## 🎯 Top Opportunity Positions")
        
        # Generate opportunities if not available
        if not st.session_state.opportunities_data:
            st.session_state.opportunities_data = self.dashboard.generate_realistic_opportunities()
        
        opportunities = st.session_state.opportunities_data
        
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            status_filter = st.selectbox("Status Filter", ['All'] + list(set([o['status'] for o in opportunities])))
        with col2:
            confidence_filter = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.1)
        with col3:
            return_filter = st.slider("Min Expected Return (%)", -20.0, 50.0, -20.0, 1.0)
        with col4:
            agent_filter = st.selectbox("Agent Filter", ['All'] + list(set([o['agent_type'] for o in opportunities])))
        
        # Apply filters
        filtered_opportunities = opportunities.copy()
        if status_filter != 'All':
            filtered_opportunities = [o for o in filtered_opportunities if o['status'] == status_filter]
        filtered_opportunities = [o for o in filtered_opportunities if o['confidence'] >= confidence_filter]
        filtered_opportunities = [o for o in filtered_opportunities if o['expected_return'] >= return_filter]
        if agent_filter != 'All':
            filtered_opportunities = [o for o in filtered_opportunities if o['agent_type'] == agent_filter]
        
        # Display table
        if filtered_opportunities:
            df = pd.DataFrame(filtered_opportunities)
            st.dataframe(
                df[[
                    'ticker', 'current_price', 'target_price', 'stop_loss', 
                    'market_order_type', 'risk', 'position_size', 'reasoning',
                    'confidence', 'expected_return', 'status', 'agent_type', 'timestamp'
                ]].round(2),
                use_container_width=True,
                hide_index=True
            )
            
            # Action buttons for each opportunity
            st.markdown("### 🎯 Opportunity Actions")
            selected_opportunity = st.selectbox(
                "Select Opportunity for Actions",
                options=filtered_opportunities,
                format_func=lambda x: f"{x['ticker']} - {x['reasoning']} (Confidence: {x['confidence']:.2%})"
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔍 Show Decision Reasoning", key="show_reasoning"):
                    self.show_reasoning_visualization(selected_opportunity)
            
            with col2:
                if st.button("📊 Execute Position", key="execute_position"):
                    self.execute_position(selected_opportunity)
            
            with col3:
                if st.button("📈 Add to Watchlist", key="add_watchlist"):
                    st.success(f"Added {selected_opportunity['ticker']} to watchlist!")
        else:
            st.warning("No opportunities match the current filters.")
        
        # Summary statistics
        self.create_summary_statistics(opportunities)
    
    def show_reasoning_visualization(self, opportunity):
        """Show decision reasoning visualization"""
        st.markdown("### 🧠 Decision Reasoning Visualization")
        
        # Create mind map visualization
        fig = go.Figure()
        
        # Central node (opportunity)
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers+text',
            marker=dict(size=30, color='red'),
            text=[opportunity['ticker']],
            textposition="middle center",
            name='Opportunity'
        ))
        
        # Reasoning factors
        reasoning_factors = {
            'Technical Analysis': [1, 1],
            'Fundamental Analysis': [-1, 1],
            'Sentiment Analysis': [1, -1],
            'ML Prediction': [-1, -1],
            'Market Conditions': [0, 2]
        }
        
        colors = ['blue', 'green', 'orange', 'purple', 'brown']
        
        for i, (factor, pos) in enumerate(reasoning_factors.items()):
            # Add factor node
            fig.add_trace(go.Scatter(
                x=[pos[0]], y=[pos[1]],
                mode='markers+text',
                marker=dict(size=20, color=colors[i]),
                text=[factor],
                textposition="middle center",
                name=factor
            ))
            
            # Add connection line
            fig.add_trace(go.Scatter(
                x=[0, pos[0]], y=[0, pos[1]],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False
            ))
        
        fig.update_layout(
            title=f"Decision Reasoning for {opportunity['ticker']}",
            xaxis=dict(range=[-2, 2], showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(range=[-2, 2], showgrid=False, zeroline=False, showticklabels=False),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Reasoning details
        st.markdown("#### 📋 Reasoning Details")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Confidence Score", f"{opportunity['confidence']:.2%}")
            st.metric("Expected Return", f"{opportunity['expected_return']:.2f}%")
            st.metric("Risk Level", f"{opportunity['risk']:.3f}")
        
        with col2:
            st.metric("Position Size", f"${opportunity['position_size']:,.0f}")
            st.metric("Current Price", f"${opportunity['current_price']:.2f}")
            st.metric("Target Price", f"${opportunity['target_price']:.2f}")
    
    def execute_position(self, opportunity):
        """Execute a position from opportunity"""
        st.success(f"Position executed for {opportunity['ticker']}!")
        
        # Add to open positions
        position = {
            'id': f"POS_{len(st.session_state.open_positions) + 1:04d}",
            'ticker': opportunity['ticker'],
            'entry_price': opportunity['current_price'],
            'current_price': opportunity['current_price'],
            'target_price': opportunity['target_price'],
            'stop_loss': opportunity['stop_loss'],
            'quantity': opportunity['position_size'] / opportunity['current_price'],
            'position_size': opportunity['position_size'],
            'entry_time': datetime.now(),
            'status': 'Open',
            'unrealized_pnl': 0,
            'pnl_percentage': 0
        }
        
        st.session_state.open_positions.append(position)
    
    def create_summary_statistics(self, opportunities):
        """Create summary statistics"""
        st.markdown("---")
        st.markdown("### 📈 Summary Statistics")
        
        if not opportunities:
            return
        
        df = pd.DataFrame(opportunities)
        
        # Calculate statistics
        daily_opportunities = len(df[df['timestamp'] >= datetime.now() - timedelta(days=1)])
        overall_predicted_returns = df['expected_return'].mean()
        current_open_positions = len(st.session_state.open_positions)
        pending_positions = len(st.session_state.pending_positions)
        total_confidence = df['confidence'].mean()
        total_risk = df['risk'].mean()
        
        # Display metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Daily Opportunities", daily_opportunities)
        
        with col2:
            st.metric("Overall Predicted Returns", f"{overall_predicted_returns:.2f}%")
        
        with col3:
            st.metric("Current Open Positions", current_open_positions)
        
        with col4:
            st.metric("Pending Positions", pending_positions)
        
        with col5:
            st.metric("Average Confidence", f"{total_confidence:.2%}")
        
        with col6:
            st.metric("Average Risk", f"{total_risk:.3f}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            status_counts = df['status'].value_counts()
            fig1 = px.pie(values=status_counts.values, names=status_counts.index, title="Opportunities by Status")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.histogram(df, x='expected_return', title="Expected Returns Distribution")
            st.plotly_chart(fig2, use_container_width=True)
    
    def show_open_positions_screen(self):
        """Show open positions screen with position management"""
        st.markdown("## 📈 Open Positions")
        
        # Generate some open positions if none exist
        if not st.session_state.open_positions:
            st.session_state.open_positions = self.generate_open_positions()
        
        positions = st.session_state.open_positions
        
        if positions:
            # Display positions table
            df = pd.DataFrame(positions)
            st.dataframe(
                df[[
                    'ticker', 'entry_price', 'current_price', 'target_price', 'stop_loss',
                    'quantity', 'position_size', 'unrealized_pnl', 'pnl_percentage', 'entry_time'
                ]].round(2),
                use_container_width=True,
                hide_index=True
            )
            
            # Position management
            st.markdown("### 🎛️ Position Management")
            selected_position = st.selectbox(
                "Select Position to Manage",
                options=positions,
                format_func=lambda x: f"{x['ticker']} - P&L: ${x['unrealized_pnl']:.2f} ({x['pnl_percentage']:.2f}%)"
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Modify Position")
                new_target = st.number_input("New Target Price", value=float(selected_position['target_price']))
                new_stop = st.number_input("New Stop Loss", value=float(selected_position['stop_loss']))
                if st.button("Update Position"):
                    selected_position['target_price'] = new_target
                    selected_position['stop_loss'] = new_stop
                    st.success("Position updated!")
            
            with col2:
                st.markdown("#### Close Position")
                if st.button("Close Position", type="primary"):
                    closed_pos = self.dashboard.position_manager.close_position(selected_position['id'])
                    if closed_pos:
                        st.session_state.open_positions = [p for p in st.session_state.open_positions if p['id'] != selected_position['id']]
                        st.success(f"Position closed! P&L: ${closed_pos['unrealized_pnl']:.2f}")
            
            with col3:
                st.markdown("#### Position Details")
                st.metric("Current P&L", f"${selected_position['unrealized_pnl']:.2f}")
                st.metric("P&L %", f"{selected_position['pnl_percentage']:.2f}%")
                st.metric("Time Open", str(datetime.now() - selected_position['entry_time']).split('.')[0])
            
            # P&L chart
            st.markdown("### 📊 Portfolio P&L")
            pnl_data = [p['unrealized_pnl'] for p in positions]
            tickers = [p['ticker'] for p in positions]
            
            fig = px.bar(x=tickers, y=pnl_data, title="Position P&L")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No open positions.")
    
    def generate_open_positions(self):
        """Generate sample open positions"""
        positions = []
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        
        for i, symbol in enumerate(symbols):
            entry_price = self.dashboard.get_realistic_price(symbol)
            current_price = entry_price * np.random.uniform(0.95, 1.05)
            
            position = {
                'id': f"POS_{i+1:04d}",
                'ticker': symbol,
                'entry_price': round(entry_price, 2),
                'current_price': round(current_price, 2),
                'target_price': round(entry_price * 1.1, 2),
                'stop_loss': round(entry_price * 0.95, 2),
                'quantity': 100,
                'position_size': round(entry_price * 100, 2),
                'entry_time': datetime.now() - timedelta(hours=np.random.randint(1, 48)),
                'status': 'Open',
                'unrealized_pnl': round((current_price - entry_price) * 100, 2),
                'pnl_percentage': round(((current_price - entry_price) / entry_price) * 100, 2)
            }
            positions.append(position)
        
        return positions
    
    def show_pending_positions_screen(self):
        """Show pending positions screen"""
        st.markdown("## ⏳ Pending Positions")
        
        # Generate pending positions
        if not st.session_state.pending_positions:
            st.session_state.pending_positions = self.generate_pending_positions()
        
        positions = st.session_state.pending_positions
        
        if positions:
            df = pd.DataFrame(positions)
            st.dataframe(
                df[[
                    'ticker', 'order_type', 'quantity', 'price', 'status', 'created_time'
                ]],
                use_container_width=True,
                hide_index=True
            )
            
            # Pending order management
            st.markdown("### 🎛️ Pending Order Management")
            selected_order = st.selectbox(
                "Select Order to Manage",
                options=positions,
                format_func=lambda x: f"{x['ticker']} - {x['order_type']} @ {x['price']}"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Cancel Order"):
                    st.session_state.pending_positions = [p for p in st.session_state.pending_positions if p['id'] != selected_order['id']]
                    st.success("Order cancelled!")
            
            with col2:
                if st.button("Modify Order"):
                    st.info("Order modification feature coming soon!")
        else:
            st.info("No pending positions.")
    
    def generate_pending_positions(self):
        """Generate sample pending positions"""
        positions = []
        symbols = ['BTC', 'ETH', 'GOLD', 'OIL']
        order_types = ['Limit', 'Stop', 'Stop Limit']
        
        for i, symbol in enumerate(symbols):
            position = {
                'id': f"PEND_{i+1:04d}",
                'ticker': symbol,
                'order_type': np.random.choice(order_types),
                'quantity': np.random.randint(10, 100),
                'price': round(self.dashboard.get_realistic_price(symbol), 2),
                'status': 'Pending',
                'created_time': datetime.now() - timedelta(hours=np.random.randint(1, 24))
            }
            positions.append(position)
        
        return positions
    
    def show_account_strategy_screen(self):
        """Show account strategy screen"""
        st.markdown("## 📋 Account Strategy")
        
        strategy = st.session_state.account_strategy
        
        # Time horizon tabs
        tabs = st.tabs(["Daily", "Weekly", "Monthly", "Quarterly", "Yearly", "5-Year"])
        
        with tabs[0]:  # Daily
            self.show_strategy_tab(strategy['daily'], "Daily Strategy")
        
        with tabs[1]:  # Weekly
            self.show_strategy_tab(strategy['weekly'], "Weekly Strategy")
        
        with tabs[2]:  # Monthly
            self.show_strategy_tab(strategy['monthly'], "Monthly Strategy")
        
        with tabs[3]:  # Quarterly
            self.show_strategy_tab(strategy['quarterly'], "Quarterly Strategy")
        
        with tabs[4]:  # Yearly
            self.show_strategy_tab(strategy['yearly'], "Yearly Strategy")
        
        with tabs[5]:  # 5-Year
            self.show_strategy_tab(strategy['five_year'], "5-Year Strategy")
    
    def show_strategy_tab(self, strategy_data, title):
        """Show strategy tab content"""
        st.markdown(f"### {title}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Target Return", f"{strategy_data['target_return']:.1f}%")
            st.metric("Max Risk", f"{strategy_data['max_risk']:.1%}")
        
        with col2:
            st.metric("Position Limit", strategy_data['position_limit'])
            if 'sector_allocation' in strategy_data:
                st.markdown("#### Sector Allocation")
                for sector, allocation in strategy_data['sector_allocation'].items():
                    st.write(f"{sector}: {allocation:.1%}")
        
        with col3:
            if 'rebalancing_frequency' in strategy_data:
                st.metric("Rebalancing", strategy_data['rebalancing_frequency'])
            if 'sector_rotation' in strategy_data:
                st.metric("Sector Rotation", "Enabled" if strategy_data['sector_rotation'] else "Disabled")
            if 'strategy_review' in strategy_data:
                st.metric("Strategy Review", "Enabled" if strategy_data['strategy_review'] else "Disabled")
    
    def show_market_sentiment_screen(self):
        """Show market sentiment screen"""
        st.markdown("## 🌍 Market Sentiment Analysis")
        
        # Country sentiment
        st.markdown("### 📊 Country Sentiment")
        countries = ['US', 'UK', 'EU', 'China', 'Japan', 'South Korea']
        country_sentiment = {
            'US': 0.75, 'UK': 0.65, 'EU': 0.60, 
            'China': 0.45, 'Japan': 0.70, 'South Korea': 0.80
        }
        
        fig = px.bar(
            x=list(country_sentiment.keys()),
            y=list(country_sentiment.values()),
            title="Market Sentiment by Country",
            labels={'x': 'Country', 'y': 'Sentiment Score'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Industry sentiment
        st.markdown("### 🏭 Industry Sentiment")
        industries = ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer', 'Industrial']
        industry_sentiment = {
            'Technology': 0.85, 'Healthcare': 0.70, 'Finance': 0.60,
            'Energy': 0.45, 'Consumer': 0.75, 'Industrial': 0.65
        }
        
        fig2 = px.bar(
            x=list(industry_sentiment.keys()),
            y=list(industry_sentiment.values()),
            title="Market Sentiment by Industry",
            labels={'x': 'Industry', 'y': 'Sentiment Score'}
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Real-time sentiment indicators
        st.markdown("### 📈 Real-time Sentiment Indicators")
        sentiment_data = st.session_state.market_sentiment
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Overall Sentiment", f"{sentiment_data.get('overall_sentiment', 0):.3f}")
        
        with col2:
            st.metric("News Sentiment", f"{sentiment_data.get('news_sentiment', 0):.3f}")
        
        with col3:
            st.metric("Social Sentiment", f"{sentiment_data.get('social_sentiment', 0):.3f}")
        
        with col4:
            st.metric("Technical Sentiment", f"{sentiment_data.get('technical_sentiment', 0):.3f}")
        
        with col5:
            st.metric("Fundamental Sentiment", f"{sentiment_data.get('fundamental_sentiment', 0):.3f}")
    
    def show_industry_analytics_screen(self):
        """Show industry analytics screen"""
        st.markdown("## 🏭 Industry Analytics")
        
        industry_data = st.session_state.industry_data
        
        # Sector rotation analysis
        st.markdown("### 📊 Sector Rotation Analysis")
        
        industries = list(industry_data.keys())
        rotation_scores = [industry_data[ind]['sector_rotation_score'] for ind in industries]
        
        fig = px.bar(
            x=industries,
            y=rotation_scores,
            title="Sector Rotation Scores",
            labels={'x': 'Industry', 'y': 'Rotation Score'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Industry correlation matrix
        st.markdown("### 🔗 Industry Correlation Matrix")
        
        correlation_data = []
        for ind1 in industries:
            row = []
            for ind2 in industries:
                if ind1 == ind2:
                    row.append(1.0)
                else:
                    row.append(industry_data[ind1]['correlation'].get(ind2, 0))
            correlation_data.append(row)
        
        fig2 = px.imshow(
            correlation_data,
            x=industries,
            y=industries,
            title="Industry Correlation Matrix",
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Industry performance metrics
        st.markdown("### 📈 Industry Performance Metrics")
        
        performance_data = []
        for industry in industries:
            data = industry_data[industry]
            performance_data.append({
                'Industry': industry,
                'Sentiment': data['sentiment'],
                'Performance': data['performance'],
                'Volatility': data['volatility'],
                'Momentum': data['momentum']
            })
        
        df = pd.DataFrame(performance_data)
        st.dataframe(df, use_container_width=True)
    
    def show_top_industries_screen(self):
        """Show top performing industries"""
        st.markdown("## 📈 Top Performing Industries")
        
        industry_data = st.session_state.industry_data
        
        # Sort by performance
        sorted_industries = sorted(
            industry_data.items(),
            key=lambda x: x[1]['performance'],
            reverse=True
        )
        
        industries = [ind[0] for ind in sorted_industries]
        performances = [ind[1]['performance'] for ind in sorted_industries]
        
        fig = px.bar(
            x=industries,
            y=performances,
            title="Top Performing Industries",
            labels={'x': 'Industry', 'y': 'Performance (%)'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top 3 industries details
        st.markdown("### 🏆 Top 3 Industries")
        
        for i, (industry, data) in enumerate(sorted_industries[:3]):
            with st.expander(f"#{i+1} {industry} - {data['performance']:.1f}%"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Sentiment", f"{data['sentiment']:.3f}")
                    st.metric("Volatility", f"{data['volatility']:.3f}")
                
                with col2:
                    st.metric("Momentum", f"{data['momentum']:.3f}")
                    st.metric("Rotation Score", f"{data['sector_rotation_score']:.3f}")
                
                with col3:
                    st.metric("Performance", f"{data['performance']:.1f}%")
    
    def show_worst_industries_screen(self):
        """Show worst performing industries"""
        st.markdown("## 📉 Worst Performing Industries")
        
        industry_data = st.session_state.industry_data
        
        # Sort by performance (ascending)
        sorted_industries = sorted(
            industry_data.items(),
            key=lambda x: x[1]['performance']
        )
        
        industries = [ind[0] for ind in sorted_industries]
        performances = [ind[1]['performance'] for ind in sorted_industries]
        
        fig = px.bar(
            x=industries,
            y=performances,
            title="Worst Performing Industries",
            labels={'x': 'Industry', 'y': 'Performance (%)'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Bottom 3 industries details
        st.markdown("### ⚠️ Bottom 3 Industries")
        
        for i, (industry, data) in enumerate(sorted_industries[:3]):
            with st.expander(f"#{i+1} {industry} - {data['performance']:.1f}%"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Sentiment", f"{data['sentiment']:.3f}")
                    st.metric("Volatility", f"{data['volatility']:.3f}")
                
                with col2:
                    st.metric("Momentum", f"{data['momentum']:.3f}")
                    st.metric("Rotation Score", f"{data['sector_rotation_score']:.3f}")
                
                with col3:
                    st.metric("Performance", f"{data['performance']:.1f}%")
    
    def show_realtime_fundamentals_screen(self):
        """Show real-time fundamentals screen"""
        st.markdown("## 📊 Real-time Fundamentals")
        
        fundamental_data = st.session_state.fundamental_data
        
        # Earnings data
        st.markdown("### 💰 Earnings Data")
        
        earnings_df = pd.DataFrame(fundamental_data['earnings_data']).T
        st.dataframe(earnings_df, use_container_width=True)
        
        # Economic indicators
        st.markdown("### 📈 Economic Indicators")
        
        economic_data = fundamental_data['economic_indicators']
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("GDP Growth", f"{economic_data['gdp_growth']:.1f}%")
        
        with col2:
            st.metric("Inflation Rate", f"{economic_data['inflation_rate']:.1f}%")
        
        with col3:
            st.metric("Unemployment", f"{economic_data['unemployment_rate']:.1f}%")
        
        with col4:
            st.metric("Interest Rate", f"{economic_data['interest_rate']:.2f}%")
        
        with col5:
            st.metric("Consumer Confidence", f"{economic_data['consumer_confidence']:.0f}")
        
        # Company financials
        st.markdown("### 🏢 Company Financials")
        
        financials_df = pd.DataFrame(fundamental_data['company_financials']).T
        st.dataframe(financials_df, use_container_width=True)
    
    def show_technical_analytics_screen(self):
        """Show technical analytics screen"""
        st.markdown("## 🔧 Technical Analytics Insights")
        
        technical_data = st.session_state.technical_data
        
        # Chart patterns
        st.markdown("### 📈 Chart Patterns")
        
        patterns_data = []
        for ticker, patterns in technical_data['chart_patterns'].items():
            for pattern in patterns:
                patterns_data.append({'Ticker': ticker, 'Pattern': pattern})
        
        patterns_df = pd.DataFrame(patterns_data)
        st.dataframe(patterns_df, use_container_width=True)
        
        # Technical indicators
        st.markdown("### 📊 Technical Indicators")
        
        indicators_data = []
        for ticker, indicators in technical_data['technical_indicators'].items():
            indicators_data.append({
                'Ticker': ticker,
                'RSI': indicators['rsi'],
                'MACD': indicators['macd'],
                'Bollinger Position': indicators['bollinger_position']
            })
        
        indicators_df = pd.DataFrame(indicators_data)
        st.dataframe(indicators_df, use_container_width=True)
        
        # Support/Resistance levels
        st.markdown("### 🎯 Support/Resistance Levels")
        
        levels_data = []
        for ticker, levels in technical_data['support_resistance'].items():
            levels_data.append({
                'Ticker': ticker,
                'Support': levels['support'],
                'Resistance': levels['resistance'],
                'Range': levels['resistance'] - levels['support']
            })
        
        levels_df = pd.DataFrame(levels_data)
        st.dataframe(levels_df, use_container_width=True)
        
        # Technical signals chart
        st.markdown("### 📈 Technical Signals")
        
        tickers = list(technical_data['technical_indicators'].keys())
        rsi_values = [technical_data['technical_indicators'][t]['rsi'] for t in tickers]
        
        fig = px.bar(
            x=tickers,
            y=rsi_values,
            title="RSI Values by Ticker",
            labels={'x': 'Ticker', 'y': 'RSI'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def show_model_learning_screen(self):
        """Show model learning screen"""
        st.markdown("## 🤖 Model Learning & Improvement")
        
        model_performance = st.session_state.model_performance
        
        # Model performance comparison
        st.markdown("### 📊 Model Performance Comparison")
        
        models = list(model_performance.keys())
        accuracy = [model_performance[m]['accuracy'] for m in models]
        precision = [model_performance[m]['precision'] for m in models]
        recall = [model_performance[m]['recall'] for m in models]
        f1_scores = [model_performance[m]['f1_score'] for m in models]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Accuracy', x=models, y=accuracy))
        fig.add_trace(go.Bar(name='Precision', x=models, y=precision))
        fig.add_trace(go.Bar(name='Recall', x=models, y=recall))
        fig.add_trace(go.Bar(name='F1 Score', x=models, y=f1_scores))
        
        fig.update_layout(
            title="Model Performance Metrics",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Training progress
        st.markdown("### 📈 Training Progress")
        
        # Show training progress for first model
        first_model = list(model_performance.keys())[0]
        training_progress = model_performance[first_model]['training_progress']
        
        fig2 = px.line(
            x=list(range(1, len(training_progress) + 1)),
            y=training_progress,
            title=f"Training Progress - {first_model}",
            labels={'x': 'Epoch', 'y': 'Loss'}
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Model retraining controls
        st.markdown("### 🔄 Model Retraining Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_model = st.selectbox("Select Model", models)
        
        with col2:
            if st.button("Retrain Model"):
                st.success(f"Retraining {selected_model}...")
        
        with col3:
            if st.button("Evaluate Model"):
                st.info(f"Evaluating {selected_model}...")
        
        # Model details
        st.markdown("### 📋 Model Details")
        
        selected_model_data = model_performance[selected_model]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{selected_model_data['accuracy']:.3f}")
        
        with col2:
            st.metric("Precision", f"{selected_model_data['precision']:.3f}")
        
        with col3:
            st.metric("Recall", f"{selected_model_data['recall']:.3f}")
        
        with col4:
            st.metric("F1 Score", f"{selected_model_data['f1_score']:.3f}")
        
        st.metric("Last Updated", selected_model_data['last_updated'].strftime("%Y-%m-%d %H:%M"))
    
    def show_trading_analytics_screen(self):
        """Show trading analytics screen"""
        st.markdown("## 📊 Trading Analytics")
        
        # Portfolio performance
        st.markdown("### 📈 Portfolio Performance")
        
        # Mock portfolio data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        portfolio_values = [100000 + i * 100 + np.random.normal(0, 500) for i in range(len(dates))]
        
        fig = px.line(
            x=dates,
            y=portfolio_values,
            title="Portfolio Value Over Time",
            labels={'x': 'Date', 'y': 'Portfolio Value ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Trading statistics
        st.markdown("### 📊 Trading Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", 156)
            st.metric("Win Rate", "68.5%")
        
        with col2:
            st.metric("Average Return", "2.3%")
            st.metric("Max Drawdown", "-8.2%")
        
        with col3:
            st.metric("Sharpe Ratio", "1.85")
            st.metric("Sortino Ratio", "2.12")
        
        with col4:
            st.metric("Profit Factor", "1.67")
            st.metric("Recovery Factor", "2.34")
        
        # Asset allocation
        st.markdown("### 🎯 Asset Allocation")
        
        assets = ['Equities', 'Crypto', 'Forex', 'Commodities', 'Bonds']
        allocations = [45, 20, 15, 12, 8]
        
        fig2 = px.pie(
            values=allocations,
            names=assets,
            title="Current Asset Allocation"
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Risk metrics
        st.markdown("### ⚠️ Risk Metrics")
        
        risk_data = {
            'VaR (95%)': '-2.1%',
            'CVaR (95%)': '-3.2%',
            'Volatility': '12.5%',
            'Beta': '0.85',
            'Correlation': '0.72'
        }
        
        for metric, value in risk_data.items():
            st.metric(metric, value)
