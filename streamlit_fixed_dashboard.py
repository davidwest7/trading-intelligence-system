"""
Fixed Streamlit Dashboard with Unique Button Keys
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path
sys.path.append('.')

class FixedTradingDashboard:
    """Fixed trading dashboard with unique button keys"""
    
    def __init__(self):
        self.init_session_state()
        
    def init_session_state(self):
        """Initialize Streamlit session state"""
        if 'current_screen' not in st.session_state:
            st.session_state.current_screen = 'Top Opportunities'
        if 'opportunities_data' not in st.session_state:
            st.session_state.opportunities_data = []
        if 'open_positions' not in st.session_state:
            st.session_state.open_positions = []
        if 'pending_positions' not in st.session_state:
            st.session_state.pending_positions = []
    
    def generate_realistic_opportunities(self):
        """Generate realistic opportunities"""
        opportunities = []
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'BTC', 'ETH', 'EUR/USD', 'GOLD', 'OIL']
        
        for i in range(25):
            symbol = np.random.choice(symbols)
            current_price = self.get_realistic_price(symbol)
            target_price = current_price * np.random.uniform(0.95, 1.15)
            stop_loss = current_price * np.random.uniform(0.85, 0.98)
            
            opportunity = {
                'id': f"OPP_{i+1:04d}",
                'ticker': symbol,
                'current_price': round(current_price, 2),
                'target_price': round(target_price, 2),
                'stop_loss': round(stop_loss, 2),
                'market_order_type': np.random.choice(['Market', 'Limit', 'Stop', 'Stop Limit']),
                'risk': round(np.random.uniform(0.01, 0.05), 3),
                'position_size': round(np.random.uniform(1000, 10000), 0),
                'reasoning': np.random.choice(['Technical Breakout', 'Fundamental Value', 'Sentiment Analysis', 'ML Prediction', 'Arbitrage']),
                'confidence': round(np.random.uniform(0.6, 0.9), 2),
                'expected_return': round((target_price - current_price) / current_price * 100, 2),
                'timestamp': datetime.now() - timedelta(hours=np.random.randint(0, 24)),
                'status': np.random.choice(['Open', 'Pending', 'Closed'], p=[0.4, 0.3, 0.3]),
                'agent_type': np.random.choice(['Technical', 'Sentiment', 'ML', 'Fundamental']),
                'time_horizon': np.random.choice(['short_term', 'medium_term', 'long_term'])
            }
            opportunities.append(opportunity)
        
        return opportunities
    
    def get_realistic_price(self, symbol):
        """Get realistic price for symbol"""
        base_prices = {
            'AAPL': 180, 'MSFT': 350, 'GOOGL': 140, 'TSLA': 250, 'NVDA': 450,
            'BTC': 45000, 'ETH': 2800, 'EUR/USD': 1.08, 'GOLD': 2000, 'OIL': 75
        }
        base_price = base_prices.get(symbol, 100)
        return base_price * np.random.uniform(0.95, 1.05)
    
    def create_navigation_menu(self):
        """Create navigation menu with unique keys"""
        st.markdown("### 📊 Trading Analytics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("🎯 Top Opportunities", key="nav_top_opps"):
                st.session_state.current_screen = 'Top Opportunities'
        
        with col2:
            if st.button("📈 Open Positions", key="nav_open_pos"):
                st.session_state.current_screen = 'Open Positions'
        
        with col3:
            if st.button("⏳ Pending Positions", key="nav_pending_pos"):
                st.session_state.current_screen = 'Pending Positions'
        
        with col4:
            if st.button("📋 Account Strategy", key="nav_account_strat"):
                st.session_state.current_screen = 'Account Strategy'
        
        with col5:
            if st.button("📊 Trading Analytics", key="nav_trading_analytics"):
                st.session_state.current_screen = 'Trading Analytics'
        
        st.markdown("### 🌍 Market Analytics")
        col6, col7, col8, col9, col10 = st.columns(5)
        
        with col6:
            if st.button("�� Market Sentiment", key="nav_market_sentiment"):
                st.session_state.current_screen = 'Market Sentiment'
        
        with col7:
            if st.button("🏭 Industry Analytics", key="nav_industry_analytics"):
                st.session_state.current_screen = 'Industry Analytics'
        
        with col8:
            if st.button("📈 Top Industries", key="nav_top_industries"):
                st.session_state.current_screen = 'Top Industries'
        
        with col9:
            if st.button("📉 Worst Industries", key="nav_worst_industries"):
                st.session_state.current_screen = 'Worst Industries'
        
        with col10:
            if st.button("📊 Real-time Fundamentals", key="nav_realtime_fundamentals"):
                st.session_state.current_screen = 'Real-time Fundamentals'
        
        st.markdown("### 🔧 Technical & ML Analytics")
        col11, col12 = st.columns(2)
        
        with col11:
            if st.button("🔧 Technical Analytics", key="nav_technical_analytics"):
                st.session_state.current_screen = 'Technical Analytics'
        
        with col12:
            if st.button("🤖 Model Learning", key="nav_model_learning"):
                st.session_state.current_screen = 'Model Learning'
    
    def show_top_opportunities_screen(self):
        """Show top opportunities screen"""
        st.markdown("## 🎯 Top Opportunity Positions")
        
        if not st.session_state.opportunities_data:
            st.session_state.opportunities_data = self.generate_realistic_opportunities()
        
        opportunities = st.session_state.opportunities_data
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            status_filter = st.selectbox("Status Filter", ['All'] + list(set([o['status'] for o in opportunities])))
        with col2:
            confidence_filter = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.1)
        with col3:
            return_filter = st.slider("Min Expected Return (%)", -20.0, 50.0, -20.0, 1.0)
        
        # Apply filters
        filtered_opportunities = opportunities.copy()
        if status_filter != 'All':
            filtered_opportunities = [o for o in filtered_opportunities if o['status'] == status_filter]
        filtered_opportunities = [o for o in filtered_opportunities if o['confidence'] >= confidence_filter]
        filtered_opportunities = [o for o in filtered_opportunities if o['expected_return'] >= return_filter]
        
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
            
            # Action buttons
            st.markdown("### 🎯 Opportunity Actions")
            selected_opportunity = st.selectbox(
                "Select Opportunity for Actions",
                options=filtered_opportunities,
                format_func=lambda x: f"{x['ticker']} - {x['reasoning']} (Confidence: {x['confidence']:.2%})"
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔍 Show Decision Reasoning", key="show_reasoning_btn"):
                    self.show_reasoning_visualization(selected_opportunity)
            
            with col2:
                if st.button("📊 Execute Position", key="execute_position_btn"):
                    self.execute_position(selected_opportunity)
            
            with col3:
                if st.button("📈 Add to Watchlist", key="add_watchlist_btn"):
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
        """Show open positions screen"""
        st.markdown("## 📈 Open Positions")
        
        if not st.session_state.open_positions:
            st.session_state.open_positions = self.generate_open_positions()
        
        positions = st.session_state.open_positions
        
        if positions:
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
                if st.button("Update Position", key="update_position_btn"):
                    selected_position['target_price'] = new_target
                    selected_position['stop_loss'] = new_stop
                    st.success("Position updated!")
            
            with col2:
                st.markdown("#### Close Position")
                if st.button("Close Position", key="close_position_btn", type="primary"):
                    st.session_state.open_positions = [p for p in st.session_state.open_positions if p['id'] != selected_position['id']]
                    st.success(f"Position closed! P&L: ${selected_position['unrealized_pnl']:.2f}")
            
            with col3:
                st.markdown("#### Position Details")
                st.metric("Current P&L", f"${selected_position['unrealized_pnl']:.2f}")
                st.metric("P&L %", f"{selected_position['pnl_percentage']:.2f}%")
                st.metric("Time Open", str(datetime.now() - selected_position['entry_time']).split('.')[0])
        else:
            st.info("No open positions.")
    
    def generate_open_positions(self):
        """Generate sample open positions"""
        positions = []
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        
        for i, symbol in enumerate(symbols):
            entry_price = self.get_realistic_price(symbol)
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
    
    def run_dashboard(self):
        """Run the dashboard"""
        st.set_page_config(
            page_title="Trading Intelligence System - Fixed",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🚀 Trading Intelligence System")
        st.markdown("### World-Class Multi-Asset Trading Platform - Fixed")
        
        # Create navigation menu
        self.create_navigation_menu()
        
        # Display current screen
        current_screen = st.session_state.current_screen
        
        if current_screen == 'Top Opportunities':
            self.show_top_opportunities_screen()
        elif current_screen == 'Open Positions':
            self.show_open_positions_screen()
        elif current_screen == 'Market Sentiment':
            self.show_market_sentiment_screen()
        else:
            st.markdown(f"## {current_screen}")
            st.info(f"Screen '{current_screen}' is under development. This will show {current_screen.lower()} data and analytics.")


def main():
    """Main function to run the dashboard"""
    dashboard = FixedTradingDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()
